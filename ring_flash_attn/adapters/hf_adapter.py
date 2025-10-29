import os
import inspect
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import transformers
import transformers.modeling_flash_attention_utils
try:
    from transformers.modeling_flash_attention_utils import (
        is_flash_attn_greater_or_equal_2_10,
    )
except ImportError:
    # transformers <= 4.53.x
    from transformers.modeling_flash_attention_utils import (
        is_flash_attn_greater_or_equal_2_10,
    )

from ..llama3_flash_attn_varlen import (
    llama3_flash_attn_varlen_func,
    llama3_flash_attn_prepare_cu_seqlens,
)
from ..zigzag_llama3_flash_attn_varlen import (
    zigzag_llama3_flash_attn_varlen_func,
)

try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except:
    ALL_ATTENTION_FUNCTIONS = None


DATA_PARAMS = {}
RING_ATTN_SWITCH = True
USE_ZIGZAG_LLAMA3 = False  # Toggle between llama3 and zigzag_llama3


def check_params(f1, f2):
    return len(inspect.signature(f1).parameters) == len(
        inspect.signature(f2).parameters
    )


def update_ring_flash_attn_params(
    cu_seqlens: torch.Tensor, process_group: dist.ProcessGroup
):
    world_size = dist.get_world_size(group=process_group)
    rank = dist.get_rank(group=process_group)

    # Calculate ORIGINAL max_seqlen (needed for zigzag_llama3)
    original_max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    (
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        local_k_slice,
    ) = llama3_flash_attn_prepare_cu_seqlens(cu_seqlens, True, rank, world_size)

    # For zigzag_llama3, prepare local cu_seqlens for Q
    if USE_ZIGZAG_LLAMA3:
        local_cu_seqlens_q = []
        offset = 0
        for i in range(len(cu_seqlens) - 1):
            seq_len = (cu_seqlens[i+1] - cu_seqlens[i]).item()
            local_seq_len = seq_len // world_size  # Each rank gets 2 chunks
            local_cu_seqlens_q.append(offset)
            offset += local_seq_len
        local_cu_seqlens_q.append(offset)
        local_cu_seqlens_q_tensor = torch.tensor(
            local_cu_seqlens_q, dtype=torch.int32, device=cu_seqlens.device
        )
    else:
        local_cu_seqlens_q_tensor = cu_seqlens_q

    DATA_PARAMS.update(
        {
            "cu_seqlens_q": local_cu_seqlens_q_tensor,
            "cu_seqlens_k": cu_seqlens if USE_ZIGZAG_LLAMA3 else cu_seqlens_k,  # Use global cu_seqlens for K in zigzag
            "max_seqlen_q": original_max_seqlen if USE_ZIGZAG_LLAMA3 else max_seqlen_q,
            "max_seqlen_k": original_max_seqlen if USE_ZIGZAG_LLAMA3 else max_seqlen_k,
            "local_k_slice": local_k_slice,
            "global_cu_seqlens": cu_seqlens,  # Store original for zigzag extraction
            "rank": rank,
            "world_size": world_size,
        }
    )


def use_ring_attn(flag):
    global RING_ATTN_SWITCH
    RING_ATTN_SWITCH = flag


def use_zigzag_llama3(flag):
    """Toggle between llama3 and zigzag_llama3 implementations."""
    global USE_ZIGZAG_LLAMA3
    USE_ZIGZAG_LLAMA3 = flag


def extract_local_zigzag(value, cu_seqlens, rank, world_size):
    """Extract local zigzag-distributed portion for this rank.

    Each sequence is split into 2*world_size chunks.
    GPU rank r gets: chunk[r] + chunk[2*world_size - 1 - r]

    This creates the interleaved zigzag pattern where each GPU
    gets tokens from both beginning and end of sequences.
    """
    local_values = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        seq_len = end - start
        # Split sequence into 2*world_size chunks
        chunk_size = seq_len // (2 * world_size)

        # Get early chunk (rank r gets chunk r)
        early_start = start + rank * chunk_size
        early_end = early_start + chunk_size
        early_chunk = value[early_start:early_end]

        # Get late chunk (rank r gets chunk 2*world_size-1-r)
        late_idx = 2 * world_size - 1 - rank
        late_start = start + late_idx * chunk_size
        late_end = late_start + chunk_size
        late_chunk = value[late_start:late_end]

        local_values.extend([early_chunk, late_chunk])

    return torch.cat(local_values, dim=0).contiguous()


def create_ring_flash_attention_forward(
    process_group: dist.ProcessGroup, heads_k_stride: int
):
    # before transformers 4.47
    def _flash_attention_forward(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
        is_causal: bool,
        dropout: float = 0.0,
        position_ids: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        sliding_window: Optional[int] = None,
        use_top_left_mask: bool = False,
        softcap: Optional[float] = None,
        deterministic: bool = None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_top_left_mask (`bool`, defaults to `False`):
                flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
            softcap (`float`, *optional*):
                Softcap for the attention logits, used e.g. in gemma2.
            deterministic (`bool`, *optional*):
                Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
        """
        if not use_top_left_mask:
            causal = is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__.
            causal = is_causal and query_length != 1

        # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
        use_sliding_windows = (
            sliding_window is not None
            and key_states.shape[1] > sliding_window
        )
        flash_kwargs = (
            {"window_size": (sliding_window, sliding_window)}
            if use_sliding_windows
            else {}
        )

        if is_flash_attn_greater_or_equal_2_10:
            if deterministic is None:
                deterministic = (
                    os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
                )
        flash_kwargs["deterministic"] = deterministic
        assert (
            softcap is None
        ), "llama3_flash_attn_varlen_func does not support softcap yet."
        # flash_kwargs["softcap"] = softcap
        flash_kwargs["group"] = process_group

        # not sure why attention_mask can be not None...
        assert causal, "only causal attention is supported yet."
        batch_size = query_states.size(0)
        assert batch_size == 1, "varlen data should be processed in advance."

        # Extract inputs (remove batch dimension)
        q = query_states.squeeze(dim=0)
        k = key_states.squeeze(dim=0)
        v = value_states.squeeze(dim=0)

        # For zigzag_llama3, extract local zigzag portions
        if USE_ZIGZAG_LLAMA3:
            q = extract_local_zigzag(
                q, DATA_PARAMS["global_cu_seqlens"],
                DATA_PARAMS["rank"], DATA_PARAMS["world_size"]
            )
            k = extract_local_zigzag(
                k, DATA_PARAMS["global_cu_seqlens"],
                DATA_PARAMS["rank"], DATA_PARAMS["world_size"]
            )
            v = extract_local_zigzag(
                v, DATA_PARAMS["global_cu_seqlens"],
                DATA_PARAMS["rank"], DATA_PARAMS["world_size"]
            )

            attn_output = zigzag_llama3_flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=DATA_PARAMS["cu_seqlens_q"],
                cu_seqlens_k=DATA_PARAMS["cu_seqlens_k"],
                max_seqlen_q=DATA_PARAMS["max_seqlen_q"],
                max_seqlen_k=DATA_PARAMS["max_seqlen_k"],
                heads_k_stride=heads_k_stride,
                local_k_slice=DATA_PARAMS["local_k_slice"],
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
                n_chunks=2,
                **flash_kwargs,
            )
        else:
            attn_output = llama3_flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=DATA_PARAMS["cu_seqlens_q"],
                cu_seqlens_k=DATA_PARAMS["cu_seqlens_k"],
                max_seqlen_q=DATA_PARAMS["max_seqlen_q"],
                max_seqlen_k=DATA_PARAMS["max_seqlen_k"],
                heads_k_stride=heads_k_stride,
                local_k_slice=DATA_PARAMS["local_k_slice"],
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
                **flash_kwargs,
            )

        attn_output = attn_output.unsqueeze(dim=0)

        return attn_output

    # transformers 4.47
    def _flash_attention_forward_v1(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
        is_causal: bool,
        dropout: float = 0.0,
        position_ids: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        sliding_window: Optional[int] = None,
        use_top_left_mask: bool = False,
        softcap: Optional[float] = None,
        deterministic: bool = None,
        cu_seq_lens_q: Optional[torch.LongTensor] = None,
        cu_seq_lens_k: Optional[torch.LongTensor] = None,
        max_length_q: Optional[int] = None,
        max_length_k: Optional[int] = None,
        target_dtype: Optional[torch.dtype] = None,
    ):
        return _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            is_causal,
            dropout,
            position_ids,
            softmax_scale,
            sliding_window,
            use_top_left_mask,
            softcap,
            deterministic,
        )

    def _flash_attention_forward_v2(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
        is_causal: bool,
        dropout: float = 0.0,
        position_ids: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        sliding_window: Optional[int] = None,
        use_top_left_mask: bool = False,
        softcap: Optional[float] = None,
        deterministic: bool = None,
        cu_seq_lens_q: Optional[torch.LongTensor] = None,
        cu_seq_lens_k: Optional[torch.LongTensor] = None,
        max_length_q: Optional[int] = None,
        max_length_k: Optional[int] = None,
        target_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        return _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            is_causal,
            dropout,
            position_ids,
            softmax_scale,
            sliding_window,
            use_top_left_mask,
            softcap,
            deterministic,
        )

    # transformers 4.53.0+
    def _flash_attention_forward_v3(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
        is_causal: bool,
        dropout: float = 0.0,
        position_ids: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        sliding_window: Optional[int] = None,
        use_top_left_mask: bool = False,
        softcap: Optional[float] = None,
        deterministic: bool = None,
        cu_seq_lens_q: Optional[torch.LongTensor] = None,
        cu_seq_lens_k: Optional[torch.LongTensor] = None,
        max_length_q: Optional[int] = None,
        max_length_k: Optional[int] = None,
        target_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
        **kwargs,
    ):
        return _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            is_causal,
            dropout,
            position_ids,
            softmax_scale,
            sliding_window,
            use_top_left_mask,
            softcap,
            deterministic,
        )

    return [
        _flash_attention_forward,
        _flash_attention_forward_v1,
        _flash_attention_forward_v2,
        _flash_attention_forward_v3,
    ]


_use_top_left_mask = False


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # This is before the transpose
    seq_len = query.shape[2]

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(
                layer
                for layer in module.modules()
                if isinstance(layer, torch.nn.Linear)
            ).weight.dtype

    # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)

    # Cache original dtype and convert QKV to target dtype
    original_dtype = query.dtype
    query = query.to(target_dtype)
    key = key.to(target_dtype)
    value = value.to(target_dtype)

    attn_output = transformers.modeling_flash_attention_utils._flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=module.is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=_use_top_left_mask,
        target_dtype=target_dtype,
        **kwargs,
    )
    attn_output = attn_output.to(original_dtype)

    return attn_output, None


def substitute_hf_flash_attn(process_group: dist.ProcessGroup, heads_k_stride: int):
    try:
        # substitute flash attn
        old_flash_attention_forward = (
            transformers.modeling_flash_attention_utils._flash_attention_forward
        )
        new_flash_attention_forward_list = create_ring_flash_attention_forward(
            process_group, heads_k_stride
        )
        for new_flash_attention_forward in new_flash_attention_forward_list:
            if check_params(old_flash_attention_forward, new_flash_attention_forward):
                transformers.modeling_flash_attention_utils._flash_attention_forward = (
                    lambda *args, **kwargs: (
                        new_flash_attention_forward(*args, **kwargs)
                        if RING_ATTN_SWITCH
                        else old_flash_attention_forward(*args, **kwargs)
                    )
                )
                break
        else:
            assert (
                False
            ), "The signature of the new flash attention forward function does not match the old one."
    except:
        raise ValueError(
            f"The current transformer version {transformers.__version__} is not supported. "
            "please use pip install -U transformers to upgrade to the latest version. "
            "If the code failed with the latest version, "
            "please file an issue to https://github.com/zhuzilin/ring-flash-attention/issues"
        )

    if ALL_ATTENTION_FUNCTIONS is not None:
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
