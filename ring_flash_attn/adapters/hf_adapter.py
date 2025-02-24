import os
import inspect
from typing import Optional

import torch
import torch.distributed as dist
import transformers
import transformers.modeling_flash_attention_utils
from transformers.modeling_flash_attention_utils import (
    _flash_supports_window_size,
    is_flash_attn_greater_or_equal,
)
from ..llama3_flash_attn_varlen import (
    llama3_flash_attn_varlen_func,
    llama3_flash_attn_prepare_cu_seqlens,
)


DATA_PARAMS = {}
RING_ATTN_SWITCH = True


def check_params(f1, f2):
    return len(inspect.signature(f1).parameters) == len(
        inspect.signature(f2).parameters
    )


def update_ring_flash_attn_params(
    cu_seqlens: torch.Tensor, process_group: dist.ProcessGroup
):
    world_size = dist.get_world_size(group=process_group)
    rank = dist.get_rank(group=process_group)
    (
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        local_k_slice,
    ) = llama3_flash_attn_prepare_cu_seqlens(cu_seqlens, True, rank, world_size)
    DATA_PARAMS.update(
        {
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "local_k_slice": local_k_slice,
        }
    )


def use_ring_attn(flag):
    global RING_ATTN_SWITCH
    RING_ATTN_SWITCH = flag


def create_ring_flash_attention_forward(
    process_group: dist.ProcessGroup, heads_k_stride: int
):
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
            _flash_supports_window_size
            and sliding_window is not None
            and key_states.shape[1] > sliding_window
        )
        flash_kwargs = (
            {"window_size": (sliding_window, sliding_window)}
            if use_sliding_windows
            else {}
        )

        if is_flash_attn_greater_or_equal("2.4.1"):
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

        attn_output = llama3_flash_attn_varlen_func(
            query_states.squeeze(dim=0),
            key_states.squeeze(dim=0),
            value_states.squeeze(dim=0),
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

    return _flash_attention_forward


def substitute_hf_flash_attn(process_group: dist.ProcessGroup, heads_k_stride: int):
    try:
        # substitute flash attn
        old_flash_attention_forward = (
            transformers.modeling_flash_attention_utils._flash_attention_forward
        )
        new_flash_attention_forward = create_ring_flash_attention_forward(
            process_group, heads_k_stride
        )
        assert check_params(old_flash_attention_forward, new_flash_attention_forward)
        transformers.modeling_flash_attention_utils._flash_attention_forward = (
            lambda *args, **kwargs: (
                new_flash_attention_forward(*args, **kwargs)
                if RING_ATTN_SWITCH
                else old_flash_attention_forward(*args, **kwargs)
            )
        )
    except:
        raise ValueError(
            f"The current transformer version {transformers.__version__} is not supported. "
            "please use pip install -U transformers to upgrade to the latest version. "
            "If the code failed with the latest version, "
            "please file an issue to https://github.com/zhuzilin/ring-flash-attention/issues"
        )
