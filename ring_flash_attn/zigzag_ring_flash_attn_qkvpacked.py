import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .utils import send_recv_kv, update_out_and_lse


def get_zigzag_rank(rank, world_size):
    rank0, rank1 = rank, 2 * world_size - 1 - rank
    return rank0, rank1


def zigzag_ring_flash_attn_forward(
    process_group,
    local_q,
    local_k,
    local_v,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)
    rank0, rank1 = get_zigzag_rank(rank, world_size)

    local_q = local_q.contiguous()
    local_k = local_k.contiguous()
    local_v = local_v.contiguous()

    local_q1 = local_q.chunk(2, dim=1)[1].contiguous()
    seq_len_per_device = local_q.shape[1]

    assert local_q.shape[-1] % 8 == 0, "unpadded head size not supported"

    out = None
    lse = None
    next_k = local_k
    next_v = local_v
    next_kv_rank = rank
    reqs = None

    def forward(q, k, v, causal):
        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            return_softmax=True and dropout_p > 0,
        )
        return block_out, block_lse

    for step in range(world_size):
        if reqs is not None:
            for req in reqs:
                req.wait()
        k, v, kv_rank = next_k, next_v, next_kv_rank
        if step + 1 < world_size:
            next_k, next_v, next_kv_rank, reqs = send_recv_kv(
                process_group, local_k, local_v, step + 1, causal=False
            )

        if step == 0:
            block_out, block_lse = forward(local_q, local_k, local_v, causal=True)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            kv_rank0, kv_rank1 = get_zigzag_rank(kv_rank, world_size)
            k0, v0 = None, None
            assert kv_rank1 > rank0
            if kv_rank0 < rank0:
                assert kv_rank1 > rank1
                k0, v0 = k.chunk(2, dim=1)[0], v.chunk(2, dim=1)[0]
                block_out, block_lse = forward(local_q, k0, v0, causal=False)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            else:
                assert kv_rank1 < rank1
                block_out, block_lse = forward(local_q1, k, v, causal=False)
                out, lse = update_out_and_lse(
                    out,
                    lse,
                    block_out,
                    block_lse,
                    slice_=(slice(None), slice(seq_len_per_device // 2, None)),
                )

    out = out.to(local_q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def zigzag_ring_flash_attn_backward(
    process_group,
    local_dout,
    local_q,
    local_k,
    local_v,
    local_out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)
    rank0, rank1 = get_zigzag_rank(rank, world_size)

    local_dout = local_dout.contiguous()
    local_q = local_q.contiguous()
    local_k = local_k.contiguous()
    local_v = local_v.contiguous()
    local_out = local_out.contiguous()

    local_dout1 = local_dout.chunk(2, dim=1)[1].contiguous()
    local_q1 = local_q.chunk(2, dim=1)[1].contiguous()
    local_out1 = local_out.chunk(2, dim=1)[1].contiguous()
    softmax_lse1 = softmax_lse.chunk(2, dim=2)[1].contiguous()
    seq_len_per_device = local_q.shape[1]

    def backward(dout, q, k, v, out, softmax_lse, causal):
        # repeatly allocating buffer may be slow...
        dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
        dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq_buffer,
            dk_buffer,
            dv_buffer,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            rng_state=None,
        )
        dq = dq_buffer.to(torch.float32)
        dk = dk_buffer.to(torch.float32)
        dv = dv_buffer.to(torch.float32)
        return dq, dk, dv

    local_dq = None
    local_dk = None
    local_dv = None

    next_k = local_k
    next_v = local_v
    next_kv_rank = rank
    reqs = None

    remote_dk = None
    remote_dv = None
    grad_reqs = None
    for step in range(world_size):
        if reqs is not None:
            for req in reqs:
                req.wait()
        k, v, kv_rank = next_k, next_v, next_kv_rank
        if step + 1 < world_size:
            next_k, next_v, next_kv_rank, reqs = send_recv_kv(
                process_group, local_k, local_v, step + 1, causal=False
            )

        if step == 0:
            dq, dk, dv = backward(
                local_dout,
                local_q,
                local_k,
                local_v,
                local_out,
                softmax_lse,
                causal=True,
            )
            local_dq, local_dk, local_dv = dq, dk, dv
        else:
            kv_rank0, kv_rank1 = get_zigzag_rank(kv_rank, world_size)
            k0, v0 = None, None
            assert kv_rank1 > rank0
            if kv_rank0 < rank0:
                assert kv_rank1 > rank1
                k0, v0 = k.chunk(2, dim=1)[0], v.chunk(2, dim=1)[0]
                dq, dk0, dv0 = backward(
                    local_dout, local_q, k0, v0, local_out, softmax_lse, causal=False
                )
                local_dq += dq
                dk = torch.zeros_like(k, dtype=torch.float32)
                dv = torch.zeros_like(v, dtype=torch.float32)
                dk[:, : seq_len_per_device // 2] = dk0
                dv[:, : seq_len_per_device // 2] = dv0
            else:
                assert kv_rank1 < rank1
                dq1, dk, dv = backward(
                    local_dout1, local_q1, k, v, local_out1, softmax_lse1, causal=False
                )
                local_dq[:, seq_len_per_device // 2 :] += dq1

        if grad_reqs is not None:
            for req in grad_reqs:
                req.wait()
            local_dk += remote_dk
            local_dv += remote_dv

        remote_dk, remote_dv, _, grad_reqs = send_recv_kv(
            process_group,
            dk,
            dv,
            step,
            causal=False,
            is_grad=True,
        )

    for req in grad_reqs:
        req.wait()
    local_dk += remote_dk
    local_dv += remote_dv

    return local_dq, local_dk, local_dv


class ZigZagRingFlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        q = qkv[:, :, 0].contiguous()
        k = qkv[:, :, 1].contiguous()
        v = qkv[:, :, 2].contiguous()
        out, softmax_lse = zigzag_ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = zigzag_ring_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        dqkv = torch.stack([dq, dk, dv], dim=2)
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None


def zigzag_ring_flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnQKVPackedFunc.apply(
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )
