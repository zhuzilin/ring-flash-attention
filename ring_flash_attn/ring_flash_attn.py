import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .comm import send_recv_kv


def ring_flash_attn_forward(
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
    rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)

    local_q = local_q.contiguous()
    local_k = local_k.contiguous()
    local_v = local_v.contiguous()

    assert local_q.shape[-1] % 8 == 0, "unpadded head size not supported"
    if softmax_scale is None:
        softmax_scale = local_q.shape[-1] ** (-0.5)

    out = None
    lse = None
    next_k = local_k
    next_v = local_v
    next_kv_rank = rank
    reqs = None
    for step in range(world_size):
        if reqs is not None:
            for req in reqs:
                req.wait()
        k, v, kv_rank = next_k, next_v, next_kv_rank
        next_k, next_v, next_kv_rank, reqs = send_recv_kv(
            process_group, local_k, local_v, step + 1, rank, world_size, causal
        )
        if k is not None:
            assert not causal or kv_rank <= rank
            local_causal = causal and kv_rank == rank
            block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
                local_q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal=local_causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )
            block_out = block_out.to(torch.float32)
            block_lse = block_lse.transpose(1, 2).unsqueeze(dim=-1)
            if out is None:
                out = block_out
                lse = block_lse
            else:
                new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
                out = (
                    torch.exp(lse - new_lse) * out
                    + torch.exp(block_lse - new_lse) * block_out
                )
                lse = new_lse

    out = out.to(torch.bfloat16)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def ring_flash_attn_backward(
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
    rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)

    local_dout = local_dout.contiguous()
    local_q = local_q.contiguous()
    local_k = local_k.contiguous()
    local_v = local_v.contiguous()
    local_out = local_out.contiguous()

    local_dq = None
    local_dk = None
    local_dv = None

    next_k = local_k
    next_v = local_v
    next_kv_rank = rank
    reqs = None

    block_dq_buffer = torch.empty(
        local_q.shape, dtype=local_q.dtype, device=local_q.device
    )
    block_dk_buffer = torch.empty(
        local_k.shape, dtype=local_k.dtype, device=local_k.device
    )
    block_dv_buffer = torch.empty(
        local_v.shape, dtype=local_v.dtype, device=local_v.device
    )

    remote_dk = None
    remote_dv = None
    grad_reqs = None
    for step in range(world_size):
        if reqs is not None:
            for req in reqs:
                req.wait()
        k, v, kv_rank = next_k, next_v, next_kv_rank
        next_k, next_v, next_kv_rank, reqs = send_recv_kv(
            process_group, local_k, local_v, step + 1, rank, world_size, causal
        )
        if k is not None:
            assert not causal or kv_rank <= rank
            local_causal = causal and kv_rank == rank

            _flash_attn_backward(
                local_dout,
                local_q,
                k,
                v,
                local_out,
                softmax_lse,
                block_dq_buffer,
                block_dk_buffer,
                block_dv_buffer,
                dropout_p,
                softmax_scale,
                local_causal,
                window_size,
                alibi_slopes,
                deterministic,
                rng_state=None,
            )

            block_dq = block_dq_buffer.to(torch.float32)
            block_dk = block_dk_buffer.to(torch.float32)
            block_dv = block_dv_buffer.to(torch.float32)

            if local_dq is None:
                local_dq = block_dq
                local_dk, local_dv = block_dk, block_dv
            else:
                local_dq += block_dq

        if grad_reqs is not None:
            for req in grad_reqs:
                req.wait()
            if remote_dk is not None:
                local_dk += remote_dk
                local_dv += remote_dv

        remote_dk, remote_dv, _, grad_reqs = send_recv_kv(
            process_group,
            block_dk,
            block_dv,
            step,
            rank,
            world_size,
            causal,
            is_grad=True,
        )

    if grad_reqs is not None:
        for req in grad_reqs:
            req.wait()
        if remote_dk is not None:
            local_dk += remote_dk
            local_dv += remote_dv

    return local_dq, local_dk, local_dv


class RingFlashAttnQKVPackedFunc(torch.autograd.Function):
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
        out, softmax_lse = ring_flash_attn_forward(
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
        dq, dk, dv = ring_flash_attn_backward(
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


def ring_flash_attn_qkvpacked_func(
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
    return RingFlashAttnQKVPackedFunc.apply(
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
