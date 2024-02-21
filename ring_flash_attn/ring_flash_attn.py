import torch
import torch.distributed as dist
from .raw_flash_attn import raw_flash_attn_forward, raw_flash_attn_backward


def send_recv_kv(
    process_group, local_k, local_v, step, rank, world_size, causal, is_grad=False
):
    if step == 0:
        assert is_grad
        return None, None, rank, None
    send_rank = (rank + step) % world_size
    recv_rank = (rank - step) % world_size

    need_to_recv = not (causal and step > rank)
    need_to_send = not (causal and step + rank >= world_size)

    if is_grad:
        # when sending grad, we need to reverse send and recv
        send_rank, recv_rank = recv_rank, send_rank
        need_to_recv, need_to_send = need_to_send, need_to_recv

    ops = []
    if need_to_recv:
        remote_k = torch.empty_like(local_k)
        remote_v = torch.empty_like(local_v)
        recv_k = dist.P2POp(dist.irecv, remote_k, recv_rank, group=process_group)
        recv_v = dist.P2POp(dist.irecv, remote_v, recv_rank, group=process_group)
        ops += [recv_k, recv_v]
    if need_to_send:
        # need to send
        send_k = dist.P2POp(dist.isend, local_k, send_rank, group=process_group)
        send_v = dist.P2POp(dist.isend, local_v, send_rank, group=process_group)
        ops += [send_k, send_v]

    if need_to_recv or need_to_send:
        reqs = dist.batch_isend_irecv(ops)
    else:
        reqs = None

    if not need_to_recv:
        return None, None, None, reqs
    return remote_k, remote_v, recv_rank, reqs


def ring_flash_attn_forward(
    process_group,
    local_q,
    local_k,
    local_v,
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
            block_out, block_lse, _ = raw_flash_attn_forward(
                local_q,
                k,
                v,
                dropout_p,
                causal=local_causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_softmax=True,
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
            block_dq, block_dk, block_dv = raw_flash_attn_backward(
                local_dout,
                local_q,
                k,
                v,
                local_out,
                softmax_lse,
                rng_state=None,
                dropout_p=dropout_p,
                causal=local_causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                softmax_scale=None,
            )
            block_dq = block_dq.to(torch.float32)
            block_dk = block_dk.to(torch.float32)
            block_dv = block_dv.to(torch.float32)

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
