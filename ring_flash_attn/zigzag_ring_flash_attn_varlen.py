import torch
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)
from .utils import (
    RingComm,
    update_out_and_lse,
    update_out_and_lse_varlen_half,
    flatten_varlen_lse,
    unflatten_varlen_lse,
)


@torch.jit.script
def get_half(x, cu_seqlens, *, front: bool):
    xs = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        if front:
            end = (start + end) // 2
        else:
            start = (start + end) // 2
        xs.append(x[start:end])
    return torch.cat(xs, dim=0)


@torch.jit.script
def get_half_lse(lse, cu_seqlens, *, front: bool):
    new_lse = torch.empty(
        (lse.shape[0], lse.shape[1], lse.shape[2] // 2),
        dtype=lse.dtype,
        device=lse.device,
    )
    for i in range(len(cu_seqlens) - 1):
        seqlen = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
        if front:
            start, end = 0, seqlen // 2
        else:
            start, end = seqlen // 2, seqlen
        new_lse[i, :, : seqlen // 2] = lse[i, :, start:end]
    return new_lse


@torch.jit.script
def add_half(x, block_x, cu_seqlens, *, front: bool):
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        block_start, block_end = start // 2, end // 2
        if front:
            end = (start + end) // 2
        else:
            start = (start + end) // 2
        x[start:end] += block_x[block_start:block_end]


def zigzag_ring_flash_attn_varlen_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens,
    max_seqlen,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    comm = RingComm(process_group)

    block_seq_len = q.shape[0] // 2
    q1 = get_half(q, cu_seqlens, front=False)

    out = None
    lse = None
    next_k, next_v = None, None
    half_cu_seqlens = cu_seqlens // 2
    half_max_seqlen = max_seqlen // 2

    def forward(q, k, v, causal):
        seqlen_q = q.shape[0]
        seqlen_kv = k.shape[0]
        cu_seqlens_q = half_cu_seqlens if seqlen_q == block_seq_len else cu_seqlens
        max_seqlen_q = half_max_seqlen if seqlen_q == block_seq_len else max_seqlen
        cu_seqlens_kv = half_cu_seqlens if seqlen_kv == block_seq_len else cu_seqlens
        max_seqlen_kv = half_max_seqlen if seqlen_kv == block_seq_len else max_seqlen
        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_varlen_forward(
            q,
            k,
            v,
            # the first half and the second half are the same
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            return_softmax=True and dropout_p > 0,
        )
        return block_out, block_lse

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        if step == 0:
            block_out, block_lse = forward(q, k, v, causal=True)
            block_lse = flatten_varlen_lse(
                block_lse,
                cu_seqlens=cu_seqlens,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.rank:
            k0 = get_half(k, cu_seqlens, front=True)
            v0 = get_half(v, cu_seqlens, front=True)
            block_out, block_lse = forward(q, k0, v0, causal=False)
            block_lse = flatten_varlen_lse(
                block_lse,
                cu_seqlens=cu_seqlens,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = forward(q1, k, v, causal=False)
            block_lse = flatten_varlen_lse(
                block_lse,
                cu_seqlens=half_cu_seqlens,
            )
            out, lse = update_out_and_lse_varlen_half(
                out,
                lse,
                block_out,
                block_lse,
                cu_seqlens=cu_seqlens,
                front=False,
            )

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    lse = (
        unflatten_varlen_lse(lse, cu_seqlens, max_seqlen)
        .squeeze(dim=-1)
        .transpose(1, 2)
        .contiguous()
    )
    return out, lse


def zigzag_ring_flash_attn_varlen_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    cu_seqlens,
    max_seqlen,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    dout1 = get_half(dout, cu_seqlens, front=False)
    q1 = get_half(q, cu_seqlens, front=False)
    out1 = get_half(out, cu_seqlens, front=False)
    softmax_lse1 = get_half_lse(softmax_lse, cu_seqlens, front=False)
    block_seq_len = q.shape[0] // 2

    half_cu_seqlens = cu_seqlens // 2
    half_max_seqlen = max_seqlen // 2

    # repeatly allocating buffer may be slow...
    dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    def backward(dout, q, k, v, out, softmax_lse, causal):
        seqlen_q = q.shape[0]
        seqlen_kv = k.shape[0]
        cu_seqlens_q = half_cu_seqlens if seqlen_q == block_seq_len else cu_seqlens
        max_seqlen_q = half_max_seqlen if seqlen_q == block_seq_len else max_seqlen
        cu_seqlens_kv = half_cu_seqlens if seqlen_kv == block_seq_len else cu_seqlens
        max_seqlen_kv = half_max_seqlen if seqlen_kv == block_seq_len else max_seqlen
        _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq_buffer[:seqlen_q],
            dk_buffer[:seqlen_kv],
            dv_buffer[:seqlen_kv],
            # the first half and the second half are the same
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            rng_state=None,
        )

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()

        if step == 0:
            backward(dout, q, k, v, out, softmax_lse, causal=True)
            dq = dq_buffer.to(torch.float32)
            dk = dk_buffer.to(torch.float32)
            dv = dv_buffer.to(torch.float32)
        else:
            if step <= kv_comm.rank:
                k0 = get_half(k, cu_seqlens, front=True)
                v0 = get_half(v, cu_seqlens, front=True)
                backward(dout, q, k0, v0, out, softmax_lse, causal=False)
                dq += dq_buffer
            else:
                backward(dout1, q1, k, v, out1, softmax_lse1, causal=False)
                add_half(dq, dq_buffer, cu_seqlens, front=False)

            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            if step <= kv_comm.rank:
                add_half(dk, dk_buffer, cu_seqlens, front=True)
                add_half(dv, dv_buffer, cu_seqlens, front=True)
            else:
                dk += dk_buffer
                dv += dv_buffer

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k = next_k
            v = next_v

        next_dk = d_kv_comm.send_recv(dk, dk_comm_buffer)
        next_dv = d_kv_comm.send_recv(dv, dv_comm_buffer)
        d_kv_comm.commit()

    d_kv_comm.wait()

    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class ZigZagRingFlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
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
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = zigzag_ring_flash_attn_varlen_forward(
            group,
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens)
        ctx.max_seqlen = max_seqlen
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
        q, k, v, out, softmax_lse, cu_seqlens = ctx.saved_tensors
        dq, dk, dv = zigzag_ring_flash_attn_varlen_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens,
            ctx.max_seqlen,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def zigzag_ring_flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnVarlenFunc.apply(
        qkv[:, 0],
        qkv[:, 1],
        qkv[:, 2],
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def zigzag_ring_flash_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnVarlenFunc.apply(
        q,
        kv[:, 0],
        kv[:, 1],
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def zigzag_ring_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )
