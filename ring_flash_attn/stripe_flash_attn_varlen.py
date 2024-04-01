import torch
from flash_attn.flash_attn_interface import _flash_attn_backward
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)
from .utils import RingComm, update_out_and_lse, update_out_and_lse_masked

try:
    from .triton_utils import (
        flatten_varlen_lse,
        unflatten_varlen_lse,
    )
except:
    from .utils import (
        flatten_varlen_lse,
        unflatten_varlen_lse,
    )


def stripe_flash_attn_varlen_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens,
    max_seqlen,
    softmax_scale,
    dropout_p=0,
):
    comm = RingComm(process_group)

    out = None
    lse = None

    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        if step <= comm.rank:
            block_out, _, _, _, _, block_lse, _, _ = _flash_attn_varlen_forward(
                q,
                k,
                v,

                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,

                dropout_p,
                softmax_scale,
                causal=True,
                window_size=(-1, -1),
                alibi_slopes=None,
                return_softmax=True and dropout_p > 0,
            )
            block_lse = flatten_varlen_lse(
                block_lse,
                cu_seqlens=cu_seqlens,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:

            seq_len = q.shape[0]
            cu_seqlens_q = cu_seqlens.clone() + 1
            cu_seqlens_q[-1] = seq_len
            cu_seqlens_k = cu_seqlens.clone()
            cu_seqlens_k[-1] = seq_len-1

            max_seqlen_q = max_seqlen
            max_seqlen_k = max_seqlen_q 
            
            block_out, _, _, _, _, block_lse, _, _ = _flash_attn_varlen_forward(
                q,
                k, 
                v,

                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,

                dropout_p,
                softmax_scale,
                causal=True,
                window_size=(-1, -1),
                alibi_slopes=None,
                return_softmax=True and dropout_p > 0,
            )
            print("cu_seqlens_q", cu_seqlens_q)
            print("cu_seqlens_k", cu_seqlens_k)
            #print("block_lse", block_lse[0, 0], block_lse.shape)
            block_lse[:,:,1:] = block_lse[:,:,:-1]
            print("block_lse", block_lse.shape)
            block_lse = flatten_varlen_lse(
                block_lse,
                cu_seqlens=cu_seqlens,
            )
            
            # print("block_lse", block_lse.shape)
            # print("cu_seqlens", cu_seqlens)
            # print("0", block_lse[:, cu_seqlens[:-1]])
            # print("+1", block_lse[:, cu_seqlens[:-1]+1])
            #print("block_lse2", block_lse[0], block_lse.shape)

            # first outputs of sequences need to be ignored            
            mask = torch.zeros(seq_len, dtype=torch.bool, device=q.device)
            mask[cu_seqlens[:-1]] = True

            out, lse = update_out_and_lse_masked(out, lse, block_out, block_lse, mask)

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    lse = unflatten_varlen_lse(lse, cu_seqlens, max_seqlen)
    return out, lse


def stripe_flash_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert (
        causal
    ), "stripe flash attn only supports causal attention, if not causal, ring flash attn instead"
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()

        shift_causal = step > kv_comm.rank
        softmax_lse_1 = None
        if not shift_causal:
            _flash_attn_backward(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                block_dq_buffer,
                block_dk_buffer,
                block_dv_buffer,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                alibi_slopes,
                deterministic,
                rng_state=None,
            )
        else:
            if softmax_lse_1 is None:
                # lazy init, since the last rank does not need softmax_lse_1
                softmax_lse_1 = softmax_lse[:, :, 1:].contiguous()
            _flash_attn_backward(
                dout[:, 1:],
                q[:, 1:],
                k[:, :-1],
                v[:, :-1],
                out[:, 1:],
                softmax_lse_1,
                block_dq_buffer[:, 1:],
                block_dk_buffer[:, :-1],
                block_dv_buffer[:, :-1],
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                alibi_slopes,
                deterministic,
                rng_state=None,
            )

        if dq is None:
            dq = block_dq_buffer.to(torch.float32)
            dk = block_dk_buffer.to(torch.float32)
            dv = block_dv_buffer.to(torch.float32)
        else:
            if not shift_causal:
                dq += block_dq_buffer
            else:
                dq[:, 1:] += block_dq_buffer[:, 1:]
            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk = next_dk
            dv = next_dv

            if not shift_causal:
                dk = block_dk_buffer + dk
                dv = block_dv_buffer + dv
            else:
                dk[:, :-1] += block_dk_buffer[:, :-1]
                dv[:, :-1] += block_dv_buffer[:, :-1]

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k = next_k
            v = next_v

        next_dk = d_kv_comm.send_recv(dk, dk_comm_buffer)
        next_dv = d_kv_comm.send_recv(dv, dv_comm_buffer)
        d_kv_comm.commit()

    d_kv_comm.wait()

    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class StripeFlashAttnVarlenFunc(torch.autograd.Function):
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
        assert window_size is None or window_size == (-1, -1)
        assert not deterministic, "deterministic not supported"
        assert causal, "stripe flash attn varlen only supports causal attention"

        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = stripe_flash_attn_varlen_forward(
            group,
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
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
        dq, dk, dv = stripe_flash_attn_backward(
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
        return dq, dk, dv, None, None, None, None, None, None, None, None


def stripe_flash_attn_varlen_qkvpacked_func(
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
    return StripeFlashAttnVarlenFunc.apply(
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


def stripe_flash_attn_varlen_kvpacked_func(
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
    return StripeFlashAttnVarlenFunc.apply(
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


def stripe_flash_attn_varlen_func(
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
    return StripeFlashAttnVarlenFunc.apply(
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
