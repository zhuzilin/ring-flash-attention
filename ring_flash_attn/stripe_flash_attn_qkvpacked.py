import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .utils import RingComm, update_out_and_lse


def stripe_flash_attn_forward(
        process_group,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale,
        dropout_p=0,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
):
    assert causal, "stripe flash attn only supports causal attention, if not causal, ring flash attn instead"
    comm = RingComm(process_group)

    out = None
    lse = None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        if step <= comm.rank:
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
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
                q[:, 1:, ],
                k[:, :-1, ],
                v[:, :-1, ],
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse,
                                          slice_=(slice(None), slice(None, -1)))

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    return out.to(q.dtype), lse


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
    assert causal, "stripe flash attn only supports causal attention, if not causal, ring flash attn instead"
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    send_recv_kwargs = {"send_direction": "decr"}
    next_dk, next_dv = None, None

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k, **send_recv_kwargs)
            next_v = kv_comm.send_recv(v, **send_recv_kwargs)
            kv_comm.commit()

        if step <= kv_comm.rank:
            block_dq, block_dk, block_dv, _ = _flash_attn_backward(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                alibi_slopes,
                deterministic,
                rng_state=None,
            )
            if dq is None:
                dq = block_dq
                dk = block_dk
                dv = block_dv
            else:
                # dq += block_dq
                d_kv_comm.wait()
                dk = next_dk + block_dk
                dv = next_dv + block_dv
        else:
            block_dq, block_dk, block_dv, _ = _flash_attn_backward(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                alibi_slopes,
                deterministic,
                rng_state=None,
            )
            d_kv_comm.wait()
            dk = next_dk
            dk[:, :-1, ] += block_dk[:, :-1, ]
            dv = next_dv
            dv[:, :-1, ] += block_dv[:, :-1, ]

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k = next_k
            v = next_v

        next_dk = d_kv_comm.send_recv(dk, **send_recv_kwargs)
        next_dv = d_kv_comm.send_recv(dv, **send_recv_kwargs)
        d_kv_comm.commit()

    d_kv_comm.wait()

    return dq, next_dk, next_dv


class StripeFlashAttnQKVPackedFunc(torch.autograd.Function):
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
        out, softmax_lse = stripe_flash_attn_forward(
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
        dqkv = torch.stack([dq, dk, dv], dim=2)
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None


def stripe_flash_attn_qkvpacked_func(
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
    return StripeFlashAttnQKVPackedFunc.apply(
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
