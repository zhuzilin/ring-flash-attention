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

        if comm.rank <= step:
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
            block_out = block_out.to(torch.float32)
            block_lse = block_lse.transpose(1, 2).unsqueeze(dim=-1)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
                q[1:, ],
                k[:-1, ],
                v[:-1, ],
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )
            block_out = block_out.to(torch.float32)
            block_lse = block_lse.transpose(1, 2).unsqueeze(dim=-1)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse,
                                          slice_=(slice(None), slice(None, -1)))

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    return out, lse


def stripe_flash_attn_backward_2(
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

        if kv_comm.rank <= step:
            block_dq, block_dk, block_dv = _flash_attn_backward(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                softmax_scale,
                dropout_p,
                causal,
                window_size,
                alibi_slopes,
                deterministic,
            )
            if dq is None:
                dq = block_dq
                dk = block_dk
                dv = block_dv
            else:
                dq += block_dq
                d_kv_comm.wait()
                dk = next_dk + block_dk
                dv = next_dv + block_dv
        else:
            block_dq, block_dk, block_dv = _flash_attn_backward(
                dout[1:, ],
                q[1:, ],
                k[:-1, ],
                v[:-1, ],
                out[1:, ],
                softmax_lse[1:, ],
                softmax_scale,
                dropout_p,
                causal,
                window_size,
                alibi_slopes,
                deterministic,
            )
            dq[1:, ] += block_dq
            d_kv_comm.wait()
            dk = next_dk
            dk[:-1, ] += block_dk
            dv = next_dv
            dv[:-1, ] += block_dv

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k = next_k
            v = next_v

        next_dk = d_kv_comm.send_recv(dk, **send_recv_kwargs)
        next_dv = d_kv_comm.send_recv(dv, **send_recv_kwargs)
        d_kv_comm.commit()

    d_kv_comm.wait()

    return dq, next_dk, next_dv
