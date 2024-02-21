import torch
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward

def raw_flash_attn_forward(
    q,
    k,
    v,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    softmax_scale=None,
    return_softmax=True,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        return_softmax=return_softmax and dropout_p > 0,
    )
    assert torch.all(out == out_padded)
    return out, softmax_lse, S_dmask


def raw_flash_attn_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    rng_state,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    softmax_scale=None,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    #qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
    dq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    _flash_attn_backward(
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
        rng_state=rng_state,
    )
    #dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
    return dq, dk, dv