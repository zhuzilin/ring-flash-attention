import torch
import torch.distributed as dist
from raw_flash_attn import raw_flash_attn_forward, raw_flash_attn_backward


def ring_flash_attn_forward(
    local_qkv,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_q = local_qkv[:, :, 0]
    local_k = local_qkv[:, :, 1].contiguous()
    local_v = local_qkv[:, :, 2].contiguous()

    ks = [torch.zeros_like(local_k) for _ in range(world_size)]
    vs = [torch.zeros_like(local_v) for _ in range(world_size)]

    dist.all_gather(ks, local_k)
    dist.all_gather(vs, local_v)

    out = None
    lse = None
    for i in range(rank + 1):
        local_causal = causal and i == rank
        k = ks[i]
        v = vs[i]
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
            out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
            lse = new_lse

    out = out.to(torch.bfloat16)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def ring_flash_attn_backward(
    local_dout,
    local_qkv,
    local_out,
    softmax_lse,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_q = local_qkv[:, :, 0]
    local_k = local_qkv[:, :, 1].contiguous()
    local_v = local_qkv[:, :, 2].contiguous()

    ks = [torch.zeros_like(local_k) for _ in range(world_size)]
    vs = [torch.zeros_like(local_v) for _ in range(world_size)]

    dist.all_gather(ks, local_k)
    dist.all_gather(vs, local_v)

    local_dq = None
    dks = []
    dvs = []
    for i in range(rank + 1):
        local_causal = causal and i == rank
        k = ks[i]
        v = vs[i]
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
        else:
            local_dq += block_dq
        dks.append(block_dk)
        dvs.append(block_dv)

    for i in range(rank + 1, world_size):
        dks.append(torch.zeros_like(block_dk))
        dvs.append(torch.zeros_like(block_dv))

    dks = torch.cat(dks, dim=1)
    dvs = torch.cat(dvs, dim=1)
    dist.all_reduce(dks)
    dist.all_reduce(dvs)

    local_dk = dks.chunk(world_size, dim=1)[rank]
    local_dv = dvs.chunk(world_size, dim=1)[rank]

    return local_dq, local_dk, local_dv
