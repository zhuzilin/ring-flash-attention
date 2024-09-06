import torch
import torch.distributed as dist
from flash_attn import flash_attn_varlen_qkvpacked_func
from ring_flash_attn import (
    llama3_flash_attn_prepare_cu_seqlens,
    llama3_flash_attn_varlen_qkvpacked_func,
)
from utils import log, set_seed


def extract_local(value, cu_seqlens, rank, world_size):
    local_values = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        local_value = value[start:end].chunk(world_size, dim=0)[rank].detach().clone()
        local_values.append(local_value)
    return torch.cat(local_values, dim=0).contiguous()


def extract_lse(lse, cu_seqlens):
    values = []
    if lse.dim() == 2:
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            value = lse[:, start:end]
            values.append(value)
    else:
        assert lse.dim() == 3
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            value = lse[i, :, : end - start]
            values.append(value)
    return values


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    nheads = 5
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False

    cu_seqlens = [0, 120, 1248, 4232]
    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    max_seqlen = (cu_seqlens_tensor[1:] - cu_seqlens_tensor[:-1]).max().item()
    total_length = cu_seqlens[-1]
    local_length = total_length // world_size
    num_seq = len(cu_seqlens) - 1

    assert cu_seqlens_tensor[-1] % world_size == 0
    assert d % 8 == 0

    qkv = torch.randn(
        total_length, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    dist.broadcast(qkv, src=0)

    dout = torch.randn(total_length, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    local_qkv = qkv[rank * local_length : (rank + 1) * local_length].detach().clone()
    local_dout = dout[rank * local_length : (rank + 1) * local_length].detach().clone()
    local_qkv.requires_grad = True

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    out, lse, _ = flash_attn_varlen_qkvpacked_func(
        qkv,
        cu_seqlens_tensor,
        max_seqlen,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    local_out = out[rank * local_length : (rank + 1) * local_length]
    local_lse = lse[:, rank * local_length : (rank + 1) * local_length]

    (
        local_cu_seqlens_q,
        local_cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        local_k_slice,
    ) = llama3_flash_attn_prepare_cu_seqlens(
        cu_seqlens_tensor,
        causal=causal,
        rank=rank,
        world_size=world_size,
    )

    llama3_out, llama3_lse, _ = llama3_flash_attn_varlen_qkvpacked_func(
        local_qkv,
        local_cu_seqlens_q,
        local_cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        heads_k_stride=nheads,
        local_k_slice=local_k_slice,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    log("out", out, rank0_only=True)
    log("out diff", local_out - llama3_out)
    log("lse", lse, rank0_only=True)
    log("lse diff", local_lse - llama3_lse)

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)

    out.backward(dout)
    dqkv = qkv.grad
    local_dqkv = dqkv[rank * local_length : (rank + 1) * local_length]

    llama3_out.backward(local_dout)
    ring_dqkv = local_qkv.grad

    log("local_dqkv", local_dqkv)
    log("dqkv diff", local_dqkv - ring_dqkv)
