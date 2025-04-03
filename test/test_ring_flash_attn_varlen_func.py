import sys
from flash_attn import flash_attn_varlen_qkvpacked_func
import torch
import torch.distributed as dist
from ring_flash_attn import ring_flash_attn_varlen_qkvpacked_func
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


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    nheads = 5
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False

    cu_seqlens = [0, 120, 1248, 4232]
    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    max_seqlen = (cu_seqlens_tensor[1:] - cu_seqlens_tensor[:-1]).max().item()
    total_length = cu_seqlens[-1]

    assert torch.all(cu_seqlens_tensor % world_size == 0)
    assert d % 8 == 0

    qkv = torch.randn(
        total_length, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    dist.broadcast(qkv, src=0)

    dout = torch.randn(total_length, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    local_cu_seqlens_tensor = cu_seqlens_tensor // world_size
    local_max_seqlen = max_seqlen // world_size

    local_qkv = extract_local(qkv, cu_seqlens, rank, world_size)
    local_qkv.requires_grad = True
    local_dout = extract_local(dout, cu_seqlens, rank, world_size)

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

    local_out = extract_local(out, cu_seqlens, rank, world_size)
    lse_list = extract_lse(lse, cu_seqlens)

    ring_out, ring_lse, _ = ring_flash_attn_varlen_qkvpacked_func(
        local_qkv,
        local_cu_seqlens_tensor,
        local_max_seqlen,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    ring_lse_list = extract_lse(ring_lse, local_cu_seqlens_tensor.tolist())

    log("out", out, rank0_only=True)
    log("out diff", local_out - ring_out)

    for lse, ring_lse in zip(lse_list, ring_lse_list):
        local_lse = lse.chunk(world_size, dim=-1)[rank]
        log("lse", lse, rank0_only=True)
        log("lse diff", local_lse - ring_lse)

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)

    out.backward(dout)
    dqkv = qkv.grad
    local_dqkv = extract_local(dqkv, cu_seqlens, rank, world_size)

    ring_out.backward(local_dout)
    ring_dqkv = local_qkv.grad

    log("local_dqkv", local_dqkv)
    log("dq diff", local_dqkv[:, 0] - ring_dqkv[:, 0])
    log("dk diff", local_dqkv[:, 1] - ring_dqkv[:, 1])
    log("dv diff", local_dqkv[:, 2] - ring_dqkv[:, 2])

    dist.destroy_process_group()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compile":
        torch._dynamo.config.capture_scalar_outputs = True
        flash_attn_varlen_qkvpacked_func = torch.compile(flash_attn_varlen_qkvpacked_func)
        ring_flash_attn_varlen_qkvpacked_func = torch.compile(ring_flash_attn_varlen_qkvpacked_func)
    main()
