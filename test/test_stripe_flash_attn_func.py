import sys
import torch
import torch.distributed as dist
from flash_attn import flash_attn_qkvpacked_func
from ring_flash_attn import stripe_flash_attn_qkvpacked_func
from utils import log, set_seed


def extract_local(value, rank, world_size, dim=1):
    value = torch.stack(value.split(world_size, dim=dim), dim=dim).transpose(
        dim, dim + 1
    )
    slicer = [rank if i == dim else slice(None) for i in range(len(value.shape))]
    return value[slicer].contiguous()


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    seqlen = 3824
    nheads = 5
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False

    assert causal
    assert seqlen % (2 * world_size) == 0
    assert d % 8 == 0

    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    dist.broadcast(qkv, src=0)

    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    local_qkv = extract_local(qkv, rank, world_size).detach().clone()
    local_qkv.requires_grad = True
    local_dout = extract_local(dout, rank, world_size).detach().clone()

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    out, lse, _ = flash_attn_qkvpacked_func(
        qkv,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    local_out = extract_local(out, rank, world_size)
    local_lse = extract_local(lse, rank, world_size, dim=2)

    ring_out, ring_lse, _ = stripe_flash_attn_qkvpacked_func(
        local_qkv,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    log("out", out, rank0_only=True)
    log("lse", lse, rank0_only=True)
    log("out diff", local_out - ring_out)
    log("lse diff", local_lse - ring_lse)

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)

    out.backward(dout)
    dqkv = qkv.grad

    local_dqkv = extract_local(dqkv, rank, world_size)

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
        flash_attn_qkvpacked_func = torch.compile(flash_attn_qkvpacked_func)
        stripe_flash_attn_qkvpacked_func = torch.compile(stripe_flash_attn_qkvpacked_func)
    main()
