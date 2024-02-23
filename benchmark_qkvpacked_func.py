from flash_attn import flash_attn_qkvpacked_func
import torch
import torch.distributed as dist
from ring_flash_attn import (
    ring_flash_attn_qkvpacked_func,
    zigzag_ring_flash_attn_qkvpacked_func,
)
from time import time


def benchmark_forward(f, num_warmup_iter=100, num_benchmark_iter=1000):
    torch.cuda.empty_cache()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    batch_size = 1
    seqlen = 4096
    nheads = 5
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False

    assert seqlen % (2 * world_size) == 0
    assert d % 8 == 0

    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )

    torch.cuda.synchronize(device=device)
    for _ in range(num_warmup_iter):
        _ = f(
            qkv,
            dropout_p=dropout_p,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=False,
        )

    dist.barrier()
    torch.cuda.synchronize(device=device)
    start = time()

    for _ in range(num_benchmark_iter):
        _ = f(
            qkv,
            dropout_p=dropout_p,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=False,
        )
    torch.cuda.synchronize(device=device)
    end = time()

    if rank == 0:
        print(
            f"{f.__name__} {num_benchmark_iter / (end - start)} iter/s, {end - start} sec"
        )


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    benchmark_forward(flash_attn_qkvpacked_func)
    benchmark_forward(ring_flash_attn_qkvpacked_func)
    benchmark_forward(zigzag_ring_flash_attn_qkvpacked_func)
