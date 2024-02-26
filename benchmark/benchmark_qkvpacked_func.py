from flash_attn import flash_attn_qkvpacked_func
import torch
import torch.distributed as dist
from ring_flash_attn import (
    ring_flash_attn_qkvpacked_func,
    ring_flash_attn_qkvpacked_func_v2,
    zigzag_ring_flash_attn_qkvpacked_func,
    stripe_flash_attn_qkvpacked_func,
)
import torch.cuda


def benchmark_forward(f, num_benchmark_iter=1000, log=True):
    torch.cuda.empty_cache()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    seqlen = 1024 * 8
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
    dist.barrier()
    begin = torch.cuda.Event(enable_timing=True)
    begin.record()
    with torch.no_grad():
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
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0
    dist.barrier()

    if rank == 0 and log:
        print(
            f"{f.__name__} {num_benchmark_iter / time} iter/s, {time} sec"
        )


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print("warmuping...")
    benchmark_forward(flash_attn_qkvpacked_func, log=False)
    benchmark_forward(ring_flash_attn_qkvpacked_func_v2, log=False)
    benchmark_forward(ring_flash_attn_qkvpacked_func, log=False)
    benchmark_forward(stripe_flash_attn_qkvpacked_func, log=False)
    benchmark_forward(zigzag_ring_flash_attn_qkvpacked_func, log=False)
    if rank == 0:
        print("benchmark:")
    benchmark_forward(flash_attn_qkvpacked_func)
    benchmark_forward(ring_flash_attn_qkvpacked_func)
    benchmark_forward(ring_flash_attn_qkvpacked_func_v2)
    benchmark_forward(stripe_flash_attn_qkvpacked_func)
    benchmark_forward(zigzag_ring_flash_attn_qkvpacked_func)