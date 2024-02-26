from flash_attn import flash_attn_varlen_qkvpacked_func
import torch
import torch.distributed as dist
from ring_flash_attn import ring_flash_attn_varlen_qkvpacked_func
from time import time


def benchmark_forward(f, num_benchmark_iter=1000, log=True):
    torch.cuda.empty_cache()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    seqlen = 8192
    nheads = 5
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False

    assert seqlen % (2 * world_size) == 0
    assert d % 8 == 0

    qkv = torch.randn(
        seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )

    cu_seqlens_list = [
        torch.tensor([0, 4096], device=device, dtype=torch.int32),
        torch.tensor([0, 128, 3824, 4096], device=device, dtype=torch.int32),
        torch.tensor([0, 2048, 4096], device=device, dtype=torch.int32),
        torch.tensor(
            [0, 1552, 3152, 3952, 4032, 4096], device=device, dtype=torch.int32
        ),
    ]
    max_seqlen_list = [
        (cu_seqlens[1:] - cu_seqlens[:1]).max().item() for cu_seqlens in cu_seqlens_list
    ]

    dist.barrier()
    torch.cuda.synchronize(device=device)
    start = time()

    for i in range(num_benchmark_iter):
        _ = f(
            qkv,
            cu_seqlens_list[i % len(cu_seqlens_list)],
            max_seqlen_list[i % len(max_seqlen_list)],
            dropout_p=dropout_p,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=False,
        )
    torch.cuda.synchronize(device=device)
    end = time()

    if rank == 0 and log:
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

    if rank == 0:
        print("warmuping...")
    benchmark_forward(flash_attn_varlen_qkvpacked_func, log=False)
    benchmark_forward(ring_flash_attn_varlen_qkvpacked_func, log=False)
    if rank == 0:
        print("benchmark:")
    benchmark_forward(flash_attn_varlen_qkvpacked_func)
    benchmark_forward(ring_flash_attn_varlen_qkvpacked_func)
