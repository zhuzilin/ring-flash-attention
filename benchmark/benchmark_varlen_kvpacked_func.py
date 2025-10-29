from flash_attn import flash_attn_varlen_kvpacked_func
import os
import sys
import torch
import torch.distributed as dist
from ring_flash_attn import (
    ring_flash_attn_varlen_kvpacked_func,
    zigzag_ring_flash_attn_varlen_kvpacked_func,
    llama3_flash_attn_varlen_kvpacked_func,
    zigzag_llama3_flash_attn_varlen_kvpacked_func,
    llama3_flash_attn_prepare_cu_seqlens,
)


def benchmark(
    f,
    use_double_cu_seqlens,
    use_llama3=False,
    use_zigzag_llama3=False,
    num_iter=100,
    forward_only=True,
    log=True,
    profile=False,
):
    dtype = torch.bfloat16
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    deterministic = False
    # config of llama3 8B
    seqlen = 1024 * 8
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    causal = True

    assert seqlen % (2 * world_size) == 0
    assert head_dim % 8 == 0

    q = torch.randn(
        seqlen, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True
    )
    kv = torch.randn(
        seqlen,
        2,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    dout = torch.randn(seqlen, num_heads, head_dim, device=device, dtype=dtype)

    cu_seqlens_list = [
        torch.tensor([0, 8192], device=device, dtype=torch.int32),
        torch.tensor([0, 256, 7648, 8192], device=device, dtype=torch.int32),
        torch.tensor([0, 4096, 8192], device=device, dtype=torch.int32),
        torch.tensor(
            [0, 3104, 6304, 7904, 8064, 8192], device=device, dtype=torch.int32
        ),
    ]

    if use_llama3 or use_zigzag_llama3:
        cu_seqlens_q_list = []
        cu_seqlens_k_list = []
        cu_seqlens_k_global_list = []  # For zigzag_llama3: global cu_seqlens
        max_seqlen_q_list = []
        max_seqlen_k_list = []
        local_k_slice_list = []
        for cu_seqlens in cu_seqlens_list:
            cu_seqlens_global = cu_seqlens * world_size
            (
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                local_k_slice,
            ) = llama3_flash_attn_prepare_cu_seqlens(
                cu_seqlens_global,
                causal=causal,
                rank=rank,
                world_size=world_size,
            )
            cu_seqlens_q_list.append(cu_seqlens_q)
            cu_seqlens_k_list.append(cu_seqlens_k)
            cu_seqlens_k_global_list.append(cu_seqlens_global)  # Save global version
            max_seqlen_q_list.append(max_seqlen_q)
            max_seqlen_k_list.append(max_seqlen_k)
            local_k_slice_list.append(local_k_slice)
    else:
        max_seqlen_list = [
            (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            for cu_seqlens in cu_seqlens_list
        ]

    if profile:
        torch.backends.cudnn.benchmark = True
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=5,
                warmup=5,
                active=5,
            ),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(
                    f"./benchmark/logs/{f.__name__}", f"rank_{dist.get_rank()}"
                )
            ),
        )

    if profile:
        profiler.start()

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    def wrapper(i: int):
        if use_zigzag_llama3:
            idx = i % len(cu_seqlens_list)
            return f(
                q,
                kv,
                cu_seqlens_list[idx],  # Use LOCAL cu_seqlens (not sliced by llama3_prepare)
                cu_seqlens_k_global_list[idx],  # Pass GLOBAL cu_seqlens (scaled by world_size)
                max_seqlen_q_list[idx],
                max_seqlen_k_list[idx],
                heads_k_stride=num_kv_heads,  # Backward requires heads_k_stride == nheads_k for now
                local_k_slice=local_k_slice_list[idx],
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=False,
                n_chunks=2,
            )
        elif use_llama3:
            return f(
                q,
                kv,
                cu_seqlens_q_list[i % len(cu_seqlens_list)],
                cu_seqlens_k_list[i % len(cu_seqlens_list)],
                max_seqlen_q_list[i % len(cu_seqlens_list)],
                max_seqlen_k_list[i % len(cu_seqlens_list)],
                heads_k_stride=1,
                local_k_slice=local_k_slice_list[i % len(cu_seqlens_list)],
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=False,
            )
        elif use_double_cu_seqlens:
            return f(
                q,
                kv,
                cu_seqlens_list[i % len(cu_seqlens_list)],
                cu_seqlens_list[i % len(cu_seqlens_list)],
                max_seqlen_list[i % len(cu_seqlens_list)],
                max_seqlen_list[i % len(cu_seqlens_list)],
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=False,
            )
        else:
            return f(
                q,
                kv,
                cu_seqlens_list[i % len(cu_seqlens_list)],
                max_seqlen_list[i % len(cu_seqlens_list)],
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=False,
            )

    if forward_only:
        with torch.no_grad():
            for i in range(num_iter):
                _ = wrapper(i)
    else:
        for i in range(num_iter):
            q.grad = None
            kv.grad = None
            out = wrapper(i)
            out.backward(dout)
            if profile:
                profiler.step()
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0

    if profile:
        profiler.stop()
    if rank == 0 and log:
        print(f"{num_iter / time} iter/s, {time} sec")


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    forward_only = False
    profile = False
    compile_func = False

    # Parse command line arguments
    for arg in sys.argv[1:]:
        if arg == "compile":
            compile_func = True
            torch._dynamo.config.capture_scalar_outputs = True
        elif arg == "forward_only":
            forward_only = True
        elif arg == "profile":
            profile = True

    num_iter = 500 if forward_only else 100

    for f, use_double_cu_seqlens in [
        (flash_attn_varlen_kvpacked_func, True),
        (ring_flash_attn_varlen_kvpacked_func, False),
        (zigzag_ring_flash_attn_varlen_kvpacked_func, False),
    ]:
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"# {f.__name__}")
        f = torch.compile(f) if compile_func else f
        benchmark(
            f,
            use_double_cu_seqlens,
            forward_only=forward_only,
            num_iter=num_iter,
            log=False,
        )
        benchmark(
            f,
            use_double_cu_seqlens,
            forward_only=forward_only,
            num_iter=num_iter,
            log=True,
            profile=profile,
        )

    for f, use_double_cu_seqlens in [
        (llama3_flash_attn_varlen_kvpacked_func, True),
    ]:
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"# {f.__name__}")
        f = torch.compile(f) if compile_func else f
        benchmark(
            f,
            use_double_cu_seqlens,
            use_llama3=True,
            forward_only=forward_only,
            num_iter=num_iter,
            log=False,
        )
        benchmark(
            f,
            use_double_cu_seqlens,
            use_llama3=True,
            forward_only=forward_only,
            num_iter=num_iter,
            log=True,
            profile=profile,
        )

    # Benchmark zigzag_llama3_flash_attn
    torch.cuda.empty_cache()
    if rank == 0:
        print(f"# zigzag_llama3_flash_attn_varlen_kvpacked_func")

    benchmark(
        zigzag_llama3_flash_attn_varlen_kvpacked_func,
        use_double_cu_seqlens=True,
        use_zigzag_llama3=True,
        forward_only=forward_only,
        num_iter=num_iter,
        log=False,
    )
    benchmark(
        zigzag_llama3_flash_attn_varlen_kvpacked_func,
        use_double_cu_seqlens=True,
        use_zigzag_llama3=True,
        forward_only=forward_only,
        num_iter=num_iter,
        log=True,
        profile=profile,
    )

    dist.destroy_process_group()
