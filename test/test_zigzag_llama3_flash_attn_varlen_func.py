import sys
import torch
import torch.distributed as dist
from flash_attn import flash_attn_varlen_kvpacked_func
from ring_flash_attn import (
    llama3_flash_attn_prepare_cu_seqlens,
    zigzag_llama3_flash_attn_varlen_kvpacked_func,
)
from utils import log, set_seed


def extract_local(value, cu_seqlens, rank, world_size):
    """Extract local zigzag-distributed portion for this rank.

    Each sequence is split into 2*world_size chunks.
    GPU gets: chunk[rank] + chunk[2*world_size - 1 - rank]

    This creates the interleaved zigzag pattern where each GPU
    gets tokens from both beginning and end of sequences.
    """
    local_values = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        local_value = value[start:end].chunk(2 * world_size, dim=0)
        local_values.extend([
            local_value[rank].detach().clone(),
            local_value[2 * world_size - 1 - rank].detach().clone(),
        ])
    return torch.cat(local_values, dim=0).contiguous()


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    # Test configuration (matching llama3 test for fair comparison)
    nheads = 5
    nheads_k = 5  # No GQA to match llama3 test
    d = 8
    dropout_p = 0
    causal = True
    deterministic = False

    # Sequences must be divisible by 2*world_size for zigzag distribution
    # For single GPU (RTX 5070): divisible by 2
    # For 8 GPUs: divisible by 16
    if world_size == 1:
        cu_seqlens = [0, 128, 1024, 3072]  # Adjusted for 1 GPU (divisible by 2)
    else:
        cu_seqlens = [0, 128, 1264, 4256]  # Adjusted for 8 GPUs (divisible by 16)
    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    max_seqlen = (cu_seqlens_tensor[1:] - cu_seqlens_tensor[:-1]).max().item()
    total_length = cu_seqlens[-1]

    # Check divisibility requirements for zigzag distribution
    assert all((cu_seqlens_tensor[i+1] - cu_seqlens_tensor[i]) % (2 * world_size) == 0
               for i in range(len(cu_seqlens_tensor) - 1)), \
           "All sequences must be divisible by 2*world_size for zigzag distribution"
    assert d % 8 == 0

    # Create test data (full sequences)
    q = torch.randn(
        total_length, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    kv = torch.randn(
        total_length, 2, nheads_k, d, device=device, dtype=dtype, requires_grad=True
    )
    dist.broadcast(q, src=0)
    dist.broadcast(kv, src=0)

    dout = torch.randn(total_length, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    # Extract local zigzag-distributed slices (CRITICAL for zigzag_llama3!)
    local_q = extract_local(q, cu_seqlens_tensor, rank, world_size)
    local_kv = extract_local(kv, cu_seqlens_tensor, rank, world_size)
    local_dout = extract_local(dout, cu_seqlens_tensor, rank, world_size)
    local_q.requires_grad = True
    local_kv.requires_grad = True

    # Prepare LOCAL cu_seqlens (each rank has 2 chunks per sequence)
    local_cu_seqlens = []
    offset = 0
    for i in range(len(cu_seqlens_tensor) - 1):
        seq_len = (cu_seqlens_tensor[i+1] - cu_seqlens_tensor[i]).item()  # Convert to int!
        local_seq_len = seq_len // world_size  # Each rank gets 2 chunks
        local_cu_seqlens.append(offset)
        offset += local_seq_len
    local_cu_seqlens.append(offset)
    local_cu_seqlens_tensor = torch.tensor(local_cu_seqlens, dtype=torch.int32, device=device)

    # Prepare cu_seqlens for all-gather (uses GLOBAL cu_seqlens)
    (
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        local_k_slice,
    ) = llama3_flash_attn_prepare_cu_seqlens(
        cu_seqlens_tensor,
        causal=causal,
        rank=rank,
        world_size=world_size,
    )

    dist.barrier()
    if rank == 0:
        print("#" * 60)
        print("# Testing Zigzag Llama3 Flash Attention (varlen)")
        print("#" * 60)

    # Reference: Flash attention on full sequence
    ref_out = flash_attn_varlen_kvpacked_func(
        q,
        kv,
        cu_seqlens_tensor,
        cu_seqlens_tensor,
        max_seqlen,
        max_seqlen,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
    )

    # Extract local reference output using zigzag pattern
    local_ref_out = extract_local(ref_out, cu_seqlens_tensor, rank, world_size)

    # Reset gradients
    local_q.grad = None
    local_kv.grad = None

    # Forward pass
    dist.barrier()
    if rank == 0:
        print("\n# Forward pass:")

    zigzag_out = zigzag_llama3_flash_attn_varlen_kvpacked_func(
        local_q,
        local_kv,
        local_cu_seqlens_tensor,  # LOCAL cu_seqlens for Q
        cu_seqlens_tensor,         # GLOBAL cu_seqlens (original, not from llama3_prepare)
        max_seqlen,                # Use original max_seqlen
        max_seqlen,                # Use original max_seqlen
        heads_k_stride=nheads_k,  # Process all KV heads
        local_k_slice=local_k_slice,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        n_chunks=2,
    )

    # Check forward output
    out_diff = (local_ref_out - zigzag_out).abs()
    log("zigzag_out", zigzag_out, rank0_only=True)
    log("forward diff vs reference", out_diff)

    # Assert reasonable error bounds
    # With matching parameters, should get same tiny errors as llama3 test:
    # - llama3 test: max ~0.0001, mean ~1e-09 (nearly exact)
    # - zigzag_llama3 should match (both use same flash attention kernel)
    max_diff = out_diff.max().item()
    mean_diff = out_diff.mean().item()
    if rank == 0:
        if max_diff > 0.01 or mean_diff > 1e-06:
            print(f"WARNING: Large forward difference detected!")
            print(f"  Max diff: {max_diff:.6f}")
            print(f"  Mean diff: {mean_diff:.9f}")

    # Numerical assertions (should match llama3 test accuracy)
    assert max_diff < 0.01, f"Forward max diff {max_diff:.6f} exceeds tolerance 0.01"
    assert mean_diff < 1e-05, f"Forward mean diff {mean_diff:.9f} exceeds tolerance 1e-05"

    # Backward pass
    dist.barrier()
    if rank == 0:
        print("\n# Backward pass:")

    zigzag_out.backward(local_dout)

    # Compare with reference gradients
    q.grad = None
    kv.grad = None
    ref_out.backward(dout)

    # Extract reference gradients using zigzag pattern
    local_ref_dq = extract_local(q.grad, cu_seqlens_tensor, rank, world_size)
    local_ref_dkv = extract_local(kv.grad, cu_seqlens_tensor, rank, world_size)

    dq_diff = (local_ref_dq - local_q.grad).abs()
    dk_diff = (local_ref_dkv[:, 0] - local_kv.grad[:, 0]).abs()
    dv_diff = (local_ref_dkv[:, 1] - local_kv.grad[:, 1]).abs()

    log("dq diff vs reference", dq_diff)
    log("dk diff vs reference", dk_diff)
    log("dv diff vs reference", dv_diff)

    # Assert reasonable gradient error bounds
    # Based on test_llama3_flash_attn_varlen_func.py baseline errors:
    # - llama3 test: dQ max ~0.0001, dK max ~0.008, dV max ~0.016
    # - zigzag_llama3 should match llama3 test accuracy
    max_dq_diff = dq_diff.max().item()
    max_dk_diff = dk_diff.max().item()
    max_dv_diff = dv_diff.max().item()
    mean_dq_diff = dq_diff.mean().item()
    mean_dk_diff = dk_diff.mean().item()
    mean_dv_diff = dv_diff.mean().item()

    if rank == 0:
        if max_dq_diff > 0.01 or max_dk_diff > 0.1 or max_dv_diff > 0.1:
            print(f"WARNING: Large gradient difference detected!")
            print(f"  Max dQ diff: {max_dq_diff:.6f}")
            print(f"  Max dK diff: {max_dk_diff:.6f}")
            print(f"  Max dV diff: {max_dv_diff:.6f}")

    # Numerical assertions (should match llama3 test accuracy)
    # llama3 test gets: dQ max ~0.000122, dK/dV max ~0.0156
    # Allow small margin for zigzag pattern differences (increased for 8 GPUs)
    assert max_dq_diff < 0.01, f"dQ max diff {max_dq_diff:.6f} exceeds tolerance 0.01"
    assert max_dk_diff < 0.1, f"dK max diff {max_dk_diff:.6f} exceeds tolerance 0.1"
    assert max_dv_diff < 0.1, f"dV max diff {max_dv_diff:.6f} exceeds tolerance 0.1"
    assert mean_dq_diff < 1e-05, f"dQ mean diff {mean_dq_diff:.9f} exceeds tolerance 1e-05"
    assert mean_dk_diff < 3e-04, f"dK mean diff {mean_dk_diff:.9f} exceeds tolerance 3e-04"
    assert mean_dv_diff < 3e-04, f"dV mean diff {mean_dv_diff:.9f} exceeds tolerance 3e-04"

    dist.barrier()
    if rank == 0:
        print("\n" + "=" * 60)
        print("# Test PASSED!")
        print("=" * 60)
        print("\n# Zigzag Llama3 uses grouped attention calculation")
        print("# (splits Q into groups, computes attention for each group)")
        print("# Results match reference Flash Attention")

    dist.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compile":
        torch._dynamo.config.capture_scalar_outputs = True
        flash_attn_varlen_kvpacked_func = torch.compile(flash_attn_varlen_kvpacked_func)
        llama3_flash_attn_prepare_cu_seqlens = torch.compile(
            llama3_flash_attn_prepare_cu_seqlens
        )
        zigzag_llama3_flash_attn_varlen_kvpacked_func = torch.compile(
            zigzag_llama3_flash_attn_varlen_kvpacked_func
        )
    main()
