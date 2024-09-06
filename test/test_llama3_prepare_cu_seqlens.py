from ring_flash_attn import llama3_flash_attn_prepare_cu_seqlens
import torch

if __name__ == "__main__":
    device = torch.device("cuda")
    cu_seqlens = [0, 7, 14, 16]
    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    world_size = 8
    for rank in range(world_size):
        (
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            local_k_slice,
        ) = llama3_flash_attn_prepare_cu_seqlens(
            cu_seqlens_tensor,
            causal=True,
            rank=rank,
            world_size=world_size,
        )

        assert max_seqlen_q == (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max()
        assert max_seqlen_k == (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max()
        print(f"RANK: {rank}")
        print(f"  cu_seqlens_q: {cu_seqlens_q}")
        print(f"  cu_seqlens_k: {cu_seqlens_k}")
        print(f"  local_k_slice: {local_k_slice}")
