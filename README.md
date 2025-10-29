## Ring Flash Attention

This repo implements [RingAttention](https://github.com/lhao499/RingAttention) using [FlashAttention](https://github.com/Dao-AILab/flash-attention). The current implementation supports:

- varlen (packing samples) api, corresponding to `flash_attn_varlen_func`:
  - `ring_flash_attn_varlen_func`:  A basic implementation of ring attention.
  - `zigzag_ring_flash_attn_varlen_func`: an more compute-balanced version of ring attention. More details in [issue#2](https://github.com/zhuzilin/ring-flash-attention/issues/2).
  - `llama3_flash_attn_varlen_func`: The context parallelism used in [llama3 tech report](https://arxiv.org/abs/2407.21783) with extra design for varlen and low memory overhead. Although technically not ring attention, this is **recommended** for most varlen use cases, as it offers a less intrusive alternative for training frameworks with fewer data manipulations and better arithmetic precision.
  - `zigzag_llama3_flash_attn_varlen_func`: Combines zigzag interleaved distribution with llama3-style all-gather communication. **Achieves 88.9% efficiency (fwd+bwd) and 82.3% efficiency (fwd only), the fastest distributed attention method!**
- batch api, corresponding to `flash_attn_func`:
  - `ring_flash_attn_func`: basic ring attention.
  - `zigzag_ring_flash_attn_func`: An more compute balanced version of ring attention, see [issue#2](https://github.com/zhuzilin/ring-flash-attention/issues/2).
  - `stripe_flash_attn_func`: Stripe attention version of `ring_flash_attn_func`, the block size is set to 1 to use flash_attn api, see: https://arxiv.org/abs/2311.09431
- [huggingface model adapter](ring_flash_attn/adapters/hf_adapter.py). Here is an example to use the adapter:

```python
# torchrun --nproc_per_node=2 this_script.py
import torch
from ring_flash_attn import substitute_hf_flash_attn, update_ring_flash_attn_params
from torch import distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Initialize distributed training
    dist.init_process_group(backend="nccl")

    # Get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    # Create group and substitute flash attention
    group = dist.new_group(ranks=range(world_size), backend="nccl")
    substitute_hf_flash_attn(group, heads_k_stride=1)

    # Get the ring attention rank
    ring_attn_rank = dist.get_rank(group=group)  # only one group for ring attention here: this should be the same as rank

    # Tokenize input and prepare position IDs
    input_ids = tokenizer(["Lorem ipsum dolor sit", "amet, consectetur adipiscing", "elit, sed do"]).input_ids
    lengths = [len(seq) for seq in input_ids]
    input_ids = torch.cat([torch.tensor(seq, device=device) for seq in input_ids]).unsqueeze(0)
    position_ids = torch.cat([torch.arange(length, device=device) for length in lengths]).unsqueeze(0)
    
    # Compute cu_seqlens and update parameters
    cu_seqlens = torch.cat([torch.tensor([0], device=device), torch.cumsum(torch.tensor(lengths, device=device), dim=0)]).to(torch.int32)
    update_ring_flash_attn_params(cu_seqlens, group)

    # Chunk input_ids and position_ids
    input_ids = torch.chunk(input_ids, world_size, dim=1)[ring_attn_rank]
    position_ids = torch.chunk(position_ids, world_size, dim=1)[ring_attn_rank]
    
    output = model(input_ids=input_ids, position_ids=position_ids)

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

Note that

- Each function includes `*_func`, `*_kvpacked_func`, `*_qkvpacked_func` variants.
- The varlen versions (except the llama3 version) only support passing one `cu_seqlens`.

## Performance Summary

### Batch API

| batch api            | GPU     | theoretic<br />flash_attn     | ring_attn     | zigzag_ring     | stripe_attn     |
| -------------------- | ------- | ----------------------------- | ------------- | --------------- | --------------- |
| fwd only (iter/sec)  | 8xH800  | 591.5 / 8 = 73.9              | 38.5          | 63.0            | 55.0            |
|                      |         |                               | 52.1%         | **85.2%**       | 74.4%           |
| fwd + bwd (iter/sec) | 8xH800  | 154.7 / 8 = 19.3              | 10.4          | 17.4            | 16.0            |
|                      |         |                               | 53.9%         | **90.2%**       | 82.9%           |
| fwd only (iter/sec)  | 8xA100  | 373.4 / 8 = 46.7              | 24.0          | 38.2            | 32.5            |
|                      |         |                               | 51.4%         | **81.7%**       | 69.6%           |
| fwd + bwd (iter/sec) | 8xA100  | 94.7 / 8 = 11.8               | 6.2           | 10.6            | 9.75            |
|                      |         |                               | 52.5%         | **89.8%**       | 82.6%           |

### Varlen API

| varlen api           | GPU     | theoretic<br />flash_attn     | ring_attn     | zigzag_ring     | llama3_attn     | zigzag_llama3   |
| -------------------- | ------- | ----------------------------- | ------------- | --------------- | --------------- | --------------- |
| fwd only (iter/sec)  | 8xH800  | 852.4 / 8 = 106.6             | 52.4          | 74.8            | 60.8            | N/A             |
|                      |         |                               | 49.1%         | **70.2%**       | 57.0%           |                 |
| fwd + bwd (iter/sec) | 8xH800  | 225.4 / 8 = 28.2              | 14.4          | 21.4            | 16.4            | N/A             |
|                      |         |                               | 51.1%         | **75.9%**       | 58.1%           |                 |
| fwd only (iter/sec)  | 8xA100   | 509.3 / 8 = 63.7              | 31.7          | 47.2            | 29.9            | **52.4**        |
|                      |         |                               | 49.7%         | 74.2%           | 47.0%           | **82.3%**       |
| fwd + bwd (iter/sec) | 8xA100   | 135.4 / 8 = 16.9              | 8.4           | 13.8            | 9.3             | **15.1**        |
|                      |         |                               | 49.4%         | 81.4%           | 55.2%           | **88.9%**       |

Note that

- The code of the benchmark is in [benchmark](benchmark/), its configuration matches the [Meta-Llama-3.1-8B](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B/blob/main/config.json) setting, with a total sequence of length 8k per GPU.
- When running the benchmark with with 8 gpu, the flash attn code is running with 1/8 computation of ring attention, as flash attn code is running `8*1^2`, while the ring attn code is running `1*8^2`.
- **Latest benchmark (8xA100, last 2 rows)**: Shows **zigzag_llama3 achieving 82.3% efficiency for forward-only** (52.4 iter/s) and **88.9% efficiency for forward+backward** (15.1 iter/s), making it the **fastest distributed attention method** in both modes. For forward+backward, this represents a **9.2% improvement over zigzag_ring** (15.1 vs 13.8 iter/s) and **79.8% improvement over standard ring** (15.1 vs 8.4 iter/s). For forward-only, **11.0% faster than zigzag_ring** and **65.4% faster than standard ring**.
- NVLink between GPUs are required for high performance.
- Please remember to adapt the RoPE offset for different api.

### Installation

```bash
pip install ring-flash-attn
```

or use the following command to build from source:

```bash
git clone https://github.com/zhuzilin/ring-flash-attention.git
cd ring-flash-attention
pip install .
```

### TODOs

- [x] Implement `ring_flash_attn_varlen_qkvpacked_func`
- [x] Implement `zigzag_ring_flash_attn_qkvpacked_func` [issue#2](https://github.com/zhuzilin/ring-flash-attention/issues/2)
- [x] Implement `stripe_flash_attn_qkvpacked_func`
- [x] Implement `zigzag_ring_flash_attn_varlen_qkvpacked_func`
- [x] Implement `*_kvpacked_func` and `*_func` variant for all APIs
- [x] ~~Optimize `*_varlen_func`~~ Implement `llama3_flash_attn_varlen_func`
- [x] ~~Add an example to train llama~~ Implement adapter for huggingface model
- [x] Implement `zigzag_llama3_flash_attn_varlen_func` - **Achieved 88.9% efficiency (fwd+bwd) and 82.3% efficiency (fwd only), the fastest distributed attention method!**

### Test

```bash
torchrun --nproc_per_node 8 test/test_llama3_flash_attn_varlen_func.py
torchrun --nproc_per_node 8 test/test_zigzag_llama3_flash_attn_varlen_func.py
torchrun --nproc_per_node 8 test/test_ring_flash_attn_func.py
torchrun --nproc_per_node 8 test/test_ring_flash_attn_varlen_func.py
torchrun --nproc_per_node 8 test/test_zigzag_ring_flash_attn_func.py
torchrun --nproc_per_node 8 test/test_zigzag_ring_flash_attn_varlen_func.py
torchrun --nproc_per_node 8 test/test_stripe_flash_attn_func.py
```

### Benchmark

```bash
torchrun --nproc_per_node 8 benchmark/benchmark_kvpacked_func.py
torchrun --nproc_per_node 8 benchmark/benchmark_varlen_kvpacked_func.py
```

### Known Limitations

There are some arithmetic errors with the current implementation. The reason for them is probably that flash attention will return bf16 value for each block, so we cannot accumluate the values with the original fp32 ones.

And also because we need to save extra fp32 buffer during computation, the memory usage would be higher than theoretic limit.

Also,

- dropout is not supported at the moment, because it's hard to save all the rng_states.
- window_size is not supported, because it will be really tricky to implement a varlen version with window_size.
