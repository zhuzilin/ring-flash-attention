## Ring Flash Attention

This repo implements [RingAttention](https://github.com/lhao499/RingAttention) using [FlashAttention](https://github.com/Dao-AILab/flash-attention). The current implementation supports:

- varlen (packing samples) api, corresponding to `flash_attn_varlen_func`:
  - `ring_flash_attn_varlen_func`:  A basic implementation of ring attention.
  - `zigzag_ring_flash_attn_varlen_func`: an more compute-balanced version of ring attention. More details in [issue#2](https://github.com/zhuzilin/ring-flash-attention/issues/2).
  - `llama3_flash_attn_varlen_func`: The context parallelism used in [llama3 tech report](https://arxiv.org/abs/2407.21783) with extra design for varlen and low memory overhead. Although technically not ring attention, this is **recommended** for most varlen use cases, as it offers a less intrusive alternative for training frameworks with fewer data manipulations and better arithmetic precision.
- batch api, corresponding to `flash_attn_func`:
  - `ring_flash_attn_func`: basic ring attention.
  - `zigzag_ring_flash_attn_func`: An more compute balanced version of ring attention, see [issue#2](https://github.com/zhuzilin/ring-flash-attention/issues/2).
  - `stripe_flash_attn_func`: Stripe attention version of `ring_flash_attn_func`, the block size is set to 1 to use flash_attn api, see: https://arxiv.org/abs/2311.09431
- [huggingface model adapter](ring_flash_attn/adapters/hf_adapter.py). Here is an example to use the adapter: [OpenRLHF/OpenRLHF/pull#439](https://github.com/OpenRLHF/OpenRLHF/pull/439/files).

Note that

- Each function includes `*_func`, `*_kvpacked_func`, `*_qkvpacked_func` variants.
- The varlen versions (except the llama3 version) only support passing one `cu_seqlens`.

## Performance Summary

The following table summarizes the performance of the implemented APIs:

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
| **varlen api**       | **GPU** | **theoretic<br />flash_attn** | **ring_attn** | **zigzag_ring** | **llama3_attn** |
| fwd only (iter/sec)  | 8xH800  | 852.4 / 8 = 106.6             | 52.4          | 74.8            | 60.8            |
|                      |         |                               | 49.1%         | **70.2%**       | 57.0%           |
| fwd + bwd (iter/sec) | 8xH800  | 225.4 / 8 = 28.2              | 14.4          | 21.4            | 16.4            |
|                      |         |                               | 51.1%         | **75.9%**       | 58.1%           |
| fwd only (iter/sec)  | 8xA100  | 532.3 / 8 = 66.5              | 33.1          | 47.9            | 34.3            |
|                      |         |                               | 49.8%         | **72.0%**       | 51.6%           |
| fwd + bwd (iter/sec) | 8xA100  | 133.8 / 8 = 16.7              | 8.7           | 13.4            | 9.7             |
|                      |         |                               | 52.1%         | **80.2%**       | 58.0%           |

Note that

- The code of the benchmark is in [benchmark](benchmark/), its configuration matches the [Meta-Llama-3.1-8B](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B/blob/main/config.json) setting, with a total sequence of length 8k per GPU.
- When running the benchmark with with 8 gpu, the flash attn code is running with 1/8 computation of ring attention, as flash attn code is running `8*1^2`, while the ring attn code is running `1*8^2`.
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
- [ ] Implement `zigzag_llama3_flash_attn_varlen_func`

### Test

```bash
torchrun --nproc_per_node 8 test/test_llama3_flash_attn_varlen_func.py
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
