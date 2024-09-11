## Ring Flash Attention

This repo implements the [RingAttention](https://github.com/lhao499/RingAttention) with [FlashAttention](https://github.com/Dao-AILab/flash-attention). Currently, this repo implements:

- varlen api, corresponding to `flash_attn_varlen_func`:
  - `llama3_flash_attn_varlen_func`: the context parallelism used in [llama3 tech report](https://arxiv.org/abs/2407.21783) with extra design for varlen and low memory overhead.
    - Technically, this is not ring attention and will bring memory overhead, but this is the **recommended** api for most use case, as the communication pattern is more friendly to GPU cluster and the arithmetic errors is lower.
  - `ring_flash_attn_varlen_func`:  naive ring attention.
  - `zigzag_ring_flash_attn_varlen_func`: an more compute balanced version of ring attention, see  [issue#2](https://github.com/zhuzilin/ring-flash-attention/issues/2).
- batch api, corresponding to `flash_attn_func`:
  - `ring_flash_attn_func`: naive ring attention.
  - `zigzag_ring_flash_attn_func`: an more compute balanced version of ring attention, see  [issue#2](https://github.com/zhuzilin/ring-flash-attention/issues/2).
  - `stripe_flash_attn_func`: stripe attention version of `ring_flash_attn_func`, the block size is set to 1 to use flash_attn api, see: https://arxiv.org/abs/2311.09431
  

Note that

- all function has the `*_func`, `*_kvpacked_func`, `*_qkvpacked_func` variant implemented.
- the varlen versions only support passing one `cu_seqlens`.

The current performance on 8xH800 is ([benchmark/benchmark_qkvpacked_func.py](benchmark/benchmark_qkvpacked_func.py)):

|                      | GPU    | theoretic flash_attn | ring_attn | zigzag_ring | stripe_attn |
| -------------------- | ------ | -------------------- | --------- | ----------- | ----------- |
| fwd only (iter/sec)  | 8xH800 | 2418.4 / 8 = 302.3   | 208.0     | 283.0       | 259.6       |
|                      |        |                      | 68.8%     | **93.6%**   | 85.9%       |
| fwd + bwd (iter/sec) | 8xH800 | 705.2 / 8 = 88.2     | 54.3      | 75.7        | 76.9        |
|                      |        |                      | 61.5%     | 85.9%       | **87.2%**   |
| fwd only (iter/sec)  | 8xA100 | 1545.9 / 8 = 193.2   | 124.4     | 179.0       | 163.9       |
|                      |        |                      | 64.3%     | **92.7%**   | 84.8%       |
| fwd + bwd (iter/sec) | 8xA100 | 470.6 / 8 = 58.8     | 33.3      | 49.5        | 45.9        |
|                      |        |                      | 56.6%     | **84.1%**   | 78.1%       |

Note that
- when running the benchmark with with 8 gpu, the flash attn code is running with 1/8 computation of ring attention.
- nvlink between GPUs are required for high performance.
- the varlen versions of the ring attention variants are slow at the moment, please use the non-varlen version or the llama3 api if possible.
- please remember to adapt the RoPE offset for different api.

### Limits

There are some arithmetic errors with the current implementation. The reason for them is probably that flash attention will return bf16 value for each block, so we cannot accumluate the values with the original fp32 ones.

And also because we need to save extra fp32 buffer during computation, the memory usage would be higher than theoretic limit.

### TODOs

- [x] Implement `ring_flash_attn_varlen_qkvpacked_func`
- [x] Implement `zigzag_ring_flash_attn_qkvpacked_func` [issue#2](https://github.com/zhuzilin/ring-flash-attention/issues/2)
- [x] Implement `stripe_flash_attn_qkvpacked_func`
- [x] Implement `zigzag_ring_flash_attn_varlen_qkvpacked_func`
- [x] Implement `*_kvpacked_func` and `*_func` variant for all APIs
- [x] ~~Optimize `*_varlen_func`~~ Implement `llama3_flash_attn_varlen_func`.
- [ ] Add an example to train llama.
- [ ] Try to upstream to flash attention.

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
torchrun --nproc_per_node 8 benchmark/benchmark_qkvpacked_func.py
torchrun --nproc_per_node 8 benchmark/benchmark_varlen_qkvpacked_func.py
```

### Known Limits

- dropout is not supported at the moment, because it's hard to save all the rng_states.
- window_size is not supported, because it will be really tricky to implement a varlen version with window_size.