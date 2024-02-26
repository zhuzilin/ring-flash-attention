## Ring Flash Attention

This repo implements the [RingAttention](https://github.com/lhao499/RingAttention) with [FlashAttention](https://github.com/Dao-AILab/flash-attention). Currently, this repo implements:

- `ring_flash_attn_qkvpacked_func`: ring attention version of `flash_attn_qkvpacked_func`
- `ring_flash_attn_qkvpacked_func_v2`: a different communication pattern of  `ring_flash_attn_qkvpacked_func`
- `ring_flash_attn_varlen_qkvpacked_func`: ring attention version of `flash_attn_varlen_qkvpacked_func`
- `zigzag_ring_flash_attn_qkvpacked_func`: an optimized version of `ring_flash_attn_varlen_qkvpacked_func`, see [issue#2](https://github.com/zhuzilin/ring-flash-attention/issues/2)
- `stripe_flash_attn_qkvpacked_func`: stripe attention version of `ring_flash_attn_varlen_qkvpacked_func`, the block size is set to 1 to use flash attn api.

The main idea is to use the `softmax_lse` output from the flash attention kernels.

The current performance on 8xH800 is ([benchmark/benchmark_qkvpacked_func.py](benchmark/benchmark_qkvpacked_func.py)):

|          | theoretic flash_attn | ring_attn | ring_attn_v2 | zigzag_ring | stripe_attn |
| -------- | -------------------- | --------- | ------------ | ----------- | ----------- |
| iter/sec | 2418.4/8=302.3       | 208.8     | 208.0        | 283.0       | 259.6       |
|          |                      | 68.8%     | 68.8%        | **93.6%**   | 85.9%       |

- Note that when running the benchmark with with 8 gpu, the flash attn code is running with 1/8 computation of ring attention.

There are some arithmetic errors with the current implementation. The reason for them is probably that flash attention will return bf16 value for each block, so we cannot accumluate the values with the original fp32 ones.

### TODOs

- [x] Implement `ring_flash_attn_varlen_qkvpacked_func`
- [x] Implement `zigzag_ring_flash_attn_qkvpacked_func` [issue#2](https://github.com/zhuzilin/ring-flash-attention/issues/2)

- [x] Implement `stripe_flash_attn_qkvpacked_func`

- [ ] Implement `zigzag_ring_flash_attn_varlen_qkvpacked_func`
- [ ] Try to upstream to flash attention.

### Test

```bash
torchrun --nproc_per_node 8 test/test_qkvpacked_func.py
torchrun --nproc_per_node 8 test/test_varlen_qkvpacked_func.py
torchrun --nproc_per_node 8 test/test_zigzag_qkvpacked_func.py
torchrun --nproc_per_node 8 test/test_stripe_qkvpacked_func.py
```

### Benchmark

```bash
torchrun --nproc_per_node 8 benchmark/benchmark_qkvpacked_func.py
torchrun --nproc_per_node 8 benchmark/benchmark_varlen_qkvpacked_func.py
```
