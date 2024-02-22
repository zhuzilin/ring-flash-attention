## Ring Flash Attention

This repo implements the [RingAttention](https://github.com/lhao499/RingAttention) with [FlashAttention](https://github.com/Dao-AILab/flash-attention). Currently, this repo implements:

- `ring_flash_attn_qkvpacked_func`: corresponding to `flash_attn_qkvpacked_func`
-  `ring_flash_attn_varlen_qkvpacked_func`: corresponding to `flash_attn_varlen_qkvpacked_func`

The main idea is to use the `softmax_lse` output from the flash attention kernels.

There are some arithmetic errors with the current implementation. The reason for them is probably that flash attention will return bf16 value for each block, so we cannont accumluate the values with the original fp32 ones.

### TODOs

- [x] Implement `ring_flash_attn_varlen_qkvpacked_func`
- [ ] Implement zigzag block [issue#2](https://github.com/zhuzilin/ring-flash-attention/issues/2)
- [ ] Try to upstream to flash attention.

### Test

```bash
torchrun --nproc_per_node 8 test_qkvpacked_func.py
torchrun --nproc_per_node 8 test_varlen_qkvpacked_func.py
```
