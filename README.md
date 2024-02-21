## Ring Flash Attention

This repo implements the [RingAttention](https://github.com/lhao499/RingAttention) with [FlashAttention](https://github.com/Dao-AILab/flash-attention). Currently, the `ring_flash_attn_qkvpacked_func`, corresponding to `flash_attn_qkvpacked_func`, is implemented.

The main idea is to use the `softmax_lse` output from the flash attention kernels.

There are some arithmetic errors with the current implementation. The reason for them is probably that flash attention will return bf16 value for each block, so we cannont accumluate the values with the original fp32 ones.

### TODOs

- [ ] Make sure the nonpadded length works.
- [ ] Implement `ring_flash_attn_varlen_qkvpacked_func`

### Test

```bash
python test.py
```
