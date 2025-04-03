#!/bin/bash

set -ex

num_gpus=8

tests=(
  test_llama3_flash_attn_varlen_func.py
  test_llama3_prepare_cu_seqlens.py
  test_ring_flash_attn_func.py
  test_ring_flash_attn_varlen_func.py
  test_stripe_flash_attn_func.py
  test_zigzag_ring_flash_attn_func.py
  test_zigzag_ring_flash_attn_varlen_func.py
)

for test in "${tests[@]}"; do
  torchrun --nproc_per_node $num_gpus test/$test
done

torchrun --nproc_per_node $num_gpus test/test_triton_kernels.py

for test in "${tests[@]}"; do
  torchrun --nproc_per_node $num_gpus test/$test compile
done
