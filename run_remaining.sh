#!/bin/bash
# Remaining configs from density sweep 2 + full depth sweep.
set -e

COMMON="ITERATIONS=200 VAL_LOSS_EVERY=0 TRAIN_BATCH_TOKENS=8192 VAL_BATCH_SIZE=8192 WANDB_ENABLED=1"
QK_BN="WEIGHT_TYPES=q=b,k=b"

run_config() {
    local run_id=$1
    shift
    echo "--- Running: $run_id ---"
    env $COMMON RUN_ID="$run_id" "$@" \
        uv run torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 \
        | grep -E "model_params|dense:|step:200|Total submission size int8|final_int8_zlib_roundtrip_exact|Error|OOM" \
        | sed "s/^/  [$run_id] /"
    echo ""
}

echo "=== Remaining Density Sweep 2 (8 configs) ==="

run_config "qk_9L_d2048_1d_r128" LAYER_PATTERN="d,s,s,s,s,s,s,s,s" SEED_NUM_BASES=16 NUM_LAYERS=9 MODEL_DIM=2048 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "qk_6L_d2048_1d_r128" LAYER_PATTERN="d,s,s,s,s,s" SEED_NUM_BASES=16 NUM_LAYERS=6 MODEL_DIM=2048 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "qk_12L_d1536_1d_r32" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=16 NUM_LAYERS=12 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=32
run_config "qk_12L_d1536_1d_r64" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=16 NUM_LAYERS=12 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=64
run_config "qk_12L_d1536_1d_r256" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=16 NUM_LAYERS=12 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=256
run_config "qkv_12L_d1024_r128" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 WEIGHT_TYPES="q=b,k=b,v=b" BOTTLENECK_RANK=128
run_config "qkv_12L_d1024_r256" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 WEIGHT_TYPES="q=b,k=b,v=b" BOTTLENECK_RANK=256
run_config "qkvo_12L_d1024_r256" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 WEIGHT_TYPES="q=b,k=b,v=b,o=b" BOTTLENECK_RANK=256

echo "=== Depth at Width Sweep (20 configs) ==="

# d1024 depth sweep
for L in 6 9 12 15 20 25 30; do
    k=$( [ $L -le 12 ] && echo 32 || echo 16 )
    pat="d$(printf ',s%.0s' $(seq 1 $((L-1))))"
    [ $L -le 25 ] && k_arg=$( [ $L -le 12 ] && echo 32 || ( [ $L -le 20 ] && echo 16 || echo 8 ) )
    run_config "depth_d1024_${L}L" LAYER_PATTERN="$pat" SEED_NUM_BASES=$k_arg NUM_LAYERS=$L MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
done

# d1536 depth sweep
for L in 6 9 12 15 20 25; do
    k=$( [ $L -le 9 ] && echo 16 || echo 8 )
    pat="d$(printf ',s%.0s' $(seq 1 $((L-1))))"
    run_config "depth_d1536_${L}L" LAYER_PATTERN="$pat" SEED_NUM_BASES=$k NUM_LAYERS=$L MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
done

# d2048 depth sweep
for L in 6 9 12 15; do
    pat="d$(printf ',s%.0s' $(seq 1 $((L-1))))"
    run_config "depth_d2048_${L}L" LAYER_PATTERN="$pat" SEED_NUM_BASES=8 NUM_LAYERS=$L MODEL_DIM=2048 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
done

echo "=== All Sweeps Complete ==="
