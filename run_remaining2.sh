#!/bin/bash
# Remaining configs: 4 density + 17 depth = 21 total
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

echo "=== Remaining Density (4 configs) ==="

run_config "qk_12L_d1536_1d_r256" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=16 NUM_LAYERS=12 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=256
run_config "qkv_12L_d1024_r128" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 WEIGHT_TYPES="q=b,k=b,v=b" BOTTLENECK_RANK=128
run_config "qkv_12L_d1024_r256" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 WEIGHT_TYPES="q=b,k=b,v=b" BOTTLENECK_RANK=256
run_config "qkvo_12L_d1024_r256" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 WEIGHT_TYPES="q=b,k=b,v=b,o=b" BOTTLENECK_RANK=256

echo "=== Depth at d1024 (7 configs) ==="

for L in 6 9 12 15 20 25 30; do
    k=$( [ $L -le 12 ] && echo 32 || ( [ $L -le 20 ] && echo 16 || echo 8 ) )
    pat="d$(printf ',s%.0s' $(seq 1 $((L-1))))"
    run_config "depth_d1024_${L}L" LAYER_PATTERN="$pat" SEED_NUM_BASES=$k NUM_LAYERS=$L MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
done

echo "=== Depth at d1536 (6 configs) ==="

for L in 6 9 12 15 20 25; do
    k=$( [ $L -le 9 ] && echo 16 || echo 8 )
    pat="d$(printf ',s%.0s' $(seq 1 $((L-1))))"
    run_config "depth_d1536_${L}L" LAYER_PATTERN="$pat" SEED_NUM_BASES=$k NUM_LAYERS=$L MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
done

echo "=== Depth at d2048 (4 configs) ==="

for L in 6 9 12 15; do
    pat="d$(printf ',s%.0s' $(seq 1 $((L-1))))"
    run_config "depth_d2048_${L}L" LAYER_PATTERN="$pat" SEED_NUM_BASES=8 NUM_LAYERS=$L MODEL_DIM=2048 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
done

echo "=== All Done ==="
