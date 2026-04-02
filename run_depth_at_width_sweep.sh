#!/bin/bash
# Depth sweep at large widths: does adding more seed layers help at d1024/d1536/d2048?
# All configs use 1 dense anchor (Q/K bottlenecked) + N seed layers.
# Artifact size is dominated by the single anchor -- depth is nearly free.

set -e

COMMON="ITERATIONS=200 VAL_LOSS_EVERY=0 TRAIN_BATCH_TOKENS=8192 VAL_BATCH_SIZE=8192 WANDB_ENABLED=1"
QK_BN="WEIGHT_TYPES=q=b,k=b"

echo "=== Depth at Width Sweep ==="
echo ""

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

# =============================================
# d1024: depth sweep with 1 Q/K-bottleneck anchor
# Anchor cost: ~4.8MB. Budget for seed: ~11MB (but seeds cost ~0).
# =============================================

run_config "depth_d1024_6L"  LAYER_PATTERN="d,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=6  MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "depth_d1024_9L"  LAYER_PATTERN="d,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=9  MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "depth_d1024_12L" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "depth_d1024_15L" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=16 NUM_LAYERS=15 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "depth_d1024_20L" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=16 NUM_LAYERS=20 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "depth_d1024_25L" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=8 NUM_LAYERS=25 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "depth_d1024_30L" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=8 NUM_LAYERS=30 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128

# =============================================
# d1536: depth sweep with 1 Q/K-bottleneck anchor
# Anchor cost: ~10MB. Tight but should fit under 16MB.
# =============================================

run_config "depth_d1536_6L"  LAYER_PATTERN="d,s,s,s,s,s" SEED_NUM_BASES=16 NUM_LAYERS=6  MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "depth_d1536_9L"  LAYER_PATTERN="d,s,s,s,s,s,s,s,s" SEED_NUM_BASES=16 NUM_LAYERS=9  MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "depth_d1536_12L" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=16 NUM_LAYERS=12 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "depth_d1536_15L" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=8 NUM_LAYERS=15 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "depth_d1536_20L" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=8 NUM_LAYERS=20 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "depth_d1536_25L" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=8 NUM_LAYERS=25 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128

# =============================================
# d2048: depth sweep with 1 Q/K-bottleneck anchor
# Anchor cost: ~19MB -- over 16MB even with 1 anchor.
# Still useful for understanding the depth trend.
# =============================================

run_config "depth_d2048_6L"  LAYER_PATTERN="d,s,s,s,s,s" SEED_NUM_BASES=8 NUM_LAYERS=6  MODEL_DIM=2048 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "depth_d2048_9L"  LAYER_PATTERN="d,s,s,s,s,s,s,s,s" SEED_NUM_BASES=8 NUM_LAYERS=9  MODEL_DIM=2048 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "depth_d2048_12L" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=8 NUM_LAYERS=12 MODEL_DIM=2048 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "depth_d2048_15L" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=8 NUM_LAYERS=15 MODEL_DIM=2048 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128

echo "=== Sweep Complete ==="
