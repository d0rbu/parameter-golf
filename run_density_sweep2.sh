#!/bin/bash
# Density sweep part 2: Q/K bottleneck rank sweep + wider/deeper models
# Key finding from part 1: Q/K can be bottlenecked freely, V/O/fc/proj need density.

set -e

COMMON="ITERATIONS=200 VAL_LOSS_EVERY=0 TRAIN_BATCH_TOKENS=8192 VAL_BATCH_SIZE=8192 WANDB_ENABLED=1"
# Standard weight type config: only Q and K are bottlenecked
QK_BN="WEIGHT_TYPES=q=b,k=b"

echo "=== Density Sweep Part 2: Q/K Rank + Wider/Deeper ==="
echo ""

run_config() {
    local run_id=$1
    shift
    echo "--- Running: $run_id ---"
    env $COMMON RUN_ID="$run_id" "$@" \
        uv run torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 \
        | grep -E "model_params|dense:|layer_types|step:200|Total submission size int8|final_int8_zlib_roundtrip_exact|Error|OOM|RuntimeError" \
        | sed "s/^/  [$run_id] /"
    echo ""
}

# =============================================
# PART 1: Q/K bottleneck rank sweep (12L d1024, 1 dense anchor + 11 seed)
# =============================================

run_config "qk_12L_d1024_r16" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=16
run_config "qk_12L_d1024_r32" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=32
run_config "qk_12L_d1024_r64" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=64
# r=128 already done as wt_12L_d1024_qkbn
run_config "qk_12L_d1024_r256" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=256

# =============================================
# PART 2: More dense anchors at d1024 (Q/K bottlenecked, more budget for anchors)
# =============================================

# 2 dense anchors + 10 seed
run_config "qk_12L_d1024_2d_r128" LAYER_PATTERN="d,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
# 3 dense anchors + 12 seed
run_config "qk_15L_d1024_3d_r128" LAYER_PATTERN="d,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=15 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
# 4 dense anchors + 16 seed
run_config "qk_20L_d1024_4d_r128" LAYER_PATTERN="d,s,s,s,s" SEED_NUM_BASES=16 NUM_LAYERS=20 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128

# =============================================
# PART 3: Push to d1536 with Q/K bottleneck dense anchors
# At d1536, anchor with Q/K bottleneck r=128:
#   V(384x1536) + O(1536x1536) + fc(3072x1536) + proj(1536x3072) = dense
#   Q(bottleneck r=128) + K(bottleneck r=128) = small
#   Total per anchor: ~10.6M params -> ~10.6MB at int8
#   1 anchor -> ~10.6MB, fits under 16MB!
# =============================================

# 1 dense anchor + seed, d1536
run_config "qk_9L_d1536_1d_r128" LAYER_PATTERN="d,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=9 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "qk_12L_d1536_1d_r128" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=16 NUM_LAYERS=12 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "qk_15L_d1536_1d_r128" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=16 NUM_LAYERS=15 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "qk_20L_d1536_1d_r128" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=8 NUM_LAYERS=20 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128

# =============================================
# PART 4: Push to d2048 with Q/K bottleneck dense anchor
# At d2048, anchor with Q/K bottleneck r=128:
#   V(512x2048) + O(2048x2048) + fc(4096x2048) + proj(2048x4096) = dense
#   Total per anchor: ~18.9M -> needs aggressive compression elsewhere
# =============================================

run_config "qk_9L_d2048_1d_r128" LAYER_PATTERN="d,s,s,s,s,s,s,s,s" SEED_NUM_BASES=16 NUM_LAYERS=9 MODEL_DIM=2048 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128
run_config "qk_6L_d2048_1d_r128" LAYER_PATTERN="d,s,s,s,s,s" SEED_NUM_BASES=16 NUM_LAYERS=6 MODEL_DIM=2048 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=128

# =============================================
# PART 5: d1536 with Q/K bottleneck at different ranks
# =============================================

run_config "qk_12L_d1536_1d_r32" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=16 NUM_LAYERS=12 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=32
run_config "qk_12L_d1536_1d_r64" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=16 NUM_LAYERS=12 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=64
run_config "qk_12L_d1536_1d_r256" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=16 NUM_LAYERS=12 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4 $QK_BN BOTTLENECK_RANK=256

# =============================================
# PART 6: Also bottleneck V in the anchor (V was sensitive, but maybe at higher rank?)
# =============================================

run_config "qkv_12L_d1024_r128" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 WEIGHT_TYPES="q=b,k=b,v=b" BOTTLENECK_RANK=128
run_config "qkv_12L_d1024_r256" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 WEIGHT_TYPES="q=b,k=b,v=b" BOTTLENECK_RANK=256
run_config "qkvo_12L_d1024_r256" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 WEIGHT_TYPES="q=b,k=b,v=b,o=b" BOTTLENECK_RANK=256

echo "=== Sweep Complete ==="
