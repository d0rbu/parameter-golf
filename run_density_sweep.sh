#!/bin/bash
# Density budget allocation sweep: test combinations of dense/bottleneck/seed
# per weight type and across widths/depths to find the pareto frontier.

set -e

COMMON="ITERATIONS=200 VAL_LOSS_EVERY=0 TRAIN_BATCH_TOKENS=8192 VAL_BATCH_SIZE=8192 WANDB_ENABLED=1"

echo "=== Density Budget Allocation Sweep ==="
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
# PART 1: Bottleneck anchors at d1536 (target: fit under 16MB)
# Previous finding: d1536 with dense anchors = 25MB (too big).
# Bottleneck rank=128 should compress anchors ~4x -> ~6MB per anchor.
# =============================================

# All-bottleneck d1536 (no dense layers at all)
run_config "bn_9L_d1536_allbn_r128" LAYER_PATTERN="b" SEED_NUM_BASES=0 BOTTLENECK_RANK=128 NUM_LAYERS=9 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4

# 1 bottleneck anchor + 8 seed, d1536
run_config "bn_9L_d1536_1bn8s_r128" LAYER_PATTERN="b,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 BOTTLENECK_RANK=128 NUM_LAYERS=9 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4

# 2 bottleneck anchors + seed, d1536
run_config "bn_12L_d1536_2bn10s_r128" LAYER_PATTERN="b,s,s,s,s,s" SEED_NUM_BASES=32 BOTTLENECK_RANK=128 NUM_LAYERS=12 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4

# Higher rank bottleneck
run_config "bn_9L_d1536_1bn8s_r256" LAYER_PATTERN="b,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 BOTTLENECK_RANK=256 NUM_LAYERS=9 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4

# =============================================
# PART 2: Bottleneck anchors at d2048
# =============================================

run_config "bn_6L_d2048_1bn5s_r128" LAYER_PATTERN="b,s,s,s,s,s" SEED_NUM_BASES=16 BOTTLENECK_RANK=128 NUM_LAYERS=6 MODEL_DIM=2048 NUM_HEADS=16 NUM_KV_HEADS=4

run_config "bn_9L_d2048_1bn8s_r128" LAYER_PATTERN="b,s,s,s,s,s,s,s,s" SEED_NUM_BASES=16 BOTTLENECK_RANK=128 NUM_LAYERS=9 MODEL_DIM=2048 NUM_HEADS=16 NUM_KV_HEADS=4

run_config "bn_9L_d2048_1bn8s_r256" LAYER_PATTERN="b,s,s,s,s,s,s,s,s" SEED_NUM_BASES=16 BOTTLENECK_RANK=256 NUM_LAYERS=9 MODEL_DIM=2048 NUM_HEADS=16 NUM_KV_HEADS=4

# =============================================
# PART 3: Per-weight-type allocation (which weights need density?)
# Base: 12L d1024, 1 dense anchor + 11 seed
# Test: selectively make some weights in the dense layer bottleneck
# =============================================

# Baseline: all dense anchor weights
run_config "wt_12L_d1024_alldense" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4

# Attention bottleneck, MLP dense
run_config "wt_12L_d1024_attnbn" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 WEIGHT_TYPES="q=b,k=b,v=b,o=b" BOTTLENECK_RANK=128

# MLP bottleneck, attention dense
run_config "wt_12L_d1024_mlpbn" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 WEIGHT_TYPES="fc=b,proj=b" BOTTLENECK_RANK=128

# Only Q and K bottleneck (they just learn rotations)
run_config "wt_12L_d1024_qkbn" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 WEIGHT_TYPES="q=b,k=b" BOTTLENECK_RANK=128

# Only V and O bottleneck
run_config "wt_12L_d1024_vobn" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 WEIGHT_TYPES="v=b,o=b" BOTTLENECK_RANK=128

# Everything bottleneck (same as full bottleneck layer)
run_config "wt_12L_d1024_allbn" LAYER_PATTERN="d,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4 WEIGHT_TYPES="q=b,k=b,v=b,o=b,fc=b,proj=b" BOTTLENECK_RANK=128

# =============================================
# PART 4: Bottleneck rank sweep (d1536, 1 anchor + 8 seed)
# =============================================

run_config "bn_9L_d1536_1bn8s_r32" LAYER_PATTERN="b,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 BOTTLENECK_RANK=32 NUM_LAYERS=9 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4

run_config "bn_9L_d1536_1bn8s_r64" LAYER_PATTERN="b,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 BOTTLENECK_RANK=64 NUM_LAYERS=9 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4

run_config "bn_9L_d1536_1bn8s_r512" LAYER_PATTERN="b,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 BOTTLENECK_RANK=512 NUM_LAYERS=9 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4

# =============================================
# PART 5: Wide + deep + mixed (pushing the frontier)
# =============================================

# 20L d1536, 2 bottleneck + 18 seed
run_config "bn_20L_d1536_2bn18s_r128" LAYER_PATTERN="b,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=16 BOTTLENECK_RANK=128 NUM_LAYERS=20 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4

# 15L d2048, 1 bottleneck + 14 seed
run_config "bn_15L_d2048_1bn14s_r128" LAYER_PATTERN="b,s,s,s,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=8 BOTTLENECK_RANK=128 NUM_LAYERS=15 MODEL_DIM=2048 NUM_HEADS=16 NUM_KV_HEADS=4

# 12L d1024, 1 dense + 1 bottleneck + 10 seed (mix all three)
run_config "mix_12L_d1024_1d1b10s_r128" LAYER_PATTERN="d,b,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 BOTTLENECK_RANK=128 NUM_LAYERS=12 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4

# 15L d1024, 1 dense + 2 bottleneck + 12 seed
run_config "mix_15L_d1024_1d2b12s_r128" LAYER_PATTERN="d,b,b,s,s,s,s,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 BOTTLENECK_RANK=128 NUM_LAYERS=15 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4

# =============================================
# PART 6: Activation function ablation for bottleneck
# =============================================

run_config "bn_9L_d1536_r128_relu" LAYER_PATTERN="b,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 BOTTLENECK_RANK=128 BOTTLENECK_ACT=relu NUM_LAYERS=9 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4

run_config "bn_9L_d1536_r128_none" LAYER_PATTERN="b,s,s,s,s,s,s,s,s" SEED_NUM_BASES=32 BOTTLENECK_RANK=128 BOTTLENECK_ACT=none NUM_LAYERS=9 MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4

echo "=== Sweep Complete ==="
echo "Check W&B project 'parameter-golf' for loss curves and artifact sizes."
