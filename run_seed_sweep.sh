#!/bin/bash
# Experiment A: Pareto frontier sweep for seed coefficient optimization
# Runs configs sequentially on a single GPU, collecting key metrics.
# Logs to W&B for loss curve comparison.

set -e

COMMON="ITERATIONS=200 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 TRAIN_BATCH_TOKENS=8192 WANDB_ENABLED=1"

echo "=== Seed Coefficient Pareto Frontier Sweep ==="
echo ""

run_config() {
    local run_id=$1
    shift
    echo "--- Running: $run_id ---"
    env $COMMON RUN_ID="$run_id" "$@" \
        uv run torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 \
        | grep -E "model_params|seed_layers|step:200|Total submission size int8|final_int8_zlib_roundtrip_exact" \
        | sed "s/^/  [$run_id] /"
    echo ""
}

# =============================================
# PART 1: Depth sweep (hold dim=512, vary layers & ratio)
# =============================================

# Baseline (no seed layers)
run_config "baseline_9L" NUM_LAYERS=9 SEED_NUM_BASES=0

# 9 layers, 5:1 ratio, k=64
run_config "seed_9L_5to1_k64" NUM_LAYERS=9 SEED_NUM_BASES=64 SEED_ANCHOR_RATIO="5:1"

# 15 layers, 5:1 ratio, k=64
run_config "seed_15L_5to1_k64" NUM_LAYERS=15 SEED_NUM_BASES=64 SEED_ANCHOR_RATIO="5:1"

# 20 layers, 9:1 ratio, k=32
run_config "seed_20L_9to1_k32" NUM_LAYERS=20 SEED_NUM_BASES=32 SEED_ANCHOR_RATIO="9:1"

# 15 layers, 5:1 ratio, k=32 (lower k)
run_config "seed_15L_5to1_k32" NUM_LAYERS=15 SEED_NUM_BASES=32 SEED_ANCHOR_RATIO="5:1"

# 15 layers, 13:1 ratio, k=64 (only first+last anchor)
run_config "seed_15L_13to1_k64" NUM_LAYERS=15 SEED_NUM_BASES=64 SEED_ANCHOR_RATIO="13:1"

# =============================================
# PART 2: Width sweep (increase dim, use seed layers to fit budget)
# =============================================

# 12 layers, 5:1 ratio, k=64, dim=640
run_config "seed_12L_5to1_k64_d640" NUM_LAYERS=12 SEED_NUM_BASES=64 SEED_ANCHOR_RATIO="5:1" MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5

# 9 layers, 5:1 ratio, k=64, dim=768 (much wider)
run_config "seed_9L_5to1_k64_d768" NUM_LAYERS=9 SEED_NUM_BASES=64 SEED_ANCHOR_RATIO="5:1" MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=4

# 12 layers, 5:1 ratio, k=64, dim=768
run_config "seed_12L_5to1_k64_d768" NUM_LAYERS=12 SEED_NUM_BASES=64 SEED_ANCHOR_RATIO="5:1" MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=4

# 15 layers, 9:1 ratio, k=32, dim=640 (wide + deep)
run_config "seed_15L_9to1_k32_d640" NUM_LAYERS=15 SEED_NUM_BASES=32 SEED_ANCHOR_RATIO="9:1" MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5

# =============================================
# PART 3: Aggressive ratios (maximize seed layers)
# =============================================

# 20 layers, 18:1 ratio, k=64 (only first+last anchor)
run_config "seed_20L_18to1_k64" NUM_LAYERS=20 SEED_NUM_BASES=64 SEED_ANCHOR_RATIO="18:1"

# 25 layers, 23:1 ratio, k=32 (extreme depth, only first+last anchor)
run_config "seed_25L_23to1_k32" NUM_LAYERS=25 SEED_NUM_BASES=32 SEED_ANCHOR_RATIO="23:1"

# 30 layers, 28:1 ratio, k=16 (ultra extreme)
run_config "seed_30L_28to1_k16" NUM_LAYERS=30 SEED_NUM_BASES=16 SEED_ANCHOR_RATIO="28:1"

# =============================================
# PART 4: k sweep (hold 15L 5:1, vary k)
# =============================================

# k=16
run_config "seed_15L_5to1_k16" NUM_LAYERS=15 SEED_NUM_BASES=16 SEED_ANCHOR_RATIO="5:1"

# k=128
run_config "seed_15L_5to1_k128" NUM_LAYERS=15 SEED_NUM_BASES=128 SEED_ANCHOR_RATIO="5:1"

# =============================================
# PART 5: Width + aggressive ratio combos
# =============================================

# 20 layers, 18:1 ratio, k=32, dim=640
run_config "seed_20L_18to1_k32_d640" NUM_LAYERS=20 SEED_NUM_BASES=32 SEED_ANCHOR_RATIO="18:1" MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5

# 15 layers, 13:1 ratio, k=64, dim=768
run_config "seed_15L_13to1_k64_d768" NUM_LAYERS=15 SEED_NUM_BASES=64 SEED_ANCHOR_RATIO="13:1" MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=4

echo "=== Sweep Complete ==="
echo "Check W&B project 'parameter-golf' for loss curves and artifact sizes."
