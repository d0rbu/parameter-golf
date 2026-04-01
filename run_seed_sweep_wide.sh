#!/bin/bash
# Experiment A: Wide model + anchor ablation sweep
# Uses fp8 basis buffers. Default batch settings (8192 tokens, 8 grad accum steps).
# Micro-batch = 1 sequence (1024 tokens) -- already the minimum.

set -e

COMMON="ITERATIONS=200 VAL_LOSS_EVERY=0 TRAIN_BATCH_TOKENS=8192 VAL_BATCH_SIZE=8192 WANDB_ENABLED=1"

echo "=== Wide Model + Anchor Ablation Sweep ==="
echo ""

run_config() {
    local run_id=$1
    shift
    echo "--- Running: $run_id ---"
    env $COMMON RUN_ID="$run_id" "$@" \
        uv run torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 \
        | grep -E "model_params|seed_layers|step:200|Total submission size int8|final_int8_zlib_roundtrip_exact|Error|OOM|RuntimeError" \
        | sed "s/^/  [$run_id] /"
    echo ""
}

# =============================================
# PART 1: dim=1024 (fp8 basis)
# =============================================

run_config "seed_9L_5to1_k64_d1024" NUM_LAYERS=9 SEED_NUM_BASES=64 SEED_ANCHOR_RATIO="5:1" MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4
run_config "seed_12L_5to1_k64_d1024" NUM_LAYERS=12 SEED_NUM_BASES=64 SEED_ANCHOR_RATIO="5:1" MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4
run_config "seed_12L_9to1_k64_d1024" NUM_LAYERS=12 SEED_NUM_BASES=64 SEED_ANCHOR_RATIO="9:1" MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4
run_config "seed_15L_5to1_k32_d1024" NUM_LAYERS=15 SEED_NUM_BASES=32 SEED_ANCHOR_RATIO="5:1" MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4

# =============================================
# PART 2: dim=1536 (fp8 basis)
# =============================================

run_config "seed_9L_5to1_k32_d1536" NUM_LAYERS=9 SEED_NUM_BASES=32 SEED_ANCHOR_RATIO="5:1" MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4
run_config "seed_12L_5to1_k16_d1536" NUM_LAYERS=12 SEED_NUM_BASES=16 SEED_ANCHOR_RATIO="5:1" MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4
run_config "seed_9L_7to1_k32_d1536" NUM_LAYERS=9 SEED_NUM_BASES=32 SEED_ANCHOR_RATIO="7:1" MODEL_DIM=1536 NUM_HEADS=16 NUM_KV_HEADS=4

# =============================================
# PART 3: dim=2048 (fp8 basis)
# =============================================

run_config "seed_6L_4to1_k32_d2048" NUM_LAYERS=6 SEED_NUM_BASES=32 SEED_ANCHOR_RATIO="4:1" MODEL_DIM=2048 NUM_HEADS=16 NUM_KV_HEADS=4
run_config "seed_9L_7to1_k16_d2048" NUM_LAYERS=9 SEED_NUM_BASES=16 SEED_ANCHOR_RATIO="7:1" MODEL_DIM=2048 NUM_HEADS=16 NUM_KV_HEADS=4

# =============================================
# PART 4: Extreme depth + width with good anchor ratios
# =============================================

run_config "seed_20L_3to1_k16_d768" NUM_LAYERS=20 SEED_NUM_BASES=16 SEED_ANCHOR_RATIO="3:1" MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=4
run_config "seed_25L_4to1_k8_d768" NUM_LAYERS=25 SEED_NUM_BASES=8 SEED_ANCHOR_RATIO="4:1" MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=4
run_config "seed_20L_3to1_k16_d1024" NUM_LAYERS=20 SEED_NUM_BASES=16 SEED_ANCHOR_RATIO="3:1" MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=4

# =============================================
# PART 5: Anchor placement ablation (12L all-seed base, k64 d768)
# Uses 100:1 ratio so all layers are naturally seed, then force specific anchors.
# =============================================

run_config "seed_12L_allseed_k64_d768_none" NUM_LAYERS=12 SEED_NUM_BASES=64 SEED_ANCHOR_RATIO="100:1" MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=4 SEED_FORCE_ANCHOR=""
run_config "seed_12L_allseed_k64_d768_first" NUM_LAYERS=12 SEED_NUM_BASES=64 SEED_ANCHOR_RATIO="100:1" MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=4 SEED_FORCE_ANCHOR="0"
run_config "seed_12L_allseed_k64_d768_last" NUM_LAYERS=12 SEED_NUM_BASES=64 SEED_ANCHOR_RATIO="100:1" MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=4 SEED_FORCE_ANCHOR="-1"
run_config "seed_12L_allseed_k64_d768_both" NUM_LAYERS=12 SEED_NUM_BASES=64 SEED_ANCHOR_RATIO="100:1" MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=4 SEED_FORCE_ANCHOR="0,-1"
run_config "seed_12L_allseed_k64_d768_mid" NUM_LAYERS=12 SEED_NUM_BASES=64 SEED_ANCHOR_RATIO="100:1" MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=4 SEED_FORCE_ANCHOR="5"

echo "=== Sweep Complete ==="
echo "Check W&B project 'parameter-golf' for loss curves and artifact sizes."
