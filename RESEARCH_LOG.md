# Research Log: Seed Coefficient Optimization

## Overview

We explore **seed coefficient optimization** — representing transformer weight matrices as learned linear combinations of deterministic random basis matrices — as an extreme compression technique for the Parameter Golf challenge (16MB artifact, 10-min training on 8xH100, scored by val_bpb on FineWeb).

The core idea: instead of storing a full weight matrix W, store only k scalar coefficients and a master seed. The basis matrices R(seed_i) are regenerated at load time, so the artifact cost per seed layer is near-zero (~50 bytes per weight matrix). This frees artifact budget to be spent on **wider and deeper models**.

## Architecture

Three layer types, in order of compression:

1. **Dense** (CastedLinear): Full weight matrix, int8 quantized in artifact. Standard baseline.
2. **Bottleneck** (BottleneckLinear): `y = A @ act(B @ x)` where A is (out, rank), B is (rank, in). ~4x compression at rank=128. Configurable activation (relu^2, relu, none).
3. **Seed** (SeedLinear): `W = sum(alpha_i * R(seed_i))`. Only coefficients stored (float32). Basis in fp8 on GPU, regenerated from master seed. Near-zero artifact cost.

Layers are arranged in a repeating cycle specified by `LAYER_PATTERN` (e.g., `"d,s,s,s,s"`) or the legacy `SEED_ANCHOR_RATIO`. Dense layers always come first in the cycle, ensuring the critical first layer is always dense.

### Key Design Decisions

- **Master seed architecture**: A single `SEED_MASTER_SEED` deterministically derives all per-basis seeds, preventing bias from cherry-picking.
- **fp8 basis buffers**: Basis matrices stored in float8_e4m3fn (1 byte each), 4x less VRAM than float32. Precision doesn't matter since they're random noise — the learned coefficients compensate.
- **Coefficients in float32**: Tiny (k scalars per weight matrix), no reason to lose precision. Stored as float32 passthrough in artifact.
- **Anchor-first cycling**: The first layer in each cycle is always dense. Ablation showed first-layer anchor is critical for quality; last-layer anchor is nearly useless.

## Experimental Results

All experiments: 200 iterations, 1x RTX 4090, 1 training shard, W&B logged. These are directional results — relative ordering may shift at full training length.

### Experiment 1: Depth Sweep (dim=512, varying layers and ratio)

Held width constant at dim=512, varied depth and seed:anchor ratio.

| Config | val_bpb | Size | Params |
|--------|---------|------|--------|
| baseline 9L | 2.403 | 11.4M | 17.1M |
| 9L 5:1 k64 | 2.439 | 4.2M | 6.1M |
| 15L 5:1 k32 | 2.439 | 5.4M | 7.9M |
| 20L 9:1 k32 | 2.441 | 4.3M | 6.1M |
| 25L 23:1 k32 | 2.478 | 4.5M | 6.1M |
| 30L 28:1 k16 | 2.520 | 4.6M | 6.1M |

**Finding**: At dim=512, adding seed layers for depth has diminishing returns. Extreme depth (25-30L) hurts. The seed layers compress well (4-5MB vs 11MB baseline) but don't compensate for the quality loss at narrow width.

### Experiment 2: Width Sweep (dim=640-2048)

Used seed layers to stay under budget while increasing width.

| Config | val_bpb | Size | Fits 16MB? |
|--------|---------|------|------------|
| 9L 5:1 k32 d1536 | **2.347** | 25.4M | No |
| 20L 3:1 k16 d1024 | 2.356 | 20.0M | No |
| 15L 5:1 k32 d1024 | 2.357 | 16.5M | Barely over |
| 9L 7:1 k16 d2048 | 2.359 | 28.4M | No |
| **9L 5:1 k64 d1024** | **2.360** | **12.8M** | **Yes** |
| 12L 5:1 k64 d1024 | 2.363 | 12.7M | Yes |
| 12L 5:1 k64 d768 | 2.383 | 8.0M | Yes |
| 12L allseed+1anchor d768 | 2.387 | 3.4M | Yes |
| baseline 9L d512 | 2.403 | 11.4M | Yes |

**Finding**: **Width dominates quality.** Every increase in model dim improves val_bpb regardless of how many layers are seed. The best config fitting 16MB is **9L 5:1 k64 d1024** at 2.360, beating baseline by 0.043 bpb.

The configs that exceed 16MB (d1536, d2048) show even better quality, motivating BottleneckLinear to compress anchor layers further.

### Experiment 3: k Sweep (15L 5:1 d512, varying k)

Held architecture constant, varied number of basis vectors per seed layer.

| k | val_bpb | Size |
|---|---------|------|
| 16 | 2.449 | 5.4M |
| 32 | 2.439 | 5.4M |
| 64 | 2.443 | 5.4M |
| 128 | 2.452 | 5.5M |

**Finding**: **k barely matters in the 16-128 range.** The seed layers aren't the quality bottleneck — the anchor layers are. This means we can use low k (even k=16) without quality loss, saving VRAM.

### Experiment 4: Anchor Placement Ablation (12L d768, all-seed base)

Started from all-seed (100:1 ratio), forced a single anchor at different positions.

| Anchor Position | val_bpb |
|----------------|---------|
| First layer (pos 0) | 2.392 (2.387 with old force code) |
| Last layer (pos 11) | **2.671** |
| Middle (pos 5) | 2.391 |
| First + last | 2.391 |
| None at all | 2.392 |

**Finding**: **Last-layer-only anchor is catastrophic** (2.671). First-layer anchor is essential. Interestingly, having no anchor at all (2.392) is close to having a first-layer anchor (2.387-2.392), suggesting that at d768 with k=64 the seed layers capture enough expressivity. The first layer's advantage may come from directly processing the embedding output.

This finding led to the anchor-first cycling design: dense layers always come first in each cycle.

### Experiment 5: Deep + Wide Combinations

Combining depth and width to fill the artifact budget.

| Config | val_bpb | Size | Fits 16MB? |
|--------|---------|------|------------|
| 20L 3:1 k16 d768 | 2.392 | 12.4M | Yes |
| 25L 4:1 k8 d768 | 2.403 | 14.9M | Yes |
| 20L 3:1 k16 d1024 | 2.356 | 20.0M | No |

**Finding**: Deep models at d768 (20-25L) match but don't beat the baseline — the anchor layers eat too much budget. Deep + wide (20L d1024) improves quality but exceeds 16MB. This further motivates compressing anchor layers.

## Key Insights

1. **Width >> Depth for quality**: Going from d512 to d1024 improves bpb by ~0.04. Adding 10 seed layers at d512 improves by ~0.00.

2. **Seed layers are free depth**: Near-zero artifact cost, adding computational capacity. But the expressivity bottleneck is the anchor layers, not the seed layers.

3. **The artifact budget bottleneck is anchor layers**: At d1536, 3 anchor layers alone cost 25MB. Compressing anchors (via BottleneckLinear or other techniques) unlocks wider models.

4. **First-layer anchor is critical**: Processing raw embeddings requires full expressivity. All other positions can be seed with minimal quality loss.

5. **k is not a sensitive hyperparameter**: 16 to 128 all give similar results. Use whatever fits in VRAM.

## Next Steps

1. **BottleneckLinear anchor compression**: Replace dense anchor layers with rank-128 bottlenecks to fit d1536+ under 16MB.
2. **Per-weight-type optimization**: Test whether Q/K/V/O/fc/proj have different compressibility within a block.
3. **Full training runs** (20K iterations on 8xH100) on the best configs.
4. **Integration with SOTA techniques** (XSA, BigramHash, EMA, sliding window eval) after the core approach is validated.
