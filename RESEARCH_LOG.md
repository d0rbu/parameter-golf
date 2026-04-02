# Research Log: Seed Coefficient Optimization

## Overview

We explore **seed coefficient optimization** — representing transformer weight matrices as learned linear combinations of deterministic random basis matrices — as an extreme compression technique for the Parameter Golf challenge (16MB artifact, 10-min training on 8xH100, scored by val_bpb on FineWeb).

The core idea: instead of storing a full weight matrix W, store only k scalar coefficients and a master seed. The basis matrices R(seed_i) are regenerated at load time, so the artifact cost per seed layer is near-zero (~50 bytes per weight matrix). This frees artifact budget to be spent on **wider and deeper models**.

## Architecture

Three layer types, in order of compression:

1. **Dense** (CastedLinear): Full weight matrix, int8 quantized in artifact. Standard baseline.
2. **Bottleneck** (BottleneckLinear): `y = A @ act(B @ x)` where A is (out, rank), B is (rank, in). ~4x compression at rank=128. Configurable activation.
3. **Seed** (SeedLinear): `W = sum(alpha_i * R(seed_i))`. Only coefficients stored (float32). Basis in fp8 on GPU, regenerated from master seed. Near-zero artifact cost. Cached W optimization: W is materialized once per optimizer step and reused across gradient accumulation micro-steps.

Layers are arranged in a repeating cycle specified by `LAYER_PATTERN` (e.g., `"d,s,s,s,s"`) or the legacy `SEED_ANCHOR_RATIO`. Dense layers always come first in the cycle, ensuring the critical first layer is always dense.

Per-weight-type overrides via `WEIGHT_TYPES` (e.g., `"q=b,k=b"`) allow mixing dense and bottleneck weights within a single block.

### Key Design Decisions

- **Master seed architecture**: A single `SEED_MASTER_SEED` deterministically derives all per-basis seeds, preventing bias from cherry-picking.
- **fp8 basis buffers**: Basis matrices stored in float8_e4m3fn (1 byte each), 4x less VRAM than float32. Precision doesn't matter since they're random noise — the learned coefficients compensate.
- **Cached W**: The materialized weight `W = einsum(coeffs, basis)` is cached as a `requires_grad` leaf tensor after each optimizer step, avoiding redundant re-materialization during gradient accumulation (7 of 8 forward passes reuse cached W). Gives ~16% speedup.
- **Coefficients in float32**: Tiny (k scalars per weight matrix), no reason to lose precision.
- **Anchor-first cycling**: Dense layers always come first in cycle (DSSSSS...) — first-layer anchor is critical for quality, last-layer anchor is optional.

## Experimental Results

All experiments: 200 iterations, 1x RTX 4090, W&B logged. These are directional results — relative ordering may shift at full training length (20K iterations on 8xH100).

### Experiment 1: Depth Sweep at dim=512

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

**Finding**: **Width dominates quality.** Every increase in model dim improves val_bpb regardless of how many layers are seed. The best config fitting 16MB (before bottleneck anchors) is **9L 5:1 k64 d1024** at 2.360, beating baseline by 0.043 bpb.

### Experiment 3: k Sweep (15L 5:1 d512, varying k)

| k | val_bpb | Size |
|---|---------|------|
| 16 | 2.449 | 5.4M |
| 32 | 2.439 | 5.4M |
| 64 | 2.443 | 5.4M |
| 128 | 2.452 | 5.5M |

**Finding**: **k barely matters in the 16-128 range.** Use whatever fits in VRAM.

### Experiment 4: Anchor Placement Ablation (12L d768)

Started from all-seed (100:1 ratio), forced a single anchor at different positions.

| Anchor Position | val_bpb |
|----------------|---------|
| First layer (pos 0) | 2.387 |
| Last layer (pos 11) | **2.671** |
| Middle (pos 5) | 2.391 |
| First + last | 2.391 |
| None at all | 2.392 |

**Finding**: **Last-layer-only anchor is catastrophic** (2.671). First-layer anchor is essential. This led to the anchor-first cycling design.

### Experiment 5: Per-Weight-Type Sensitivity (12L d1024, 1 dense anchor + 11 seed)

Tested selectively bottlenecking different weight matrices within the dense anchor layer at rank=128.

| What's bottlenecked | bpb | Delta vs all-dense |
|---------------------|-----|--------------------|
| Nothing (all dense) | 2.377 | -- |
| **Q + K only** | **2.374** | **-0.003** (better!) |
| V + O | 2.633 | +0.256 |
| All attention (Q/K/V/O) | 2.744 | +0.368 |
| MLP (fc/proj) | 2.589 | +0.212 |
| Everything | 2.966 | +0.589 |

**Finding**: **Q and K can be bottlenecked with zero quality loss** — they learn rotation subspaces for attention, which are inherently low-rank. V and O are very sensitive at r=128. MLP is intermediate.

### Experiment 6: Bottleneck Activation Ablation

Tested bottleneck activation at d1536, 1 bottleneck anchor + 8 seed, r=128.

| Activation | bpb |
|-----------|-----|
| none (linear) | **2.524** |
| relu | 2.566 |
| relu^2 | 3.426 |

**Finding**: **Linear (no activation) is best** for bottleneck layers. The bottleneck functions as a pure low-rank factorization. relu^2 kills information in the narrow bottleneck.

### Experiment 7: Q/K Bottleneck Rank Sweep (12L d1024)

| Rank | bpb | Size |
|------|-----|------|
| 16 | 2.373 | 4.6M |
| **32** | **2.363** | 4.7M |
| 64 | 2.364 | 4.8M |
| 128 | 2.374 | 4.8M |
| 256 | 2.367 | 5.1M |

**Finding**: **Q/K bottleneck rank 32 is the sweet spot** at d1024. Even r=16 works. The Q/K projections are extremely low-rank.

### Experiment 8: Q/K Bottleneck Rank at d1536

| Rank | bpb | Size |
|------|-----|------|
| 32 | 2.350 | 10.2M |
| 64 | 2.358 | 10.2M |
| 128 | 2.365 | 8.9M |
| 256 | 2.353 | 10.2M |

**Finding**: Rank insensitivity holds at d1536 too. Q/K can use very low rank regardless of model width.

### Experiment 9: V/O Bottleneck at Higher Rank (12L d1024)

| Config | bpb | Delta vs all-dense |
|--------|-----|--------------------|
| Q/K bottleneck r=128 | 2.374 | -0.003 |
| Q/K/V bottleneck r=128 | 2.378 | +0.001 |
| Q/K/V bottleneck r=256 | 2.380 | +0.003 |
| Q/K/V/O bottleneck r=256 | 2.377 | +0.000 |

**Finding**: At r=256, even V and O can be bottlenecked without significant damage (2.377 vs 2.377 all-dense). The sensitivity from Experiment 5 was due to r=128 being too low for V/O, not a fundamental limitation.

### Experiment 10: Depth at Width (1 dense anchor + N seed layers)

The key experiment: does adding seed layers for depth help at wider models?

**d1024 depth curve:**
| Depth | bpb | ms/step |
|-------|-----|---------|
| **6L** | **2.369** | 75 |
| 9L | 2.371 | 114 |
| 12L | 2.381 | 149 |
| 15L | 2.385 | 129 |
| 20L | 2.391 | 173 |
| 25L | 2.382 | 159 |
| 30L | 2.400 | 189 |

**d1536 depth curve:**
| Depth | bpb | ms/step |
|-------|-----|---------|
| 6L | 2.372 | 106 |
| 9L | 2.358 | 156 |
| **12L** | **2.357** | 152 |
| 15L | 2.356 | 186 |
| 20L | 2.358 | 252 |
| 25L | 2.370 | 305 |

**d2048 depth curve:**
| Depth | bpb | ms/step |
|-------|-----|---------|
| 6L | 2.362 | 119 |
| 9L | 2.354 | 172 |
| **12L** | **2.350** | 228 |
| 15L | 2.358 | 281 |

**Finding**: **Depth helps at wider models but not narrow ones.**
- At d1024, 6L is best — more seed layers hurt.
- At d1536, depth helps up to 12-15L, then plateaus and degrades.
- At d2048, depth helps up to 12L, then starts declining.

There's a **depth sweet spot per width**: d1024→6L, d1536→12-15L, d2048→12L. Beyond this, additional seed layers add compute cost without quality gain. Wider models can leverage more depth from seed layers, likely because the wider hidden state has more capacity for the seed layers to usefully transform.

### Experiment 11: Full Density Allocation with Q/K Bottleneck Anchors

Combining Q/K bottleneck with wider models to push under 16MB.

| Config | bpb | Size | ms/step |
|--------|-----|------|---------|
| **9L d2048 Q/K bn r128** | **2.334** | **13.7M** | 293 |
| 12L d2048 depth | 2.350 | 15.9M | 228 |
| 12L d1536 Q/K bn r32 | 2.350 | 10.2M | 213 |
| 15L d1536 depth | 2.356 | 10.3M | 186 |
| 6L d2048 Q/K bn r128 | 2.361 | 15.8M | 178 |
| 15L d1024 3 dense Q/K bn | 2.363 | 11.2M | 208 |
| baseline 9L d512 | 2.403 | 11.4M | 53 |

**Finding**: **d2048 fits under 16MB with Q/K bottleneck anchors and wins outright.** The 9L d2048 config at 13.7MB achieves 2.334 bpb — a 0.069 bpb improvement over baseline. The Q/K bottleneck saves enough anchor budget to enable d2048 while staying under the cap.

## Key Insights

1. **Width >> Depth for quality**: Going from d512 to d2048 improves bpb by ~0.07. Adding seed layers at d512 improves by ~0.00.

2. **Seed layers are free depth in artifact terms**: Near-zero artifact cost, but their quality contribution depends on width. At narrow widths (d512-d1024) they don't help and can hurt. At wider widths (d1536+) they help up to a sweet spot.

3. **Depth sweet spot scales with width**: d1024→6L, d1536→12-15L, d2048→9-12L. Beyond this, more seed layers add compute without quality gain.

4. **The artifact budget bottleneck is anchor layers**: A single dense anchor at d2048 costs ~14MB. Q/K bottleneck saves ~2MB per anchor, enabling wider models to fit.

5. **First-layer anchor is critical**: Processing raw embeddings requires full expressivity. All other positions can be seed with minimal quality loss.

6. **k is not a sensitive hyperparameter**: 16 to 128 all give similar results. Use low k (16-32) to save VRAM and compute.

7. **Q/K projections are inherently low-rank**: Can be bottlenecked at r=32 with zero quality loss at any model width. V/O need r=256+ to be bottlenecked safely.

8. **Bottleneck activation should be linear**: The bottleneck works as a low-rank factorization, not a nonlinear module.

9. **Training speed matters**: Seed layers add significant forward-pass overhead from materializing W via einsum (reading fp8 basis tensors from VRAM). The cached W optimization helps (~16% speedup) but wider+deeper seed models are still ~3-6x slower per step than baseline. At d2048, the 10-minute training budget on 8xH100 allows ~16-20K steps.

## Recommended Configurations for Competition

| Priority | Config | bpb | Size | Est. 8xH100 steps |
|----------|--------|-----|------|-------------------|
| 1 | 9L d2048, 1 dense anchor (Q/K bn r128) | 2.334 | 13.7M | ~16K |
| 2 | 12L d2048, 1 dense anchor (Q/K bn r128) | 2.350 | 15.9M | ~20K |
| 3 | 12L d1536, 1 dense anchor (Q/K bn r32) | 2.350 | 10.2M | ~22K |
| 4 | 15L d1536, 1 dense anchor (Q/K bn r128) | 2.356 | 10.3M | ~25K |

Config 1 has the best quality but fewest training steps. Config 3 is the safe choice — similar quality with more steps and 6MB of artifact headroom. Full training runs on 8xH100 are needed to determine which converges best.

## Next Steps

1. **Full training runs** (20K iterations on 8xH100) on top configs to get real val_bpb numbers.
2. **Integration with SOTA techniques** (XSA, BigramHash, EMA, sliding window eval) — these are orthogonal to the compression technique and should stack.
3. **Experiment B (Low-Rank + Sparse)** on a separate branch — a different compression paradigm that may complement or outperform seed coefficients.
4. **Training speed optimization** — further reduce seed layer overhead for wider models (kernel fusion, precision reduction, better caching).
