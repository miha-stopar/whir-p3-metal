# whir-p3

A version of https://github.com/WizardOfMenlo/whir/ which uses the Plonky3 library.

## GPU acceleration (Apple Silicon / Metal)

Enable with `--features gpu-metal`. Accelerates the `whir_prove` pipeline by
offloading NTT (Number Theoretic Transform) and Merkle tree construction to the
GPU via Metal compute shaders.

### Architecture

The GPU pipeline lives in two files:

- **`shaders/babybear_ntt.metal`** — Metal Shading Language kernels for BabyBear
  field arithmetic, DIF-NTT butterfly stages, bit-reversal, and Poseidon2 hashing.
- **`src/gpu_dft.rs`** — Rust orchestration: Metal pipeline setup, buffer
  management, kernel dispatch, and the `GpuMmcs` wrapper that plugs into
  Plonky3's `Mmcs` trait.

### Key optimizations

**1. Radix-16 DIF NTT with zero-copy I/O**

NTT uses decimation-in-frequency (DIF) with fused radix-16/8/4/2 butterfly
kernels. On Apple Silicon's unified memory, input data is wrapped as a
zero-copy Metal buffer (no CPU→GPU copy). DIF stages run in-place on a
GPU-managed buffer, and the final bit-reversal gather writes results back to
the zero-copy buffer with coalesced sequential writes.

**2. GPU Poseidon2 Merkle tree**

Merkle tree construction uses Poseidon2 (width-16, 8+13+4 rounds, x^7 S-box)
implemented entirely in Metal. Leaf hashing and all compression layers are
dispatched in a single command buffer — Apple GPU guarantees dispatch ordering
within a compute command encoder, so no explicit barriers are needed.

**3. Fused DFT → Merkle pipeline (`DftCommitFusion` trait)**

Instead of running DFT on GPU, copying the result back to CPU, and
re-uploading for Merkle hashing, the fused path runs DFT + bit-reversal +
Poseidon2 leaf hashing + all Merkle compression layers in a **single GPU
command buffer** with zero CPU round-trips between stages.

This is exposed via the `DftCommitFusion<F>` trait, which `GpuMmcs`
implements. The fusion is used in two places:

1. **Initial commit** — `CommitmentWriter::commit_fused()` fuses the base-field
   DFT + Merkle for the initial polynomial commitment.
2. **Per-round commits** — `Prover::prove_fused()` fuses the extension-field
   DFT (`dft_algebra_batch`) + Merkle for every STIR round that has a large
   enough matrix. Falls back to separate DFT + commit when the matrix is
   below the GPU threshold (8 MB).

The fused pipeline for a single commit:

```
CPU input (zero-copy) → R16 OOP → DIF stages (managed) → bitrev gather (managed→zero-copy)
    → Poseidon2 leaf hash → compress layers → [single wait] → result already in CPU memory
```

The bitrev gather writes directly back into the zero-copy buffer (the
caller's `values` Vec), and Merkle hashing reads from the same buffer.
This eliminates both a separate GPU buffer allocation and the full-matrix
memcpy that was previously needed after GPU completion.

vs the non-fused pipeline:

```
CPU input (zero-copy) → DIF stages → bitrev → CPU    [GPU command buffer 1, wait]
CPU → upload to GPU → leaf hash → compress layers     [GPU command buffer 2, wait]
```

**4. Montgomery arithmetic in Metal**

BabyBear field operations use Montgomery form throughout. The Montgomery
multiply uses only 32-bit `mul`/`mulhi` instructions (no 64-bit arithmetic),
mapping efficiently to Apple GPU's ALU.

### Benchmarks

All benchmarks on Apple M-series silicon (unified memory). Best of 3 runs
(median). Parameters:
- `n` = `num_variables` (polynomial has 2^n coefficients)
- `fold` = `folding_factor` (each STIR round folds 2^fold evaluations)
- `rate` = `starting_log_inv_rate` (RS code rate = 1/2^rate, domain = 2^(n+rate) points)
- **GPU** = fused initial commit only (rounds use separate DFT + commit)
- **Fused** = fused initial commit + fused per-round DFT+Merkle (`prove_fused`)

#### n=18 (256K coefficients)

| fold | rate | CPU (ms) | GPU (ms) | Fused (ms) | GPU | Fused |
|------|------|----------|----------|------------|-----|-------|
| 4 | 1 | 15 | 20 | 23 | 0.77x | 0.66x |
| 4 | 2 | 31 | 34 | 31 | 0.90x | 0.97x |
| 4 | 3 | 36 | 42 | 48 | 0.85x | 0.75x |
| 6 | 1 | 10 | 21 | 17 | 0.48x | 0.61x |
| 6 | 2 | 18 | 19 | 23 | 0.94x | 0.79x |
| 6 | 3 | 24 | 28 | 34 | 0.88x | 0.71x |
| 8 | 1 | 12 | 16 | 16 | 0.77x | 0.74x |
| 8 | 2 | 16 | 21 | 26 | 0.78x | 0.62x |
| 8 | 3 | 29 | 36 | 40 | 0.81x | 0.74x |

> GPU overhead dominates at this size. All configs slower than CPU.

#### n=20 (1M coefficients)

| fold | rate | CPU (ms) | GPU (ms) | Fused (ms) | GPU | Fused |
|------|------|----------|----------|------------|-----|-------|
| 4 | 1 | 53 | 55 | 59 | 0.97x | 0.90x |
| 4 | 2 | 87 | 92 | 78 | 0.94x | **1.11x** |
| 4 | 3 | 186 | 196 | 143 | 0.95x | **1.30x** |
| 6 | 1 | 47 | 49 | 50 | 0.96x | 0.93x |
| 6 | 2 | 114 | 92 | 108 | **1.23x** | 1.05x |
| 6 | 3 | 611 | 439 | 516 | **1.39x** | **1.18x** |
| 8 | 1 | 26 | 32 | 34 | 0.83x | 0.78x |
| 8 | 2 | 52 | 45 | 48 | **1.16x** | 1.08x |
| 8 | 3 | 100 | 86 | 83 | **1.16x** | **1.20x** |

> GPU starts winning at rate=2-3. Crossover around 50-100 ms CPU time.

#### n=22 (4M coefficients)

| fold | rate | CPU (ms) | GPU (ms) | Fused (ms) | GPU | Fused |
|------|------|----------|----------|------------|-----|-------|
| 4 | 1 | 192 | 153 | 132 | **1.26x** | **1.45x** |
| 4 | 2 | 354 | 256 | 232 | **1.38x** | **1.53x** |
| 4 | 3 | 780 | 656 | 687 | **1.19x** | **1.13x** |
| 6 | 1 | 168 | fail | 134 | - | **1.25x** |
| **6** | **2** | **678** | **424** | **361** | **1.60x** | **1.88x** |
| 6 | 3 | 2575 | 1767 | 2381 | **1.46x** | 1.08x |
| 8 | 1 | 95 | 71 | 73 | **1.34x** | **1.30x** |
| 8 | 2 | 193 | 182 | 161 | 1.06x | **1.19x** |
| 8 | 3 | 465 | 344 | 321 | **1.35x** | **1.45x** |

> Best result: **1.88x** at fold=6, rate=2.

#### n=24 (16M coefficients)

| fold | rate | CPU (ms) | GPU (ms) | Fused (ms) | GPU | Fused |
|------|------|----------|----------|------------|-----|-------|
| 4 | 1 | 992 | fail | 970 | - | 1.02x |
| 6 | 1 | 663 | 515 | 526 | **1.29x** | **1.26x** |

> Limited by GPU domain cap (2^25). Higher rates/folds push domain beyond safe limit.

#### Summary

| n | Best speedup | Config |
|---|-------------|--------|
| 18 | < 1x | GPU overhead dominates |
| 20 | **1.39x** | fold=6, rate=3 |
| 22 | **1.88x** | fold=6, rate=2 |
| 24 | **1.29x** | fold=6, rate=1 |

#### Running benchmarks

```bash
# Quick benchmark (focused configs, 3 runs each, ~10 min)
./bench.sh

# Full sweep (all configs, 3 runs each, ~60 min)
./bench.sh --full

# Fewer/more runs per config
./bench.sh --runs 5
./bench.sh --runs 1 22 1 1  # single config, 1 run (fast)

# Or run the sweep binary directly:
cargo run --release --features gpu-metal,cli --bin sweep
```

Each config is run N times (default 3) and the **median** is reported, which
reduces variance from PoW nonce search and system noise.

#### Benchmarking on another Mac

Any Mac with Apple Silicon (M1/M2/M3/M4) works:

```bash
# 1. Install Rust nightly
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y
source "$HOME/.cargo/env"

# 2. Clone and run
git clone https://github.com/miha-stopar/whir-p3-metal.git
cd whir-p3-metal
./bench.sh
```

The script prints system info (chip model, macOS version) and saves results to a
timestamped file. Share the file to compare GPU/CPU ratios across machines.

**Prerequisites**: Rust nightly toolchain + Xcode Command Line Tools
(`xcode-select --install`).

#### Benchmarking on iPhone (iOS)

A SwiftUI benchmark app is included in `ios/`. It calls the same prover code
via C FFI, so CPU vs GPU comparisons are apples-to-apples.

```bash
cd ios

# 1. Build Rust static library for iOS device + simulator
./build-rust.sh

# 2. Generate Xcode project (requires: brew install xcodegen)
xcodegen

# 3. Open in Xcode, select your iPhone, and press Cmd+R
open WhirBench.xcodeproj
```

See [`ios/README.md`](ios/README.md) for prerequisites and manual setup
without xcodegen.

For a detailed optimization log, see
[`docs/gpu-optimizations.md`](docs/gpu-optimizations.md).
