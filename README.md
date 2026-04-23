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

Apple M1, macOS 15.5, Rust nightly 1.97.0, release + LTO.
3 runs per config, **median** reported. All times in milliseconds.

Parameters:
- `n` = `num_variables` (polynomial has 2^n coefficients)
- `fold` = `folding_factor` (each STIR round folds 2^fold evaluations)
- `rate` = `starting_log_inv_rate` (RS code rate = 1/2^rate)
- **GPU** = GPU DFT + Merkle, CPU challenger
- **FUSED** = fused DFT+Merkle pipeline, CPU challenger
- **GRIND** = fused pipeline + GPU PoW grinding

| n  | fold | rate | CPU (ms) | GPU (ms) | FUSED (ms) | GRIND (ms) | Speedup |
|----|------|------|----------|----------|------------|------------|---------|
| 20 | 1    | 1    | 302.9    | 219.6    | 189.6      | 185.7      | **1.63x** |
| 20 | 2    | 2    | 212.8    | 148.1    | 132.7      | 133.3      | **1.60x** |
| 20 | 4    | 3    | 180.9    | 143.5    | 138.1      | 138.2      | **1.31x** |
| 22 | 1    | 1    | 1134.9   | 766.2    | 671.6      | 675.7      | **1.69x** |
| 22 | 1    | 2    | 2035.6   | 1163.7   | 1052.4     | 1058.4     | **1.93x** |
| 22 | 2    | 1    | 440.2    | 297.6    | 268.7      | 287.0      | **1.64x** |
| 22 | 3    | 2    | 394.3    | 267.2    | 269.9      | 259.6      | **1.52x** |
| 22 | 4    | 3    | 733.4    | 556.4    | 541.6      | 614.8      | **1.35x** |
| 22 | 6    | 2    | 379.2    | 392.7    | 344.1      | 490.9      | **1.10x** |
| 24 | 1    | 1    | 4772.7   | 2965.9   | 2509.8     | 2429.9     | **1.96x** |
| 24 | 2    | 1    | 1860.6   | 1244.7   | 1084.8     | 1063.3     | **1.75x** |
| 24 | 3    | 1    | 840.2    | 611.9    | 538.2      | 536.6      | **1.57x** |
| 24 | 4    | 1    | 889.7    | 803.5    | 753.8      | 760.2      | **1.18x** |
| 24 | 6    | 1    | 582.3    | 457.1    | 405.3      | 439.4      | **1.44x** |

#### Summary

| n  | Best speedup | Config           |
|----|-------------|------------------|
| 20 | **1.63x**   | fold=1, rate=1   |
| 22 | **1.93x**   | fold=1, rate=2   |
| 24 | **1.96x**   | fold=1, rate=1   |

GPU is faster than CPU for **all tested configurations** at n >= 20.
Best speedups at low fold values where NTT dominates.

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
