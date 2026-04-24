# GPU-Accelerated WHIR Proving on Apple Silicon: Benchmarks and Lessons from Client-Side Metal Compute

## TL;DR

We accelerated the [WHIR](https://eprint.iacr.org/2024/1586) prover on Apple Silicon GPUs using Metal compute shaders, achieving **up to 2.03x speedup** over highly optimized CPU code (SIMD + LTO + `target-cpu=native`) on an M1 chip. The GPU pipeline fuses NTT (Number Theoretic Transform), bit-reversal, Poseidon2 Merkle tree hashing, and proof-of-work grinding into single command buffer submissions, exploiting Apple Silicon's unified memory to eliminate CPU-GPU data transfers. The implementation is open source and runs on any Mac with Apple Silicon, with an iOS benchmark app for iPhone testing.

**Key findings:**
- GPU wins for all tested configurations at polynomial sizes >= 2^20 (1M+ coefficients)
- Fused DFT+Merkle pipeline avoids CPU round-trips and provides the biggest gains
- Apple Silicon's unified memory model is a major advantage -- zero-copy buffer sharing eliminates the #1 bottleneck reported by other GPU proving projects
- The Poseidon2 Merkle kernel is the dominant cost (~58% of GPU time), already at near-peak throughput for Montgomery multiplication on Apple GPU
- Compiler optimizations (LTO, `target-cpu=native`) improved the CPU baseline by ~25%, making the GPU harder to beat but improving absolute end-to-end performance

**Repository**: [github.com/miha-stopar/whir-p3-metal](https://github.com/miha-stopar/whir-p3-metal)

---

## 1. Motivation: Why Client-Side Proving Matters

Ethereum's transparency comes at a privacy cost: every transaction is permanently visible, and chain analysis tools can link pseudonymous addresses to real identities. Zero-knowledge proofs can restore privacy, but delegating proof generation to a server defeats the purpose -- the server sees your private inputs.

True privacy requires **client-side proving**: users generate proofs on their own devices. This matters for:
- **Private payments** -- hiding amounts, counterparties, and transaction patterns
- **Identity** -- proving facts about credentials (age, citizenship, membership) without revealing the credential itself
- **Voting** -- anonymous participation in DAOs and governance

The barrier is performance. Proving on consumer hardware must be fast enough for interactive use. Server-side GPU provers (CUDA on datacenter GPUs) achieve dramatic speedups, but client devices have different constraints: thermal limits, shared memory bandwidth, smaller GPU core counts, and no discrete VRAM.

### The Client-Side GPU Opportunity

Modern phones and laptops contain increasingly capable GPUs. Apple Silicon's M-series and A-series chips share unified memory between CPU and GPU, eliminating the PCIe transfer bottleneck that dominates datacenter GPU proving. This architectural difference creates a unique opportunity: on Apple Silicon, the CPU and GPU can read/write the same memory without copies, making fine-grained CPU-GPU cooperation practical.

Several projects are exploring this space:
- **[Mopro](https://zkmopro.org/)** -- Metal MSM acceleration (v2: 40-100x over v1), WebGPU field ops benchmarks showing 100x+ throughput on small fields vs BN254
- **[ICICLE Metal](https://dev.ingonyama.com/)** -- MSM and NTT primitives for Apple Metal, up to 5x acceleration (v3.6)
- **[zkSecurity / Stwo WebGPU](https://blog.zksecurity.xyz/posts/webgpu/)** -- 2x overall proving speedup for Circle STARKs in the browser via WebGPU
- **[Ligetron](https://github.com/ligeroinc/ligero-prover)** -- WebGPU SHA-256 and NTT for cross-platform proving
- **[FibRace](https://arxiv.org/abs/2510.14693)** -- large-scale mobile benchmark (6,000+ participants, 2.1M proofs) showing most modern smartphones prove in <5 seconds

Our work focuses on a specific, practically relevant target: accelerating the **WHIR** polynomial commitment scheme, which is hash-based and post-quantum secure, using Apple's Metal API for native GPU compute.

---

## 2. WHIR: Background

[WHIR](https://eprint.iacr.org/2024/1586) (published at EUROCRYPT 2025) is an Interactive Oracle Proof of proximity for Reed-Solomon codes. It serves as an efficient replacement for FRI, STIR, and BaseFold, with notably fast verification (hundreds of microseconds vs. milliseconds for alternatives).

For proving, the dominant costs are:
1. **NTT (Number Theoretic Transform)** -- polynomial evaluation over extension domains
2. **Merkle tree construction** -- Poseidon2 hashing for polynomial commitments
3. **Proof-of-work grinding** -- finding nonces satisfying hash difficulty targets (Fiat-Shamir security)

All three are massively parallel and map well to GPU compute. The prover executes multiple STIR rounds, each involving an NTT, a Merkle commitment, and a PoW grind. Between rounds, the CPU performs sumcheck operations (sequential, not GPU-friendly).

Our implementation builds on the [Plonky3](https://github.com/Plonky3/Plonky3) library's WHIR implementation (`p3-whir`) over the BabyBear field (31-bit prime, Montgomery form).

---

## 3. GPU Architecture and Implementation

### 3.1 Why Metal (not WebGPU or CUDA)

- **Unified memory**: Apple Silicon shares physical memory between CPU and GPU. Metal buffers created with `MTLResourceStorageModeShared` are directly accessible by both without copies. This eliminates the #1 bottleneck reported by WebGPU and CUDA client-side proving projects.
- **Low dispatch overhead**: Metal command buffers can be built on CPU while previous work executes on GPU, enabling tight pipelining.
- **Native performance**: Metal Shading Language (MSL) compiles to AIR (Apple Intermediate Representation) at build time and to device-specific machine code at load time. No runtime shader compilation overhead.
- **iOS compatibility**: The same Metal code runs on iPhone and iPad, enabling mobile benchmarking.

The tradeoff is platform lock-in to Apple. For cross-platform deployment, WebGPU (via wgpu) would be the alternative, but at a performance cost.

### 3.2 Pipeline Architecture

The GPU pipeline fuses multiple stages into a single Metal command buffer:

```
CPU input buffer (zero-copy shared memory)
    │
    ├── Radix-16/32 DIF-NTT stages (in-place on GPU-managed buffer)
    │       Uses R32, R16, R8, R4, R2 butterfly kernels
    │       Shared-memory kernels for final stages (up to 4096 elements/threadgroup)
    │
    ├── Bit-reversal permutation (fused with final NTT stage)
    │       Writes directly back to zero-copy buffer
    │
    ├── Poseidon2 leaf hashing (width-16, 8+13+4 rounds, x^7 S-box)
    │       4-leaf fused kernel: hashes 4 leaves + first Merkle compress in one dispatch
    │
    ├── Poseidon2 Merkle compression (all remaining tree levels)
    │       SIMD shuffle kernel for small levels (avoids threadgroup memory)
    │
    └── [Single GPU wait] → Results in CPU-accessible memory
```

This fusion eliminates 3-4 CPU-GPU synchronization points per STIR round compared to a naive implementation.

### 3.3 Key Optimizations (30 iterations)

We went through 30 optimization iterations. The most impactful ones:

| # | Optimization | Impact |
|---|-------------|--------|
| 1-4 | Basic Metal NTT + Merkle kernels | Baseline GPU path |
| 5-8 | Radix-16 DIF, shared-memory butterflies | 2-3x NTT speedup |
| 9-12 | Fused DFT+Merkle pipeline, zero-copy I/O | 1.5-2x end-to-end |
| 13-16 | Poseidon2 4-leaf fused kernel, SIMD Merkle | 1.3x Merkle speedup |
| 17-20 | GPU PoW grinding, zero-copy EF conversions | Helps PoW-dominated configs |
| 21-24 | Extension field zero-copy, per-round fusion | 1.2x for large n |
| 25-28 | LTO, `target-cpu=native`, profiling-guided | 10% absolute improvement |
| 29-30 | R32 DIF kernel, packed transpose bypass | 5-15ms/round savings |

The most important lesson: **fusing operations to avoid CPU-GPU round-trips matters more than optimizing individual kernels**. The fused pipeline (single command buffer for DFT+Merkle) typically beats the non-fused path by 15-30%.

### 3.4 Montgomery Arithmetic in Metal

BabyBear field operations use Montgomery form throughout the GPU kernels. The Montgomery multiply uses only 32-bit `mul`/`mulhi` instructions -- no 64-bit arithmetic needed. This maps efficiently to Apple GPU's ALU, which has native 32-bit integer multiply with high part extraction (`mulhi` in MSL).

```metal
inline uint bb_mont_mul(uint a, uint b) {
    uint lo = a * b;
    uint q  = lo * BB_MONT_NINV;    // q = lo * N^{-1} mod 2^32
    uint hi = mulhi(a, b);
    uint qn_hi = mulhi(q, BB_P);
    uint t = hi - qn_hi;
    return (t >= BB_P) ? t - BB_P : t;  // conditional subtract
}
```

This achieves ~112 GOP/s on Apple M-series for BabyBear field multiplication, compared to <1 GOP/s for BN254 operations (from Mopro benchmarks).

---

## 4. Benchmark Results

### Setup

- **Hardware**: Apple M1 (8 GPU cores, 16GB unified memory, 68.25 GB/s bandwidth)
- **Software**: Rust nightly 1.97.0, macOS 15.5, release build with LTO (thin) + `codegen-units=1` + `target-cpu=native`
- **Methodology**: 3 runs per configuration, **median** reported
- **Baseline**: Highly optimized CPU path using Plonky3's Radix2DFT with NEON SIMD, rayon parallelism, and the same LTO/native settings

### Parameters

- `n` = number of variables (polynomial has 2^n coefficients)
- `fold` = folding factor per STIR round
- `rate` = starting log inverse rate (RS code rate = 1/2^rate)

### Results

29 configurations tested. "Best GPU" = minimum of GPU, FUSED, and GRIND modes.

#### n=20 (1M coefficients)

| fold | rate | CPU (ms) | Best GPU (ms) | Speedup |
|------|------|----------|---------------|---------|
| 1    | 1    | 267      | 171           | **1.56x** |
| 1    | 2    | 453      | 290           | **1.56x** |
| 1    | 3    | 897      | 483           | **1.86x** |
| 2    | 1    | 127      | 75            | **1.70x** |
| 2    | 2    | 217      | 122           | **1.78x** |
| 2    | 3    | 414      | 210           | **1.98x** |
| 4    | 1    | 49       | 35            | **1.41x** |
| 4    | 2    | 89       | 53            | **1.70x** |
| 4    | 3    | 200      | 120           | **1.67x** |

#### n=22 (4M coefficients)

| fold | rate | CPU (ms) | Best GPU (ms) | Speedup |
|------|------|----------|---------------|---------|
| 1    | 1    | 1174     | 579           | **2.03x** |
| 1    | 2    | 1938     | 1092          | **1.77x** |
| 1    | 3    | 3459     | 1934          | **1.79x** |
| 2    | 1    | 441      | 287           | **1.54x** |
| 2    | 2    | 805      | 484           | **1.66x** |
| 2    | 3    | 1676     | 897           | **1.87x** |
| 3    | 1    | 206      | 138           | **1.49x** |
| 3    | 2    | 381      | 248           | **1.54x** |
| 3    | 3    | 897      | 611           | **1.47x** |
| 4    | 1    | 166      | 128           | **1.30x** |
| 4    | 2    | 322      | 215           | **1.50x** |
| 4    | 3    | 661      | 530           | **1.25x** |
| 6    | 1    | 141      | 98            | **1.44x** |
| 6    | 2    | 410      | 327           | **1.25x** |
| 6    | 3    | 1763     | 1320          | **1.34x** |

#### n=24 (16M coefficients)

| fold | rate | CPU (ms) | Best GPU (ms) | Speedup |
|------|------|----------|---------------|---------|
| 1    | 1    | 4153     | 2463          | **1.69x** |
| 2    | 1    | 1814     | 1049          | **1.73x** |
| 3    | 1    | 890      | 531           | **1.68x** |
| 4    | 1    | 978      | 694           | **1.41x** |
| 6    | 1    | 588      | 405           | **1.45x** |

> n=24 rate>=2 exceeds the GPU domain cap (2^25 elements) and is not tested.

### Key Observations

**GPU is faster than CPU for all 29 tested configurations at n >= 20.** The speedup ranges from 1.25x to 2.03x, with the best results at low fold values and higher rates.

**Low fold values give the best speedups** (1.6-1.96x at fold=1-2) because the NTT and Merkle tree dominate the runtime -- exactly the operations we accelerated. Higher fold values (fold=4-8) reduce the number of NTT elements per round but increase the number of rounds and PoW grinding, shifting work toward CPU-bound operations (sumcheck) where the GPU can't help.

**Rate increases the workload and generally increases speedup** because the domain expansion (2^rate more points) creates more NTT and Merkle work, amplifying the GPU advantage.

**The CPU baseline is very strong.** Plonky3's BabyBear implementation uses NEON SIMD intrinsics with 4-wide packed operations. Combined with LTO and `target-cpu=native`, the CPU path improved ~25% during our optimization work (from Rust compiler updates and build settings). This makes the GPU speedup harder to achieve but more meaningful -- we're beating a genuinely optimized baseline.

---

## 5. Where the Time Goes

Profiling breakdown for a representative configuration (n=24, fold=3, rate=1, total GPU time ~537ms):

| Component | Time | % | Notes |
|-----------|------|---|-------|
| GPU Poseidon2 Merkle | ~315 ms | 58% | Compute-bound (Montgomery mul) |
| GPU DFT (NTT) | ~75 ms | 14% | Memory-bandwidth-limited |
| CPU sumcheck | ~80 ms | 15% | External crate, SIMD-optimized |
| CPU constraint combination | ~45 ms | 8% | Partially GPU-offloaded |
| GPU readback + dispatch | ~24 ms | 4% | Zero-copy already in use |

The Poseidon2 Merkle kernel dominates. Each Poseidon2 permutation (width-16) requires 25 rounds of S-box + MDS matrix operations, where each S-box is `x^7 = x * x * x * x * x * x * x` (6 Montgomery multiplications). With millions of leaves to hash and a full binary tree to compress, this is ~58% of the GPU runtime and is already near peak ALU throughput.

The NTT is memory-bandwidth-limited (not compute-limited) because butterfly operations are simple (1 multiply + 1 add per pair) but access non-sequential memory addresses at large strides. Our radix-16/32 kernels reduce the number of global memory passes.

The CPU sumcheck is the remaining sequential bottleneck -- it lives in the external `p3-whir` crate and uses SIMD-optimized polynomial arithmetic. This is fundamentally sequential between rounds and not easily GPU-accelerated.

---

## 6. Unified Memory: Apple Silicon's Advantage

The most common bottleneck reported by GPU proving projects is **CPU-GPU data transfer**. On discrete GPUs, transferring a 64MB polynomial over PCIe takes ~4ms (16 GB/s) -- comparable to the NTT computation itself. On Apple Silicon, this cost is **zero**: the CPU and GPU access the same physical DRAM.

This enabled our fused pipeline design: the CPU writes polynomial coefficients into a shared buffer, the GPU runs NTT + Merkle in a single command buffer submission, and the CPU reads back only the Merkle root (~256 bytes). No copies, no transfers, no synchronization between NTT and Merkle stages.

In practice, "zero-copy" still has caveats:
- **Cache coherence**: The GPU may need to flush CPU caches before reading, adding ~0.1ms for large buffers
- **Memory bandwidth sharing**: CPU and GPU compete for the same memory bus (68.25 GB/s on M1, ~200 GB/s on M4 Max)
- **No true concurrent access**: While the GPU is writing, the CPU should not read the same region (no hardware coherence during GPU execution)

Despite these caveats, unified memory eliminates the #1 bottleneck and makes the fused pipeline architecture practical.

---

## 7. Lessons Learned

### What worked

1. **Fusing operations eliminates synchronization overhead.** A single command buffer for DFT + bit-reversal + Merkle avoids 3-4 CPU-GPU round-trips per round. Each round-trip costs ~0.5-1ms of dispatch overhead plus cache flush costs.

2. **Radix-16/32 DIF reduces global memory passes.** A radix-2 NTT of size 2^24 needs 24 global memory passes. Radix-16 needs 6, and radix-32 needs 5. Since the NTT is bandwidth-limited, fewer passes directly translates to speed.

3. **Shared-memory kernels for small NTTs.** When the NTT fits in threadgroup memory (up to 4096 elements = 16KB on Apple GPU), we can do all butterfly stages in shared memory with a single global read and write. This handles the final stages after the global radix-16/32 passes.

4. **Montgomery form everywhere.** Converting between standard and Montgomery form is expensive. We keep all data in Montgomery form across all GPU kernels (NTT twiddle factors, Poseidon2 round constants, Merkle digests) and only convert at the final output.

### What didn't work

1. **Four-step FFT decomposition.** We tried decomposing large NTTs into smaller ones via the four-step algorithm (row NTTs + twiddle + transpose + column NTTs) to maximize shared memory usage. This was **20% slower** due to shared memory bank conflicts during transpose and the overhead of the extra twiddle multiply pass.

2. **Overlapping DFT readback with Merkle GPU work.** We tried reading NTT results back to CPU while the GPU started Merkle hashing. This was **slower** because the readback competed with the GPU for memory bandwidth, and the unified memory controller couldn't efficiently serve both.

3. **GPU sumcheck.** The sumcheck protocol's sequential round structure (each round depends on the verifier's challenge from the previous round) makes it inherently serial. Parallelizing within a round helps, but the round-to-round dependency limits GPU utilization.

### Practical challenges

- **Benchmark variance from PoW.** Proof-of-work grinding has exponentially distributed completion times (searching for a random nonce). Single-run benchmarks can vary by 50%+ for PoW-heavy configurations. Using the median of 3 runs significantly reduces this noise.

- **Compiler optimizations help the CPU more than the GPU.** LTO and `target-cpu=native` improved CPU performance by ~25% but had minimal effect on GPU kernel performance (which is Metal shader code, not Rust). This narrowed the GPU/CPU ratio despite improving absolute performance.

- **Thermal throttling on mobile.** Extended benchmark runs on laptops (and especially phones) can trigger thermal throttling, reducing GPU clock speeds. Our benchmarks use the median of 3 runs to mitigate this, but real-world sustained performance may be lower.

---

## 8. Comparison with Related Work

| Project | Target | Protocol | Speedup | Field | API |
|---------|--------|----------|---------|-------|-----|
| **This work** | Apple M1 GPU | WHIR/Poseidon2 | **1.3-2.0x** vs CPU | BabyBear (31-bit) | Metal |
| Mopro Metal MSM v2 | Apple GPU | MSM (BN254) | 40-100x vs v1 | BN254 (254-bit) | Metal |
| ICICLE Metal | Apple GPU | MSM, NTT | up to 5x | Multiple | Metal |
| ICICLE-Stwo (CUDA) | Datacenter GPU | Circle STARK | 3.25-7x vs CPU SIMD | M31 (31-bit) | CUDA |
| zkSecurity Stwo WebGPU | Browser GPU | Circle STARK | 2x overall | M31 | WebGPU |
| Ligetron | Browser/native | SHA-256, NTT | N/A (WIP) | Multiple | WebGPU |

Our speedup numbers (1.3-2.0x) are more modest than MSM-focused projects because:
1. **We benchmark end-to-end proving**, including CPU-bound sumcheck rounds that can't be GPU-accelerated
2. **Our CPU baseline is extremely strong** -- Plonky3's BabyBear implementation with NEON SIMD, LTO, and native CPU features
3. **WHIR's workload is hash-dominated** (Poseidon2 Merkle), which has less parallelism than MSM or NTT alone

For the GPU-only portions (NTT + Merkle), the speedup is 3-4x over non-SIMD CPU code.

---

## 9. Future Directions

### Higher-arity Merkle trees
The Poseidon2 Merkle kernel dominates runtime (58%). A 4-ary or 8-ary tree would reduce the number of hash invocations by 2-3x, proportionally speeding up the GPU pipeline. This requires protocol-level changes.

### Newer Apple Silicon
Our benchmarks are on M1 (2020). The M4 Max has 40 GPU cores (vs 8 on M1) and 273 GB/s memory bandwidth (vs 68 GB/s). We expect 3-4x improvement from hardware alone. The repository includes a `bench.sh` script and an iOS app to make cross-device benchmarking easy -- contributions welcome.

### WebGPU backend
For cross-platform reach, a WebGPU (wgpu) backend could reuse the same kernel algorithms. The main cost would be losing unified memory zero-copy on non-Apple platforms and shader compilation overhead. The Mopro project's WebGPU field operation benchmarks suggest ~50% of Metal performance is achievable.

### GPU sumcheck
The sumcheck protocol is the main remaining CPU bottleneck (~15% of runtime). While the round-to-round dependency is inherently serial, the *within-round* computation (summing over a large hypercube) could benefit from GPU parallelism for very large instances.

---

## 10. Reproducibility

All code is open source: **[github.com/miha-stopar/whir-p3-metal](https://github.com/miha-stopar/whir-p3-metal)**

### Running on Mac
```bash
# Install Rust nightly
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y
source "$HOME/.cargo/env"

# Clone and benchmark
git clone https://github.com/miha-stopar/whir-p3-metal.git
cd whir-p3-metal
./bench.sh           # ~15 min, saves results to bench_results_<timestamp>.txt
./bench.sh --full    # ~60 min, all configurations
```

### Running on iPhone
```bash
cd ios
./build-rust.sh      # builds Rust static library for iOS
xcodegen             # generates Xcode project (brew install xcodegen)
open WhirBench.xcodeproj  # build and run on device
```

### Detailed optimization log
See [`docs/gpu-optimizations.md`](https://github.com/miha-stopar/whir-p3-metal/blob/main/docs/gpu-optimizations.md) for a detailed log of all 30 optimization iterations with before/after benchmarks.
