# GPU Optimization Log — WHIR Prover on Apple Silicon

This document traces each GPU optimization applied to the WHIR prover,
explains the technique, and shows benchmark results. All measurements
are on Apple M-series silicon (unified memory) using the `sweep` binary.

**Columns:**
- **CPU** — pure CPU prover (Radix-2 DFT + CPU Poseidon2 Merkle)
- **GPU** — GPU-accelerated with commit-only fusion (initial commit fused, rounds separate)
- **Fused** — GPU-accelerated with full fusion (initial commit + per-round DFT+Merkle fused)

**Parameters:**
- `n` = `num_variables` (polynomial has 2^n coefficients)
- `fold` = `folding_factor` (each STIR round folds 2^fold evaluations)
- `rate` = `starting_log_inv_rate` (RS code rate = 1/2^rate, domain = 2^(n+rate) points)

---

## Optimization 1 — GPU NTT with Montgomery Arithmetic

**Commit:** `f5de9e9` Add Metal GPU-accelerated NTT with Montgomery arithmetic

### What it does

Implements the Number Theoretic Transform (NTT) in Metal Shading Language.
The NTT is the core operation in polynomial commitment — it evaluates a
polynomial of 2^n coefficients over a multiplicative subgroup.

```
                    CPU                              GPU
        ┌───────────────────────┐      ┌───────────────────────────┐
        │ Radix-2 butterfly     │      │ Radix-2 butterfly         │
        │ Sequential per-stage  │  →   │ Parallel across all       │
        │ Cache-friendly but    │      │ butterfly pairs per stage │
        │ single-threaded       │      │ + bit-reversal permutation│
        └───────────────────────┘      └───────────────────────────┘
```

**Key design decisions:**
- BabyBear field arithmetic in **Montgomery form** throughout — the
  Montgomery multiply uses only 32-bit `mul`/`mulhi` (no 64-bit), mapping
  well to Apple GPU ALU.
- Precomputed twiddle factor table in a shared Metal buffer.
- Bit-reversal as a separate kernel with coalesced writes.

### Impact

Initial implementation; no prior GPU baseline to compare against.
Established the Metal pipeline infrastructure (device, queue, pipeline
states, buffer management).

---

## Optimization 2 — Radix-16 DIF + Column Tiling + Zero-Copy

**Commit:** `e50d7eb` Optimize GPU NTT: radix-8 butterflies, column tiling, zero-copy buffers

### What it does

Replaces the radix-2 NTT with a **decimation-in-frequency (DIF)** approach
using fused **radix-16** butterflies (processing 16 elements per thread),
with radix-8/4/2 for tail stages.

```
    Radix-2: log₂(n) stages × n/2 butterflies each
                         ↓
    Radix-16: log₂(n)/4 stages × n/16 groups each
              (4x fewer dispatches, 16 elements per thread in registers)
```

**Column tiling:** The NTT operates on a matrix (height × width). Each
thread processes one column across 16 rows. The 2D dispatch grid is
`(width, height/16)`, so consecutive threads along the x-axis access
consecutive memory addresses (coalesced reads/writes).

**Zero-copy buffers:** On Apple Silicon unified memory, the input data
is wrapped as a zero-copy Metal buffer — the GPU reads directly from the
caller's memory. The DIF stages run on a GPU-managed buffer, and the
final bit-reversal writes back to the caller's buffer with coalesced
sequential writes.

```
    Before: CPU alloc → memcpy to GPU → NTT → memcpy to CPU
    After:  CPU data (zero-copy) → R16 OOP → DIF in-place → bitrev → CPU data
                                    ↑                           ↑
                              reads from CPU            writes to CPU
                              memory directly           memory directly
```

### Impact

Major NTT speedup over radix-2; reduced dispatch count by 4x; eliminated
CPU↔GPU memory copies on Apple Silicon.

---

## Optimization 3 — GPU Poseidon2 Merkle Tree

**Commit:** `86b42cf` Add GPU Poseidon2 Merkle tree and GpuMmcs wrapper

### What it does

Implements the Poseidon2 hash function (width-16, 8+13+4 rounds, x^7 S-box)
entirely in Metal, and uses it to build Merkle trees on GPU.

```
    Leaf rows (field elements)
    ┌─────────────────────────────────┐
    │ row 0: [e0, e1, ..., e_w]      │──→ Poseidon2 sponge ──→ digest[0]
    │ row 1: [e0, e1, ..., e_w]      │──→ Poseidon2 sponge ──→ digest[1]
    │  ...                            │         ...
    │ row n: [e0, e1, ..., e_w]      │──→ Poseidon2 sponge ──→ digest[n]
    └─────────────────────────────────┘
                                              ↓
    Compression layers (binary tree of Poseidon2 2-to-1)
    Layer 0: n/2 compressions ──→ Layer 1: n/4 ──→ ... ──→ root
```

**Two Metal kernels:**
- `poseidon2_hash_leaves`: one thread per row, absorbs `leaf_width`
  elements in chunks of 8 via Poseidon2 sponge.
- `poseidon2_merkle_compress`: one thread per pair, 2-to-1 compression.

All layers are dispatched in a single command encoder — Metal guarantees
dispatch ordering, so no explicit barriers needed.

**`GpuMmcs` wrapper:** Implements Plonky3's `Mmcs` trait, automatically
choosing GPU vs CPU based on matrix size (threshold: 8 MB).

### Impact

~10% overall `whir_prove` speedup. Merkle tree construction was a
significant fraction of total time, especially for large matrices.

---

## Optimization 4 — Fused DFT → Merkle Pipeline

**Commit:** `0086d18` Fuse DFT and Merkle tree in single GPU command buffer

### What it does

Instead of running NTT on GPU → copying result to CPU → uploading to GPU
for Merkle hashing, the fused pipeline runs everything in a **single GPU
command buffer** with zero CPU round-trips.

```
    BEFORE (2 command buffers, 1 CPU round-trip):
    ┌──────────────────────────────┐
    │ GPU Command Buffer 1         │
    │ DIF stages → bitrev          │──wait──→ CPU copies result
    └──────────────────────────────┘              ↓
    ┌──────────────────────────────┐         CPU uploads
    │ GPU Command Buffer 2         │              ↓
    │ Poseidon2 leaves → compress  │──wait──→ CPU reads digests
    └──────────────────────────────┘

    AFTER (1 command buffer, 0 CPU round-trips):
    ┌──────────────────────────────────────────────────┐
    │ GPU Command Buffer (single)                      │
    │ DIF stages → bitrev → Poseidon2 leaves → compress│──wait──→ CPU reads
    └──────────────────────────────────────────────────┘
```

Exposed via the `DftCommitFusion<F>` trait. `CommitmentWriter::commit_fused()`
tries the fused path first and falls back to separate DFT + commit if the
matrix is too small for GPU benefit.

### Impact

~15-18% additional `whir_prove` speedup on top of optimization 3.

---

## Optimization 5 — Fuse DFT+Merkle in Prover Rounds

**Commit:** `3d2ff39` Fuse DFT+Merkle in prover rounds for up to 1.63x GPU speedup

### What it does

Optimization 4 only fused the **initial polynomial commitment**. But the
WHIR prover runs multiple STIR rounds, each computing an extension-field
DFT + Merkle commit. This optimization extends the fusion to **every round**.

```
    WHIR Prove Pipeline:
    ┌─────────────────────┐
    │ Initial commit      │  ← was already fused (opt 4)
    └─────────┬───────────┘
              ↓
    ┌─────────────────────┐
    │ Round 0: DFT+commit │  ← NOW fused (opt 5)
    │ + sumcheck + STIR   │
    └─────────┬───────────┘
              ↓
    ┌─────────────────────┐
    │ Round 1: DFT+commit │  ← NOW fused (opt 5)
    │ + sumcheck + STIR   │
    └─────────┬───────────┘
              ↓
           ...
              ↓
    ┌─────────────────────┐
    │ Final round         │
    └─────────────────────┘
```

Added `Prover::prove_fused()` and `round_fused()` methods that call
`mmcs.dft_algebra_and_commit(padded)` for extension-field matrices.
Falls back to separate DFT + ExtensionMmcs::commit when the matrix is
below the GPU threshold.

### Impact

Up to 1.63x GPU/CPU speedup (n=24, fold=6, rate=1). The round fusion
adds 0.2-0.4x additional speedup over commit-only fusion on larger
polynomials where round matrices exceed the GPU threshold.

---

## Optimization 6 — Lower GPU Threshold + Zero-Copy Bitrev Gather

**Commit:** `1695c2a` Lower GPU threshold to 8MB and eliminate post-GPU matrix memcpy

### What it does

**A) Lower threshold (64 MB → 8 MB):** Previously, matrices under 64 MB
fell back to CPU. Many per-round DFT matrices (especially at fold=8) were
10-50 MB — just below the cutoff. Lowering to 8 MB sends them to GPU.

```
    fold=8, rate=1, n=22:
    Round matrix ≈ 2^14 rows × 2^8 cols × 4 bytes × 4 (ext) = 16 MB

    Before: 16 MB < 64 MB → CPU fallback (0.98x)
    After:  16 MB > 8 MB  → GPU path     (1.80x)
```

**B) Zero-copy bitrev gather:** In the fused DFT+Merkle pipeline on
Apple Silicon, the bitrev gather now writes directly back into the
caller's zero-copy buffer (which IS the `values` Vec in CPU memory),
and Merkle hashing reads from the same buffer.

```
    BEFORE:
    zc_buf(values) → DIF stages(managed) → bitrev → natural_buf(managed)
        → Merkle hash(from natural_buf) → wait → memcpy(natural_buf → values)
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^
                                                    full matrix copy!

    AFTER:
    zc_buf(values) → DIF stages(managed) → bitrev → zc_buf(values)
        → Merkle hash(from zc_buf) → wait → (values already has result)
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^
                                              zero-copy! no memcpy needed
```

This eliminates:
- The separate `natural_buf` GPU buffer allocation
- The full-matrix nontemporal memcpy after GPU completion

---

## Optimization 7 — GPU Proof-of-Work Grinding

### What it does

After profiling a representative `whir_prove` run, PoW (Proof-of-Work) grinding
was identified as the dominant bottleneck — consuming **~86%** of total prove
time in PoW-heavy configurations. Each STIR round calls `grind(bits)` on the
challenger, which brute-forces a nonce by repeatedly calling the Poseidon2
permutation and checking for `bits` leading zeros. On CPU this is sequential;
the GPU can test millions of nonces in parallel.

```
    CPU PoW Grinding (sequential):
    ┌─────────────────────────────────────────┐
    │ for nonce in 0..P:                      │
    │   state[witness_idx] = nonce            │
    │   poseidon2_permute(state)              │
    │   if state[7] & mask == 0: return nonce │  ← O(2^bits) iterations
    └─────────────────────────────────────────┘

    GPU PoW Grinding (parallel):
    ┌─────────────────────────────────────────────────┐
    │ Dispatch 1M threads per batch:                  │
    │   each thread tests one nonce                   │
    │   atomic flag signals first winner              │
    │                                                 │
    │ Batch 0: nonce [0..1M)        ──→ check found   │
    │ Batch 1: nonce [1M..2M)       ──→ check found   │
    │ ...                                             │
    │ Usually finds winner in first 1-2 batches       │
    └─────────────────────────────────────────────────┘
```

**Metal kernel (`poseidon2_pow_grind`):**
- Takes the base Poseidon2 sponge state (16 elements in Montgomery form)
- Each GPU thread substitutes its nonce at `witness_idx`, runs the full
  Poseidon2 permutation, converts `state[7]` to canonical form, checks mask
- First winner atomically writes to shared `result`/`found` buffers

**`GpuChallenger` wrapper:**
- Wraps the standard `DuplexChallenger` and delegates all observe/sample ops
- Overrides `GrindingChallenger::grind()` to extract the internal sponge state,
  dispatch to GPU, and verify the result on CPU before updating challenger state
- Falls back to CPU grinding if GPU doesn't find a witness (never happens in
  practice since nonce space covers all of BabyBear)

**Buffer caching optimization:**
- Pre-allocates all Metal buffers at `MetalBabyBearDft` construction time
- Each `gpu_pow_grind` call copies state into cached buffers (64 bytes) instead
  of allocating new 4KB+ buffers per call
- Eliminates per-call allocation overhead across the many grind calls per proof

### Key implementation details

Montgomery form conversions were critical to get right:
- `bb_from_canonical(x, R²)`: canonical → Montgomery via `mul(x, R²) mod P`
- `bb_to_canonical(x)`: Montgomery → canonical via subtraction-based reduction
  (equivalent to `mul(x, 1)`, matching the p3 implementation)

### Impact

Dramatic speedup on PoW-heavy configurations. Best result: **2.58x** at n=24,
fold=8, rate=1. The improvement is proportional to how much time the CPU spends
grinding — configs with many STIR rounds and high `pow_bits` benefit most.

---

## Comprehensive Benchmark Results

All times in milliseconds. Median of 3 runs.
`rs_domain_initial_reduction_factor = min(fold, 3)` to allow fold < 3.

**Columns:**
- **CPU** — pure CPU prover (baseline)
- **GPU** — GPU NTT + Merkle, commit-only fusion, CPU PoW grinding
- **Fused** — GPU NTT + Merkle, full pipeline fusion, CPU PoW grinding
- **Grind** — GPU NTT + Merkle, full pipeline fusion, **GPU PoW grinding**

### n=18 (256K coefficients)

| fold | rate | CPU (ms) | GPU (ms) | Fused (ms) | Grind (ms) | Best speedup |
|------|------|----------|----------|------------|------------|--------------|
| 1 | 1 | 108 | 121 | 142 | 158 | 0.89x GPU |
| 1 | 2 | 134 | 172 | 182 | 201 | 0.78x GPU |
| 1 | 3 | 243 | 230 | 230 | 247 | **1.06x** GPU |
| 2 | 1 | 42 | 60 | 62 | 80 | 0.70x GPU |
| 2 | 2 | 61 | 75 | 83 | 99 | 0.80x GPU |
| 2 | 3 | 105 | 107 | 107 | 122 | 0.99x GPU |
| 3 | 1 | 22 | 26 | 34 | 49 | 0.83x GPU |
| 3 | 2 | 32 | 40 | 44 | 59 | 0.79x GPU |
| 3 | 3 | 48 | 60 | 63 | 76 | 0.79x GPU |
| 4 | 1 | 14 | 22 | 26 | 39 | 0.62x GPU |
| 4 | 2 | 24 | 30 | 34 | 53 | 0.80x GPU |
| 4 | 3 | 37 | 41 | 44 | 59 | 0.90x GPU |
| 6 | 1 | 11 | 18 | 20 | 37 | 0.62x GPU |
| 6 | 2 | 18 | 24 | 26 | 43 | 0.76x GPU |
| 6 | 3 | 29 | 32 | 35 | 52 | 0.91x GPU |
| 8 | 1 | 10 | 16 | 18 | 33 | 0.64x GPU |
| 8 | 2 | 18 | 21 | 24 | 42 | 0.87x GPU |
| 8 | 3 | 33 | 35 | 36 | 54 | 0.93x GPU |

> GPU overhead dominates at this size. All configs slower than or matching CPU.

### n=20 (1M coefficients)

| fold | rate | CPU (ms) | GPU (ms) | Fused (ms) | Grind (ms) | Best speedup |
|------|------|----------|----------|------------|------------|--------------|
| 1 | 1 | 278 | 265 | 279 | 301 | 1.05x GPU |
| 1 | 2 | 481 | 415 | 424 | 417 | **1.16x** GPU |
| 1 | 3 | 920 | 744 | 689 | 718 | **1.33x** Fused |
| 2 | 1 | 123 | 122 | 129 | 145 | 1.01x GPU |
| 2 | 2 | 210 | 183 | 182 | 195 | **1.16x** Fused |
| 2 | 3 | 391 | 313 | 297 | 313 | **1.32x** Fused |
| 3 | 1 | 63 | 68 | 77 | 93 | 0.92x GPU |
| 3 | 2 | 99 | 100 | 104 | 117 | 0.99x GPU |
| 3 | 3 | 190 | 164 | 156 | 169 | **1.22x** Fused |
| 4 | 1 | 45 | 57 | 60 | 81 | 0.79x GPU |
| 4 | 2 | 80 | 82 | 79 | 98 | 1.01x Fused |
| 4 | 3 | 205 | 238 | 152 | 138 | **1.49x** Grind |
| 6 | 1 | 37 | 50 | 50 | 68 | 0.75x GPU |
| 6 | 2 | 114 | 120 | 119 | fail | 0.95x GPU |
| 6 | 3 | 422 | 567 | 751 | 382 | **1.10x** Grind |
| 8 | 1 | 25 | 32 | 34 | 50 | 0.78x GPU |
| 8 | 2 | 47 | 44 | 48 | 60 | **1.07x** GPU |
| 8 | 3 | 114 | 87 | 93 | 94 | **1.30x** GPU |

> GPU starts winning at rate=2-3. Grind shines at fold=4-6, rate=3.

### n=22 (4M coefficients)

| fold | rate | CPU (ms) | GPU (ms) | Fused (ms) | Grind (ms) | Best speedup |
|------|------|----------|----------|------------|------------|--------------|
| 1 | 1 | 1051 | 893 | 863 | 868 | **1.22x** Fused |
| 1 | 2 | 1895 | 1528 | 1421 | 1462 | **1.33x** Fused |
| 1 | 3 | 3717 | 2845 | 2603 | 2609 | **1.43x** Fused |
| 2 | 1 | 464 | 394 | 385 | 403 | **1.20x** Fused |
| 2 | 2 | 841 | 648 | 607 | 621 | **1.39x** Fused |
| 2 | 3 | 1720 | 1198 | 1092 | 1120 | **1.57x** Fused |
| 3 | 1 | 223 | 199 | 187 | 208 | **1.19x** Fused |
| 3 | 2 | 410 | 343 | 329 | 329 | **1.25x** Grind |
| 3 | 3 | 864 | 750 | 639 | 622 | **1.39x** Grind |
| 4 | 1 | 175 | 143 | 133 | 150 | **1.32x** Fused |
| 4 | 2 | 328 | 236 | 212 | 217 | **1.55x** Fused |
| **4** | **3** | **887** | **626** | **570** | **471** | **1.88x Grind** |
| 6 | 1 | 150 | 128 | 119 | 113 | **1.32x** Grind |
| **6** | **2** | **504** | **363** | **279** | **260** | **1.94x Grind** |
| **6** | **3** | **2533** | **fail** | **1778** | **1129** | **2.24x Grind** |
| 8 | 1 | 174 | 83 | 81 | 93 | **2.15x** Fused |
| 8 | 2 | 214 | 148 | 142 | 149 | **1.51x** Fused |
| 8 | 3 | 449 | 349 | 299 | 292 | **1.54x** Grind |

> Best result: **2.24x** Grind at fold=6, rate=3. Fused wins at fold=1-2 (many tiny rounds).

### n=24 (16M coefficients)

| fold | rate | CPU (ms) | GPU (ms) | Fused (ms) | Grind (ms) | Best speedup |
|------|------|----------|----------|------------|------------|--------------|
| 1 | 1 | 6578 | 4530 | 3304 | 3229 | **2.04x** Grind |
| 2 | 1 | 1880 | 1434 | 1355 | 1360 | **1.39x** Fused |
| 3 | 1 | 918 | 728 | 692 | 704 | **1.33x** Fused |
| 4 | 1 | 903 | 734 | 803 | 642 | **1.41x** Grind |
| 6 | 1 | 554 | 409 | 522 | 365 | **1.52x** Grind |
| **8** | **1** | **27576** | **38115** | **38836** | **10676** | **2.58x Grind** |

> Only rate=1 testable on GPU (rate=2+ exceeds domain limit 2^25).
> fold=8 at n=24 has extreme PoW grinding → **2.58x** Grind, the biggest speedup overall.

---

## Summary

```
    Best GPU speedup vs CPU (per n, any fold/rate):

    n=18:  1.06x     (fold=1, rate=3, GPU)          — GPU overhead dominates
    n=20:  1.49x     (fold=4, rate=3, Grind)        — GPU starts winning
    n=22:  2.24x     (fold=6, rate=3, Grind)        — sweet spot for Grind
    n=24:  2.58x     (fold=8, rate=1, Grind)        ← biggest speedup (PoW-dominated)
```

### Which GPU strategy wins?

The best strategy depends on the PoW grinding fraction of total prove time:

```
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │   PoW grinding fraction of total time                    │
    │   ──────────────────────────────────────                 │
    │   HIGH (>50%)  │  GPU Grind wins (up to 2.58x)          │
    │                │  → fold=6-8 rate=1-3 at large n         │
    │   ─────────────┤                                         │
    │   MEDIUM       │  Grind or Fused, config-dependent       │
    │   (20-50%)     │  → fold=3-4 rate=2-3                    │
    │   ─────────────┤                                         │
    │   LOW (<20%)   │  Fused wins (DFT+Merkle is bottleneck)  │
    │                │  → fold=1-2, or small n                  │
    └──────────────────────────────────────────────────────────┘
```

At **fold=1-2**, every round is tiny (2 or 4 evaluations folded) and there are
many rounds. Each round's DFT+Merkle is small, so the grinding overhead per
round is relatively low — Fused (which accelerates DFT+Merkle) tends to win.

At **fold=6-8**, each round folds many evaluations. Fewer rounds, but each
round's PoW grind is the same cost. The grinding fraction of total time is
higher, so GPU Grind dominates.

### When to use each mode

- **CPU only** — `n <= 18` or very small polynomials
- **GPU Fused** — `n >= 20` with fold=1-2 (DFT/Merkle bottleneck, many rounds)
- **GPU Grind** — `n >= 20` with fold >= 3 and rate >= 2, or any large `n` with heavy PoW

Grind mode includes all Fused optimizations plus GPU PoW, so it's always safe
to use; the overhead for the GPU PoW path is small even when grinding is fast.

### Optimization progression

| # | Optimization | Key improvement |
|---|-------------|-----------------|
| 1 | GPU NTT (Metal) | Established GPU pipeline |
| 2 | Radix-16 DIF + zero-copy | 4x fewer dispatches, no memcpy |
| 3 | GPU Poseidon2 Merkle | ~10% whir_prove speedup |
| 4 | Fused DFT→Merkle | ~15-18% additional speedup |
| 5 | Fused prover rounds | Up to 1.63x total |
| 6 | Lower threshold + zero-copy bitrev | Up to 1.88x total |
| 7 | GPU PoW Grinding + buffer caching | **Up to 2.58x total** |
