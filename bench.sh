#!/usr/bin/env bash
set -euo pipefail

# Quick GPU vs CPU benchmark for whir-p3-metal.
# Usage:
#   ./bench.sh                  # default configs, 3 runs each (~10 min)
#   ./bench.sh --full           # all configs, 3 runs each (~60 min)
#   ./bench.sh --runs 5         # default configs, 5 runs each
#   ./bench.sh 22 1 1           # single config, 3 runs
#   ./bench.sh --runs 1 22 1 1  # single config, 1 run (fast)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Parse args ────────────────────────────────────────────────────────
RUNS=3
FULL=false
SINGLE_N="" SINGLE_F="" SINGLE_R=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)  FULL=true; shift ;;
        --runs)  RUNS="$2"; shift 2 ;;
        *)
            if [[ -z "$SINGLE_N" ]]; then SINGLE_N="$1"
            elif [[ -z "$SINGLE_F" ]]; then SINGLE_F="$1"
            elif [[ -z "$SINGLE_R" ]]; then SINGLE_R="$1"
            fi
            shift ;;
    esac
done

# ── Prerequisites check ──────────────────────────────────────────────
if ! command -v rustc &>/dev/null; then
    echo "ERROR: Rust toolchain not found."
    echo "Install via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

RUSTC_VERSION=$(rustc --version)
HOST_ARCH=$(uname -m)
HOST_OS=$(sw_vers -productName 2>/dev/null || uname -s)
HOST_VER=$(sw_vers -productVersion 2>/dev/null || uname -r)
HOST_CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")

echo "Rust:  $RUSTC_VERSION"
echo "Host:  $HOST_ARCH $HOST_OS $HOST_VER"
echo "Chip:  $HOST_CHIP"
echo "Runs:  $RUNS per config (median reported)"
echo ""

# ── Build ─────────────────────────────────────────────────────────────
echo "Building sweep binary (release + LTO, may take ~30s on first build)..."
cargo build --release --bin sweep --features gpu-metal,cli 2>&1 \
    | grep -E "Compiling whir|Finished|^error" || true
echo ""

SWEEP="./target/release/sweep"
if [[ ! -x "$SWEEP" ]]; then
    echo "ERROR: Build failed. Run manually to see errors:"
    echo "  cargo build --release --bin sweep --features gpu-metal,cli"
    exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────
median() {
    # Read values from args, sort numerically, return middle value.
    local -a vals=("$@")
    local n=${#vals[@]}
    if [[ $n -eq 0 ]]; then echo "fail"; return; fi
    IFS=$'\n' sorted=($(sort -g <<<"${vals[*]}")); unset IFS
    echo "${sorted[$(( n / 2 ))]}"
}

run_mode() {
    local n=$1 f=$2 r=$3 mode=$4
    local -a times=()
    for (( i=0; i<RUNS; i++ )); do
        local t
        t=$($SWEEP "$n" "$f" "$r" "$mode" 2>/dev/null) || continue
        times+=("$t")
    done
    if [[ ${#times[@]} -eq 0 ]]; then echo "fail"; return; fi
    median "${times[@]}"
}

run_config() {
    local n=$1 f=$2 r=$3

    local cpu_s gpu_s fused_s grind_s
    cpu_s=$(run_mode "$n" "$f" "$r" cpu)
    gpu_s=$(run_mode "$n" "$f" "$r" gpu)
    fused_s=$(run_mode "$n" "$f" "$r" gpu_fused)
    grind_s=$(run_mode "$n" "$f" "$r" gpu_grind)

    local cpu_ms gpu_ms fused_ms grind_ms
    if [[ "$cpu_s" == "fail" ]]; then cpu_ms="fail"; else cpu_ms=$(printf "%.1f" "$(echo "$cpu_s * 1000" | bc)"); fi
    if [[ "$gpu_s" == "fail" ]]; then gpu_ms="fail"; else gpu_ms=$(printf "%.1f" "$(echo "$gpu_s * 1000" | bc)"); fi
    if [[ "$fused_s" == "fail" ]]; then fused_ms="fail"; else fused_ms=$(printf "%.1f" "$(echo "$fused_s * 1000" | bc)"); fi
    if [[ "$grind_s" == "fail" ]]; then grind_ms="fail"; else grind_ms=$(printf "%.1f" "$(echo "$grind_s * 1000" | bc)"); fi

    local best_gpu="fail" ratio="-"
    for v in "$gpu_ms" "$fused_ms" "$grind_ms"; do
        if [[ "$v" != "fail" ]]; then
            if [[ "$best_gpu" == "fail" ]] || (( $(echo "$v < $best_gpu" | bc -l) )); then
                best_gpu="$v"
            fi
        fi
    done

    if [[ "$cpu_ms" != "fail" && "$best_gpu" != "fail" ]]; then
        ratio=$(printf "%.2fx" "$(echo "$cpu_ms / $best_gpu" | bc -l)")
    fi

    printf "%-6s %-6s %-6s %10s %10s %10s %10s %10s\n" \
        "$n" "$f" "$r" "$cpu_ms" "$gpu_ms" "$fused_ms" "$grind_ms" "$ratio"
}

print_header() {
    printf "%-6s %-6s %-6s %10s %10s %10s %10s %10s\n" \
        "n" "fold" "rate" "CPU(ms)" "GPU(ms)" "FUSED(ms)" "GRIND(ms)" "speedup"
    echo "--------------------------------------------------------------------------------"
}

# ── Run benchmarks ────────────────────────────────────────────────────
OUTFILE="bench_results_$(date +%Y%m%d_%H%M%S).txt"

{
    echo "# whir-p3-metal GPU benchmark"
    echo "# Date:  $(date)"
    echo "# Rust:  $RUSTC_VERSION"
    echo "# Host:  $HOST_ARCH $HOST_OS $HOST_VER"
    echo "# Chip:  $HOST_CHIP"
    echo "# Runs:  $RUNS per config (median reported)"
    echo "#"
    print_header

    if [[ -n "$SINGLE_N" ]]; then
        run_config "$SINGLE_N" "$SINGLE_F" "$SINGLE_R"
    elif $FULL; then
        for n in 18 20 22 24; do
            for f in 1 2 3 4 6 8; do
                for r in 1 2 3; do
                    echo "# Running n=$n f=$f r=$r ..." >&2
                    run_config "$n" "$f" "$r"
                done
            done
        done
    else
        for n_f_r in "20 1 1" "20 1 2" "20 1 3" "20 2 1" "20 2 2" "20 2 3" "20 4 1" "20 4 2" "20 4 3" \
                     "22 1 1" "22 1 2" "22 1 3" "22 2 1" "22 2 2" "22 2 3" "22 3 1" "22 3 2" "22 3 3" "22 4 1" "22 4 2" "22 4 3" "22 6 1" "22 6 2" "22 6 3" \
                     "24 1 1" "24 2 1" "24 3 1" "24 4 1" "24 6 1"; do
            read -r n f r <<< "$n_f_r"
            echo "# Running n=$n f=$f r=$r ..." >&2
            run_config "$n" "$f" "$r"
        done
    fi
} | tee "$OUTFILE"

echo ""
echo "Results saved to: $OUTFILE"
echo "Share this file to compare across devices."
