// C header for whir-p3-metal FFI benchmark functions.
// Auto-generated from src/ffi.rs — keep in sync.

#ifndef WHIR_BENCH_H
#define WHIR_BENCH_H

#include <stdint.h>

typedef struct {
    double cpu_ms;
    double gpu_ms;
    double fused_ms;
    double grind_ms;
} BenchResult;

// Run a benchmark config. Returns times in milliseconds (-1.0 = failed/skipped).
// mode: 0 = all, 1 = CPU only, 2 = GPU fused only, 3 = GPU grind only.
BenchResult whir_bench(uint32_t n, uint32_t fold, uint32_t rate, uint32_t mode);

// Returns a static string describing the Metal device.
const char* whir_device_info(void);

#endif
