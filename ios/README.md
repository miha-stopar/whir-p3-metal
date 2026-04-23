# WHIR Bench — iOS App

Benchmark the WHIR prover (CPU vs GPU/Metal) on iPhone and iPad.

## Prerequisites

1. **Rust nightly** with iOS targets:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y
   source "$HOME/.cargo/env"
   rustup target add aarch64-apple-ios aarch64-apple-ios-sim
   ```

2. **Xcode 15+** with iOS SDK

3. **xcodegen** (generates the Xcode project from `project.yml`):
   ```bash
   brew install xcodegen
   ```

## Quick Start

```bash
cd ios

# 1. Build the Rust static library for iOS
./build-rust.sh

# 2. Generate the Xcode project
xcodegen

# 3. Open in Xcode
open WhirBench.xcodeproj
```

Then in Xcode:
- Select your iPhone (or simulator) as the target device
- Press **⌘R** to build and run
- Tap the **Play** button in the app to start benchmarking

## Without xcodegen

If you prefer not to install xcodegen, create the Xcode project manually:

1. Open Xcode → File → New → Project → iOS → App
2. Name it `WhirBench`, language Swift, interface SwiftUI
3. Delete the auto-generated `ContentView.swift`
4. Add the files from `ios/WhirBench/` to the project
5. In Build Settings:
   - Set **Objective-C Bridging Header** to `WhirBench/WhirBench-Bridging-Header.h`
   - Add `$(SRCROOT)` to **Header Search Paths**
   - Add `$(SRCROOT)/lib` to **Library Search Paths**
   - Add to **Other Linker Flags**: `-lwhir_p3 -framework Metal -lc++ -lresolv`
6. Build the Rust library first: `./build-rust.sh`
7. Drag `ios/lib/libwhir_p3_device.a` into the Xcode project
8. Build and run

## Architecture

```
Swift App (ContentView.swift)
    │
    ├── whir_bench(n, fold, rate, mode)  ← C FFI call
    │       │
    │       └── src/ffi.rs  (Rust)
    │               │
    │               ├── run_cpu()    ← CPU prover (Radix2DFT + CPU Merkle)
    │               └── run_gpu()    ← GPU prover (Metal DFT + GPU Merkle)
    │
    └── whir_device_info()  ← returns Metal device name
```

The Rust code is compiled to a static library (`libwhir_p3.a`) for
`aarch64-apple-ios`. The Swift app calls it via the C functions declared
in `whir_bench.h`.

## Sharing Results

After running benchmarks, tap the share button to export results as text.
The export includes the device name, all timing results, and GPU/CPU speedup
ratios.
