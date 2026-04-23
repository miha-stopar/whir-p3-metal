#!/usr/bin/env bash
set -euo pipefail

# Build the whir-p3 static library for iOS (device) and iOS Simulator.
# Usage: ./build-rust.sh [--release]
#
# Output: ios/lib/libwhir_p3.a  (fat binary: arm64-device + arm64-simulator)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

PROFILE="release"
CARGO_FLAGS="--release"

echo "=== Building whir-p3 static library for iOS ==="
echo ""

# Ensure iOS targets are installed
rustup target add aarch64-apple-ios aarch64-apple-ios-sim 2>/dev/null || true

# Build for device (arm64)
echo "[1/3] Building for aarch64-apple-ios (device)..."
cargo build $CARGO_FLAGS --lib --features ffi \
    --target aarch64-apple-ios

# Build for simulator (arm64, e.g. M-series Mac)
echo "[2/3] Building for aarch64-apple-ios-sim (simulator)..."
cargo build $CARGO_FLAGS --lib --features ffi \
    --target aarch64-apple-ios-sim

# Create output directory
mkdir -p "$SCRIPT_DIR/lib"

DEVICE_LIB="$ROOT/target/aarch64-apple-ios/$PROFILE/libwhir_p3.a"
SIM_LIB="$ROOT/target/aarch64-apple-ios-sim/$PROFILE/libwhir_p3.a"

# Create xcframework (preferred for Xcode) or just copy device lib
if xcodebuild -version &>/dev/null; then
    echo "[3/3] Creating xcframework..."
    rm -rf "$SCRIPT_DIR/lib/WhirBench.xcframework"
    xcodebuild -create-xcframework \
        -library "$DEVICE_LIB" -headers "$SCRIPT_DIR" \
        -library "$SIM_LIB" -headers "$SCRIPT_DIR" \
        -output "$SCRIPT_DIR/lib/WhirBench.xcframework"
    echo ""
    echo "Created: ios/lib/WhirBench.xcframework"
else
    echo "[3/3] Copying libraries..."
    cp "$DEVICE_LIB" "$SCRIPT_DIR/lib/libwhir_p3_device.a"
    cp "$SIM_LIB" "$SCRIPT_DIR/lib/libwhir_p3_sim.a"
    echo ""
    echo "Created: ios/lib/libwhir_p3_device.a"
    echo "Created: ios/lib/libwhir_p3_sim.a"
fi

echo ""
echo "Done. Open ios/WhirBench/WhirBench.xcodeproj in Xcode to build the app."
