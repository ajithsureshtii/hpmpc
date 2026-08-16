#!/bin/bash
# Builds the hpmpc party executables for Flotilla's secure aggregation (see
# flotilla/docs/secure_aggregation/hpmpc_backend.md), for one of three
# protocols, and writes a build-metadata sidecar file that backend_hpmpc.py's
# config-consistency check reads at party startup, so a
# protocol/bitlength/frac_bits mismatch between a party's YAML config and
# these compiled executables fails loudly at start() instead of silently
# corrupting a round.
#
# Usage: scripts/build_secure_agg.sh <replicated|trio|tetrad|trio_mult> [bitlength] [frac_bits]
#
# "trio_mult" builds mult_fedavg_secure_aggregation.hpp (FUNCTION_IDENTIFIER=91
# -- the "mpc_product" weighting-mode variant, see
# docs/secure_aggregation/mult_fedavg.md) instead of the default
# fedavg_secure_aggregation.hpp (FUNCTION_IDENTIFIER=90); every other
# protocol name here builds the default program. Currently Trio (PROTOCOL=5)
# only -- see mult_fedavg_secure_aggregation.hpp's module docstring.
#
# Builds into executables/<protocol_name>/, NOT executables/ directly --
# the Makefile's compiled output filenames (run-P{i}.o) are NOT
# protocol-namespaced, so building a second protocol into the same directory
# would silently overwrite the first one's binaries. Point a given party's
# backend.hpmpc.executable_dir at the matching executables/<protocol_name>/
# subdirectory.
#
# Must be run from a host/container with a working C++20 toolchain. If you
# hit "'algorithm' file not found" or similar on macOS, that's almost always
# a broken/incomplete Xcode Command Line Tools install (missing libc++
# headers), not an hpmpc problem -- building inside a Linux container
# sidesteps it entirely, which is also what docker/Dockerfile.secure_agg_party
# does for the real deployment image.
#
# RANDOM_ALGORITHM=0 (portable xorshift) and USE_SSL=0 are deliberately used
# here as dev/test-friendly defaults that avoid needing x86 AES-NI intrinsics
# (see config.h/core/crypto/aes/AES.h) -- fine for correctness testing, but a
# production deployment should switch to RANDOM_ALGORITHM=2 (AES-based) and
# USE_SSL=1 on a platform where that's verified to build (x86 with AES-NI, or
# after confirming the ARM-specific AES/SHA paths), and provision real
# per-deployment TLS certs instead of the checked-in self-signed one.
#
# PRE (preprocessing phase) and MAL are deliberately left at their config.h
# defaults (PRE=0, MAL auto-1-for-PROTOCOL>7) for all three protocols here --
# PRE=1 swaps in partly-stubbed *_POST_Share/_init classes (e.g.
# OECL_MAL3_POST_Share for Tetrad's party 3) not covered by this build.

set -euo pipefail

PROTOCOL_NAME="${1:?usage: build_secure_agg.sh <replicated|trio|tetrad|trio_mult> [bitlength] [frac_bits]}"
BITLENGTH="${2:-64}"
FRAC_BITS="${3:-13}"

case "$PROTOCOL_NAME" in
    replicated) PROTOCOL_NUM=2; PARTIES="0 1 2"; FUNCTION_IDENTIFIER=90; METADATA_NAME=fedavg_secure_aggregation ;;
    trio)       PROTOCOL_NUM=5; PARTIES="0 1 2"; FUNCTION_IDENTIFIER=90; METADATA_NAME=fedavg_secure_aggregation ;;
    tetrad)     PROTOCOL_NUM=8; PARTIES="0 1 2 3"; FUNCTION_IDENTIFIER=90; METADATA_NAME=fedavg_secure_aggregation ;;
    trio_mult)  PROTOCOL_NUM=5; PARTIES="0 1 2"; FUNCTION_IDENTIFIER=91; METADATA_NAME=mult_fedavg_secure_aggregation ;;
    *)
        echo "unknown protocol: $PROTOCOL_NAME (expected replicated|trio|tetrad|trio_mult)" >&2
        exit 1
        ;;
esac

cd "$(dirname "$0")/.."
OUT_DIR="executables/$PROTOCOL_NAME"
mkdir -p "$OUT_DIR"

for PARTY in $PARTIES; do
    echo "Building $PROTOCOL_NAME party $PARTY (bitlength=$BITLENGTH, frac_bits=$FRAC_BITS, function_identifier=$FUNCTION_IDENTIFIER)..."
    make PARTY="$PARTY" PROTOCOL="$PROTOCOL_NUM" FUNCTION_IDENTIFIER="$FUNCTION_IDENTIFIER" \
        BITLENGTH="$BITLENGTH" FRACTIONAL="$FRAC_BITS" DATTYPE=64 \
        RANDOM_ALGORITHM=0 USE_SSL=0
    mv "executables/run-P${PARTY}.o" "$OUT_DIR/run-P${PARTY}.o"
done

cat > "$OUT_DIR/${METADATA_NAME}.build_metadata.json" <<EOF
{
  "bitlength": $BITLENGTH,
  "frac_bits": $FRAC_BITS,
  "protocol": $PROTOCOL_NUM,
  "function_identifier": $FUNCTION_IDENTIFIER
}
EOF

echo "Built $OUT_DIR/run-P{$(echo "$PARTIES" | tr ' ' ',')}.o"
echo "Wrote $OUT_DIR/${METADATA_NAME}.build_metadata.json"
