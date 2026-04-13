#!/usr/bin/env bash
# Download MiDaS model weights into third_party/MiDaS/weights/
# Weights are hosted on GitHub Releases for the intel-isl/MiDaS repo.
set -e

WEIGHTS_DIR="third_party/MiDaS/weights"
mkdir -p "$WEIGHTS_DIR"

download() {
    local name="$1"
    local url="$2"
    if [ -f "$WEIGHTS_DIR/$name" ]; then
        echo "  $name already exists, skipping."
    else
        echo "  Downloading $name ..."
        curl -L -o "$WEIGHTS_DIR/$name" "$url"
    fi
}

# Primary benchmark model
download "dpt_hybrid_384.pt" \
    "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt"

# Fast baseline (download later when needed)
# download "midas_v21_small_256.pt" \
#     "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"

echo "Done. Weights are in $WEIGHTS_DIR/"
