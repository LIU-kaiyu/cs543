#!/usr/bin/env bash
# Clone the upstream MiDaS repo into third_party/MiDaS if it is missing.
set -e

TARGET_DIR="third_party/MiDaS"
REPO_URL="https://github.com/isl-org/MiDaS.git"

if [ -d "$TARGET_DIR/midas" ]; then
    echo "MiDaS already present at $TARGET_DIR"
    exit 0
fi

mkdir -p third_party
echo "Cloning MiDaS into $TARGET_DIR ..."
git clone --depth 1 "$REPO_URL" "$TARGET_DIR"
echo "Done."
