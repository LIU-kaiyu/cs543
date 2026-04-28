#!/usr/bin/env bash
# Download and extract the official DIODE validation split.
set -e

ROOT_DIR="data/raw/diode"
VAL_DIR="$ROOT_DIR/val"
ARCHIVE="$ROOT_DIR/val.tar.gz"
URL="https://diode-dataset.s3.amazonaws.com/val.tar.gz"

mkdir -p "$ROOT_DIR"

if [ -d "$VAL_DIR" ]; then
    echo "DIODE validation split already exists at $VAL_DIR"
    exit 0
fi

echo "Downloading DIODE validation split ..."
curl -L -o "$ARCHIVE" "$URL"

echo "Extracting to $ROOT_DIR ..."
tar -xzf "$ARCHIVE" -C "$ROOT_DIR"

echo "Done. Validation data is in $VAL_DIR"
