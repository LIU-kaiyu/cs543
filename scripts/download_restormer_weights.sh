#!/usr/bin/env bash
# Download Restormer pretrained weights for deraining and real denoising.
#
# Deraining weights:      Deraining/pretrained_models/deraining.pth
# Real denoising weights: Denoising/pretrained_models/real_denoising.pth
#
# Run once before using the "restormer" or "auto-restormer" preprocessing strategies.
set -e

RESTORMER_DIR="third_party/Restormer"

if [ ! -d "$RESTORMER_DIR" ]; then
    echo "Cloning Restormer..."
    git clone --depth 1 https://github.com/swz30/Restormer "$RESTORMER_DIR"
fi

echo "  Installing gdown..."
pip install -q gdown

# ── Deraining ────────────────────────────────────────────────────────────────
DERAIN_DIR="$RESTORMER_DIR/Deraining/pretrained_models"
mkdir -p "$DERAIN_DIR"

if [ -f "$DERAIN_DIR/deraining.pth" ]; then
    echo "  deraining.pth already exists, skipping."
else
    echo "  Downloading deraining weights..."
    # Folder: https://drive.google.com/drive/folders/1ZEDDEVW0UgkpWi-N4Lj_JUoVChGXCu_u
    gdown --folder "1ZEDDEVW0UgkpWi-N4Lj_JUoVChGXCu_u" -O "$DERAIN_DIR"
fi

# ── Real Denoising ───────────────────────────────────────────────────────────
DENOISE_DIR="$RESTORMER_DIR/Denoising/pretrained_models"
mkdir -p "$DENOISE_DIR"

if [ -f "$DENOISE_DIR/real_denoising.pth" ]; then
    echo "  real_denoising.pth already exists, skipping."
else
    echo "  Downloading real denoising weights (~170 MB)..."
    # Direct file: https://drive.google.com/file/d/1FF_4NTboTWQ7sHCq4xhyLZsSl0U0JfjH
    gdown "1FF_4NTboTWQ7sHCq4xhyLZsSl0U0JfjH" -O "$DENOISE_DIR/real_denoising.pth"
fi

echo ""
echo "Done."
echo "  Deraining:      $DERAIN_DIR/deraining.pth"
echo "  Real denoising: $DENOISE_DIR/real_denoising.pth"
