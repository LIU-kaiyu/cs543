#!/usr/bin/env bash
# Create and activate the midas-benchmark conda environment.
set -e

conda create -n midas-benchmark python=3.10 -y
conda activate midas-benchmark
pip install -r requirements.txt

echo "Environment ready. Run: conda activate midas-benchmark"
