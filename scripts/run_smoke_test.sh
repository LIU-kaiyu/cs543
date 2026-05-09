#!/usr/bin/env bash
# Run smoke test: MiDaS inference on sample images in third_party/MIDAS/input/
set -e

python - <<'EOF'
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.adapters.midas_adapter import MiDaSAdapter
from pathlib import Path
import glob

adapter = MiDaSAdapter(model_type="dpt_hybrid_384")

input_dir = Path("third_party/MIDAS/input")
image_paths = sorted(glob.glob(str(input_dir / "*.jpg")) + glob.glob(str(input_dir / "*.png")))

if not image_paths:
    print("No images found in third_party/MIDAS/input/ — add some RGB images first.")
    sys.exit(1)

records = adapter.run_batch(image_paths, output_dir="outputs/predictions/smoke", verbose=True)
errors = [r for r in records if r["error"]]
print(f"\nSmoke test complete: {len(records) - len(errors)}/{len(records)} succeeded.")
if errors:
    for e in errors:
        print(f"  ERROR: {e['image_path']} — {e['error']}")
EOF
