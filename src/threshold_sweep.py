"""
Task D threshold sweep, in Python instead of a shell loop (the zsh/bash
word-splitting bug from Day 12 can't happen here). Loads the model once and
reuses it for every threshold pair, so it's also much faster than N separate runs.

Example (reproduces the Day05/Day12 sweeps):
    python src/threshold_sweep.py --input_folder data/occluded --prompt "human face" \
        --pairs 0.25:0.20 0.35:0.25 0.45:0.30 --output_prefix results/thresh_occluded

Creates results/thresh_occluded_box025_text020/ etc. -- same naming and file layout
as batch_inference.py, so compare_thresholds.py picks them up -- plus a
<prefix>_summary.csv with one row per (image, threshold pair).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_inference  # noqa: E402


def parse_pair(s):
    try:
        box, text = s.split(":")
        return float(box), float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected BOX:TEXT like 0.35:0.25, got {s!r}")


def tag(x):
    return f"{round(x * 100):03d}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_folder", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--pairs", nargs="+", type=parse_pair, default=[(0.25, 0.20), (0.35, 0.25), (0.45, 0.30)])
    p.add_argument("--output_prefix", required=True, help="e.g. results/thresh_occluded")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--config", default=None, help="Defaults to the Swin-T config in GroundingDINO/")
    p.add_argument("--checkpoint", default=None, help="Defaults to weights/groundingdino_swint_ogc.pth")
    p.add_argument("--overwrite", action="store_true")
    args, passthrough = p.parse_known_args()

    # Load the model once and hand it to every run.
    from detector import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, GroundingDinoDetector
    shared = GroundingDinoDetector(args.config or DEFAULT_CONFIG, args.checkpoint or DEFAULT_CHECKPOINT, device=args.device)
    batch_inference.GroundingDinoDetector = lambda *a, **k: shared

    summary = []
    for box_t, text_t in args.pairs:
        out = f"{args.output_prefix}_box{tag(box_t)}_text{tag(text_t)}"
        argv = ["--input_folder", args.input_folder, "--output_folder", out, "--prompt", args.prompt,
                "--box_threshold", str(box_t), "--text_threshold", str(text_t), *passthrough]
        if args.overwrite:
            argv.append("--overwrite")
        print(f"\n=== box={box_t} text={text_t} -> {out}")
        batch_inference.main(argv)

        for e in json.load(open(Path(out) / "predictions.json")):
            summary.append({"image": e["image"], "category": e["category"], "box_threshold": box_t,
                            "text_threshold": text_t, "num_boxes": e["num_boxes"],
                            "max_score": max(e["scores"], default=""),
                            "scores": " ".join(map(str, e["scores"]))})

    summary_path = Path(f"{args.output_prefix}_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
