"""
Batch Grounding DINO inference over a folder of images (Task E).

Example:
    python src/batch_inference.py \
        --input_folder data --output_folder results/prompt_human_face \
        --prompt "human face" --box_threshold 0.35 --text_threshold 0.25

Outputs in --output_folder:
    images/<category>/<name>   annotated images (subfolder structure preserved,
                               so same-named files in different folders can't collide)
    predictions.json           one entry per image (schema below)
    detections.csv             one row per detection (images with 0 boxes get one empty row)
    run_config.json            args, device, library versions, git commit, timestamp
    crops/...                  optional, with --save_crops (input for a stage-2 classifier)

predictions.json keeps every field the original script wrote, so
summarize_results.py / compare_thresholds.py and older results/ folders still work:
    image, category, prompt, box_threshold, text_threshold, num_boxes,
    boxes            -> normalized (cx, cy, w, h), raw Grounding DINO output
    boxes_xyxy       -> NEW: absolute pixel (x1, y1, x2, y2)
    phrases, scores, inference_time_sec,
    image_width, image_height -> NEW
"""

import argparse
import csv
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, GroundingDinoDetector, crop  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# Folders skipped when scanning recursively. subsets_for_thresholds holds *copies*
# of images from other categories, so including it double-counts them.
DEFAULT_EXCLUDE = ["subsets_for_thresholds", "test_*"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_folder", required=True, help="Folder of images (scanned recursively)")
    p.add_argument("--output_folder", required=True)
    p.add_argument("--prompt", required=True, help='e.g. "human face" or "face . eyes . mouth"')
    p.add_argument("--box_threshold", type=float, default=0.35)
    p.add_argument("--text_threshold", type=float, default=0.25)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--cpu_only", action="store_true", help="Alias for --device cpu (kept for old commands)")
    p.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE,
                   help="Subfolder name patterns to skip (default: %(default)s). Pass --exclude with no values to include everything.")
    p.add_argument("--save_crops", action="store_true", help="Also save each detected region as its own image")
    p.add_argument("--crop_pad", type=float, default=0.2, help="Crop padding as a fraction of box size")
    p.add_argument("--overwrite", action="store_true", help="Allow writing into a folder that already has predictions.json")
    args = p.parse_args(argv)
    if args.cpu_only:
        args.device = "cpu"
    return args


def find_images(input_dir: Path, exclude_patterns):
    def excluded(path: Path) -> bool:
        rel_parts = path.relative_to(input_dir).parts[:-1]
        return any(Path(part).match(pat) for part in rel_parts for pat in exclude_patterns)

    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not excluded(p)
    )


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def library_versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "torchvision", "transformers"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return versions


def main(argv=None):
    args = parse_args(argv)
    input_dir, out_dir = Path(args.input_folder), Path(args.output_folder)

    if not input_dir.is_dir():
        sys.exit(f"Input folder not found: {input_dir}")
    if (out_dir / "predictions.json").exists() and not args.overwrite:
        sys.exit(f"{out_dir}/predictions.json already exists. Use a new --output_folder or pass --overwrite.")

    image_paths = find_images(input_dir, args.exclude or [])
    if not image_paths:
        sys.exit(f"No images ({', '.join(sorted(IMAGE_EXTS))}) found under {input_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    detector = GroundingDinoDetector(args.config, args.checkpoint, device=args.device)
    print(f"Device: {detector.device} | {len(image_paths)} images | prompt={args.prompt!r} "
          f"box={args.box_threshold} text={args.text_threshold}")

    from groundingdino.util.inference import annotate

    results, csv_rows = [], []
    for img_path in image_paths:
        rel = img_path.relative_to(input_dir)
        category = img_path.parent.name if img_path.parent != input_dir else input_dir.name

        res, image_rgb, (boxes, logits, phrases) = detector.detect(
            img_path, args.prompt, args.box_threshold, args.text_threshold
        )

        annotated = annotate(image_source=image_rgb, boxes=boxes, logits=logits, phrases=phrases)  # returns BGR
        out_img = out_dir / "images" / rel
        out_img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img), annotated)

        if args.save_crops:
            for i, det in enumerate(res.detections):
                c = crop(image_rgb, det.box_xyxy, pad=args.crop_pad)
                if c.size == 0:
                    continue
                crop_path = out_dir / "crops" / rel.parent / f"{rel.stem}_{i:02d}_{det.phrase.replace(' ', '-')}.jpg"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(crop_path), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))

        results.append({
            "image": str(rel),
            "category": category,
            "prompt": args.prompt,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "num_boxes": res.num_boxes,
            "boxes": [d.box_cxcywh_norm for d in res.detections],
            "boxes_xyxy": [d.box_xyxy for d in res.detections],
            "phrases": [d.phrase for d in res.detections],
            "scores": [d.score for d in res.detections],
            "inference_time_sec": res.inference_time_sec,
            "image_width": res.image_width,
            "image_height": res.image_height,
        })

        base = {"image": str(rel), "category": category, "prompt": args.prompt,
                "box_threshold": args.box_threshold, "text_threshold": args.text_threshold,
                "inference_time_sec": res.inference_time_sec}
        if res.detections:
            for i, d in enumerate(res.detections):
                csv_rows.append({**base, "det_index": i, "phrase": d.phrase, "score": d.score,
                                 "x1": d.box_xyxy[0], "y1": d.box_xyxy[1], "x2": d.box_xyxy[2], "y2": d.box_xyxy[3]})
        else:
            csv_rows.append({**base, "det_index": "", "phrase": "", "score": "", "x1": "", "y1": "", "x2": "", "y2": ""})

        print(f"{rel}: {res.num_boxes} boxes, {res.inference_time_sec:.2f}s")

    with open(out_dir / "predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(out_dir / "detections.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    times = [r["inference_time_sec"] for r in results]
    with open(out_dir / "run_config.json", "w") as f:
        json.dump({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": shlex.join(sys.argv),
            "args": vars(args),
            "device": detector.device,
            "git_commit": git_commit(),
            "versions": library_versions(),
            "num_images": len(results),
            "mean_inference_time_sec": round(sum(times) / len(times), 3),
        }, f, indent=2)

    print(f"Done: {len(results)} images -> {out_dir}")


if __name__ == "__main__":
    main()
