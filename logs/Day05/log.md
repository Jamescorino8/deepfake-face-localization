# Day 05 — Tuesday, 8/4

**Focus:** Task C prompt experiments + Task D threshold sweep, on available categories (deepfake frames still pending)

## To-Do
- [ ✅ ] Confirm final image counts per category before running experiments
- [ ✅ ] Run detections with prompts "human face" and "person" across the full available set (single, multiple, profile, occluded)
- [ ✅ ] Save annotated outputs and note observations per category
- [ ✅ ] Run detections with prompts "face . eyes . mouth", "real face", "manipulated face"
- [ ✅ ] Document results per category; note that "real face" / "manipulated face" results are exploratory, not reliable forensic signals
- [ ✅ ] Consolidate per-category/per-prompt results (num_boxes, scores) into a summary
- [ ✅ ] Run initial threshold sweep (Task D) on a representative subset (1 image each from multiple, single, profile) across 3 box/text threshold combinations
- [ ✅ ] Compare threshold sweep results for missed faces, duplicate boxes, false detections, confidence scores
- [ ✅ ] Update README to reflect final image counts/categories used, note on pending deepfake category, and instructions for running the threshold sweep

## Experiments conducted

### Confirming image counts
```bash
python -c "
from pathlib import Path
total = 0
for folder in sorted(Path('data').iterdir()):
    if folder.is_dir():
        imgs = list(folder.glob('*.jp*g')) + list(folder.glob('*.png'))
        print(folder.name, len(imgs))
        total += len(imgs)
print('total:', total)
"
```
- Confirmed 21 images total across four categories (10 single, 5 profile, 4 multiple, 2 occluded) — meets the "at least 20 images" requirement even before the deepfake category is added.

### Task C — running all five prompts
Ran each prompt across the full `data/` folder, one output folder per prompt to avoid overwriting `predictions.json`:
```bash
python src/batch_inference.py \
  --input_folder data \
  --output_folder results/prompt_human_face \
  --prompt "human face" \
  --cpu_only

python src/batch_inference.py \
  --input_folder data \
  --output_folder results/prompt_person \
  --prompt "person" \
  --cpu_only

python src/batch_inference.py \
  --input_folder data \
  --output_folder results/prompt_face_eyes_mouth \
  --prompt "face . eyes . mouth" \
  --cpu_only

python src/batch_inference.py \
  --input_folder data \
  --output_folder results/prompt_real_face \
  --prompt "real face" \
  --cpu_only

python src/batch_inference.py \
  --input_folder data \
  --output_folder results/prompt_manipulated_face \
  --prompt "manipulated face" \
  --cpu_only
```
- Spot-checked a handful of annotated images per category between runs to sanity-check box placement before moving to the next prompt.

### Consolidating results
```python
import json
from pathlib import Path

summary = {}
for run_dir in Path("results").glob("prompt_*"):
    data = json.load(open(run_dir / "predictions.json"))
    prompt = data[0]["prompt"] if data else run_dir.name
    for entry in data:
        cat = entry["category"]
        summary.setdefault(cat, {}).setdefault(prompt, []).append(entry["num_boxes"])

for cat, prompts in summary.items():
    print(cat)
    for prompt, counts in prompts.items():
        print(f"  {prompt}: {counts}")
```
Saved as `src/summarize_results.py`:
```bash
python src/summarize_results.py
```

### Task D — building the threshold-sweep subset
```bash
mkdir -p data/subsets_for_thresholds
```
- Copied 3 representative images into the subset: the `multiple` outlier (highest box count), one `single` image, and one `profile` image.

### Task D — running the threshold sweep
```bash
python src/batch_inference.py --input_folder data/subsets_for_thresholds \
  --output_folder results/thresh_box025_text020 --prompt "human face" \
  --box_threshold 0.25 --text_threshold 0.20 --cpu_only

python src/batch_inference.py --input_folder data/subsets_for_thresholds \
  --output_folder results/thresh_box035_text025 --prompt "human face" \
  --box_threshold 0.35 --text_threshold 0.25 --cpu_only

python src/batch_inference.py --input_folder data/subsets_for_thresholds \
  --output_folder results/thresh_box045_text030 --prompt "human face" \
  --box_threshold 0.45 --text_threshold 0.30 --cpu_only
```

### Comparing threshold results
```python
import json
from pathlib import Path

for run_dir in sorted(Path("results").glob("thresh_*")):
    data = json.load(open(run_dir / "predictions.json"))
    print(run_dir.name)
    for entry in data:
        print(f"  {entry['image']}: {entry['num_boxes']} boxes, scores={entry['scores']}")
    print()
```
Saved as `src/compare_thresholds.py`:
```bash
python src/compare_thresholds.py
```

## Results obtained

### Task C — per-category/per-prompt observations
- **"face . eyes . mouth"** consistently produced more boxes than "human face" across all categories (e.g. single: `[5,7,4,3,7,5,7,5,11,8]` vs. `[1,1,4,1,1,1,2,1,2,2]`) — expected, since the `.`-separated prompt creates distinct grounding targets (face, eyes, mouth as separate detections) rather than one box per face.
- **"real face" and "manipulated face"** tracked closely with "human face" in nearly all cases (e.g. multiple: human=`[18,3,5,6]`, real=`[17,3,5,6]`, manipulated=`[3,2,5,6]`), indicating the model is not meaningfully responding to the "real"/"manipulated" qualifier — it is still primarily matching on "face." Confirms Tamer's caveat that these prompts are exploratory and not a reliable forensic signal on their own, especially since no actual deepfake content was tested in this round.
- **profile** and single-face categories were mostly stable at 1 box per image across face-related prompts, suggesting reliable single-face detection despite angle/occlusion in this sample.
- One clear outlier in `multiple` (18–26 boxes depending on prompt/threshold) selected for the Task D sweep to investigate whether this reflects a genuinely crowded scene or duplicate-box artifacts.

### Task D — threshold sweep results

| Image (category) | box=0.25/text=0.20 | box=0.35/text=0.25 | box=0.45/text=0.30 | Interpretation |
|---|---|---|---|---|
| lfw_02.jpg (single) | 4 boxes, scores `[0.55, 0.404, 0.363, 0.358]` | 4 boxes, identical scores | 1 box, score `[0.55]` | Duplicate boxes on the same face at low/mid threshold; 0.45 correctly collapses to the single true detection |
| nicholas-green...jpg (multiple) | 26 boxes | 18 boxes | 1 box | Heavy duplication/noise at low threshold; 0.45 is too aggressive and likely undercounts real faces in a crowded scene |
| pexels-moh-adbelghaffar...jpg (profile) | 3 boxes, scores `[0.402, 0.321, 0.258]` | 1 box, score `[0.402]` | 0 boxes | 0.45 misses the face entirely — occluded/angled faces need a lower threshold to avoid false negatives |

**Overall finding:** no single threshold performs well across all conditions. Frontal, unobstructed faces tolerate (and benefit from) a higher threshold to eliminate duplicate boxes, while occluded, angled, or crowded images require a lower threshold or risk missing real detections entirely. The library default (box=0.35/text=0.25) appears to be a reasonable general-purpose middle ground but is not optimal for every category — a strong, concrete finding for the short report's threshold analysis and limitations sections.

## Challenges Encountered
| # | Challenge | Fix |
|---|-----------|-----|
| 1 | Large disparity in box counts (up to 26) for the `multiple` outlier image made it unclear whether results reflected real detections or duplicate-box artifacts | Selected this image for the Task D threshold sweep specifically to investigate; flagged for visual confirmation against the annotated output before finalizing report conclusions |

## Planned next steps
- Write the short report using Task C and Task D findings above.
- Build 5–7 slide presentation from the short report.
- Follow up on FaceForensics++ access requests and incorporate deepfake frames once available.