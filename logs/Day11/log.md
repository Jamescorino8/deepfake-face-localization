# Day 11 — Monday, 8/17

**Focus:** Complete occluded/multiple data expansion; re-run batch inference across new/expanded categories

## To-Do
- [x] Expand `data/occluded/` from 2 → 7 images (masks, hands, hair, sunglasses+scarf, partial crops)
- [x] Expand `data/multiple/` from 4 → 5 images (one additional dense crowd scene)
- [x] Re-run `src/batch_inference.py` with the default 5-prompt set across occluded/multiple/small_lowres/deepfake (deepfake folder still empty at this point — ran as a harmless no-op, same behavior as the Day04 empty-folder edge case)
- [x] Spot-check annotated outputs per new category
- [x] Confirm final image counts across all categories

## Experiments conducted
- Confirmed image counts: single 10, profile 5, multiple 5 (was 4), occluded 7 (was 2), small_lowres 6, deepfake 0 (pending).
- Ran the 5 standard prompts ("human face", "person", "face . eyes . mouth", "real face", "manipulated face") against each of occluded, multiple, small_lowres, and deepfake individually (20 runs total, `results/prompt_v2_<category>_<prompt>/`).

## Results obtained
- **Occluded (7 images):** highly consistent at default threshold — 6 of 7 images produced exactly 1 box; the one exception (a hair-covering-face image) produced 2 overlapping boxes at different confidences (0.60, 0.39).
- **Multiple (5 images):** the new crowd photo (hundreds of people, faces ~30–40px) returned **0 detections** across 4 of 5 prompts — a different failure mode than the Day05 duplicate-box outlier (nicholas-green, still 18 boxes on the same prompt). This is the most significant finding of the week so far.
- **Small_lowres (6 images):** single-subject thumbnails (32–56px) all still detected at 1 box each, comparable confidence to full resolution. Multi-subject thumbnails (90px) still detected 4–5 faces but at reduced count/confidence — partial, not complete, degradation.
- **Deepfake:** empty folder, 0 images processed, no errors — confirmed the pipeline handles this gracefully ahead of tomorrow's Kaggle sample.

## Challenges Encountered
| # | Challenge | Fix |
|---|-----------|-----|
| 1 | None significant today | — |

## Planned next steps
- Download Kaggle deepfake-image sample into `data/deepfake/`
- Run the 5 standard prompts against the deepfake sample
- Threshold sweep (0.25/0.20, 0.35/0.25, 0.45/0.30) on multiple/occluded/small_lowres, specifically checking whether a lower threshold recovers anything on the zero-detection crowd photo
- Update report and weekly progress report with this week's findings
