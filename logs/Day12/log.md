# Day 12 — Tuesday, 8/18

**Focus:** Deepfake-sample prompt experiments; threshold sweep on new/expanded categories; build Findings Report v2

## To-Do
- [x] Download Kaggle deepfake-image sample (manjilkarki/deepfake-and-real-images, Test/Fake split) — 12 images added to `data/deepfake/`
- [x] Run the 5 standard prompts against `data/deepfake/`
- [x] Threshold sweep (0.25/0.20, 0.35/0.25, 0.45/0.30) on occluded/multiple/small_lowres
- [x] Fix a shell-portability bug in the threshold-sweep script (see Challenges)
- [x] Build `Reports/GroundingDINO_Report_v2.pdf` incorporating occluded/multiple/small_lowres/deepfake findings
- [x] Final README pass — category counts, dataset-access status
- [x] Update weekly progress report for next week's meeting

## Experiments conducted
- Ran the 5 standard prompts against the 12-image Kaggle deepfake sample at default thresholds.
- Ran the threshold sweep on the full `occluded`, `multiple`, and `small_lowres` folders (9 runs total).
- Re-ran the sweep a second time after fixing the shell bug below.

## Results obtained
- **Deepfake sample (exploratory):** avg boxes/image — human face 1.33, person 1.58, face.eyes.mouth 4.58, real face 1.25, manipulated face 1.17 (with the sample's only zero-detection). "Real face" and "manipulated face" both track closely with plain "human face," replicating the Day05 finding on ordinary photos, now on a (non-forensic) deepfake-labeled sample.
- **Threshold sweep confirms the crowd-density finding is not threshold-fixable:** even the most permissive setting (0.25/0.20) recovers only 2 very-low-confidence boxes (0.296, 0.264) on the zero-detection crowd photo, vs. hundreds of visible faces. Occluded and single/profile-derived small_lowres images behave like the original categories (0.35/0.25 remains a reasonable default); the dense-crowd case does not.
- Multi-subject small_lowres images (downscaled group photos) show a steeper decline across thresholds than their full-resolution counterparts.
- Built and finalized `Reports/GroundingDINO_Report_v2.pdf` (8 pages) — a standalone update to the original findings report covering all of this week's new categories and results, including the multiple-category crowd-density finding.

## Challenges Encountered
| # | Challenge | Fix |
|---|-----------|-----|
| 1 | Threshold-sweep script failed silently — `set -- $combo` (used to split a "box text" pair) does not word-split the same way in zsh (macOS's default shell) as in bash, so all 9 sweep runs errored out with an `invalid float value` argparse error before writing any output | Switched to shell-portable parameter-expansion splitting (`${combo%%:*}` / `${combo##*:}`), re-ran successfully |

## Planned next steps
- Continue following up on FaceForensics++/RWDF-23/FakeAVCeleb access
- Once one of the three datasets is granted, re-run the "real face"/"manipulated face" comparison on genuine manipulated content and replace the Kaggle placeholder
- Investigate the crowd-density detection failure further (e.g. tiling the image before inference)
- Continue PyTorch course
- Commit this week's work (data, scripts, logs, report) — pending final go-ahead