# Day 10 — Friday, 8/14

**Focus:** Response to Tamer's Aug 2026 feedback — dataset access research, submit RWDF-23/FakeAVCeleb requests, add small_lowres category, curate occluded/multiple expansion candidates

## To-Do
- [x] Research RWDF-23 and FakeAVCeleb (DASH Lab) as alternative deepfake-video sources
- [x] Submit RWDF-23 access request (Google Form)
- [x] Submit FakeAVCeleb access request (Google Form + license agreement)
- [x] Send a follow-up nudge on the still-pending FaceForensics++ request
- [x] Create `data/small_lowres/` (6 images) — thumbnail-resolution downscales of existing curated photos
- [x] Write `src/make_small_lowres.py` to make the downscale step reproducible
- [x] Curate a shortlist of candidate photos for occluded (masks, hands, hair, sunglasses+scarf) and multiple (second dense crowd scene) expansion
- [x] Update README with new category counts / dataset-access status (initial pass)
- [x] Draft `DayXX/log.md` templates for the rest of the week

## Notes — dataset access research

Checked the DASH Lab datasets page and both leads Tamer suggested (RWDF-23, FakeAVCeleb). Neither is instant — both are Google Form-gated (FakeAVCeleb also requires a license agreement), same as FaceForensics++, just likely faster turnaround since they're the advisor's own lab's datasets. Submitted both today so the clock starts; not blocking the rest of the week's work on approval landing in time. Identified a Kaggle deepfake-image sample (manjilkarki/deepfake-and-real-images) as an immediate, clearly-labeled placeholder in the meantime (no approval wait, free account only) — to be downloaded once the occluded/multiple expansion is in place.

## Experiments conducted
- Generated `data/small_lowres/` (6 images: 2 single, 1 profile, 1 occluded, 2 multiple, all downscaled to 32–90px long edge) via `src/make_small_lowres.py`. No new source images needed — isolates resolution as the sole variable by downscaling already-curated photos.

## Results obtained
- `small_lowres` category in place and ready for prompt experiments once the rest of the week's data expansion is done.

## Challenges Encountered
| # | Challenge | Fix |
|---|-----------|-----|
| 1 | Needed new occluded/multiple photos but had no fast, license-safe way to batch-download stock images automatically | Curated a manual shortlist (Unsplash/Pexels search links) to download by hand, same workflow as the original Day04 curation |

## Planned next steps
- Download and add the curated occluded (+5) and multiple (+1 crowd scene) images
- Re-run `src/batch_inference.py` across the new/expanded categories
- Threshold sweep on occluded/multiple/small_lowres
- Deepfake-sample prompt experiments once Kaggle sample is added
- Report update, README final pass, weekly progress report
