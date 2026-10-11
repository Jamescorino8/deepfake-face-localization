# Day 15 — Thursday, 10/8

**Focus:** RWDF-23 access granted: sample real in-the-wild deepfake videos, re-run the 5 standard prompts on genuine
deepfake frames, and finalize the repo / Report v3

## To-Do

- [x] Verify `git remote -v` after the repo rename: already `Jamescorino8/deepfake-face-localization`, no change needed
- [x] Check for unpushed commits: none (`origin/main..HEAD` empty), and no stray "copy" PDFs in `reports/`, so nothing
  to delete
- [x] Gitignore all RWDF-23 data (metadata CSV, URL list, raw videos, extracted frames): licensed data, local only
- [x] Sample video links from the RWDF-23 metadata CSV and download with yt-dlp (720p max) until 20 succeed, logging
  failures
- [x] Extract 26 frames with visible faces into `data/deepfake_rwdf23/` (kept separate from the Kaggle placeholder in
  `data/deepfake/`)
- [x] Run the 5 standard prompts on the new frames, one output folder per prompt
- [x] Summarize results and compare against the Kaggle placeholder (Day 12)
- [x] Update README (test-set table, dataset status, RWDF-23 citation) and build `reports/report_v3.pdf`

## Experiments conducted

### Downloading RWDF-23 videos

```bash
yt-dlp --ignore-errors --no-playlist \
  -f "bv*[height<=720]+ba/b[height<=720]/bv*[height<=720]/b" \
  --merge-output-format mp4 -o "rwdf_XX.%(ext)s" "<url>"
```

Looped this over links sampled from the `Link` column of the RWDF-23 CSV until 20 videos downloaded successfully.

### Frame extraction

```bash
ffmpeg -ss <t> -i rwdf_XX.mp4 -frames:v 1 -q:v 2 candidates/rwdf_XX_t<t>.jpg
```

Looped this over 6 evenly spaced timestamps per video, then kept 26 frames with clearly visible faces in
`data/deepfake_rwdf23/`.

### Prompt runs

```bash
for pair in "human face:human_face" "person:person" "face . eyes . mouth:face_eyes_mouth" \
            "real face:real_face" "manipulated face:manipulated_face"; do
  python src/batch_inference.py --input_folder data/deepfake_rwdf23 \
    --output_folder results/prompt_v3_rwdf23_${pair##*:} --prompt "${pair%%:*}" --cpu_only --overwrite
done
python src/summarize_results.py
```

Then compared each "real face" / "manipulated face" box against its best-matching "human face" box on the same image
(IoU and score difference), for both RWDF-23 and the Day 12 Kaggle runs.

## Results obtained

| Prompt              | RWDF-23 avg boxes/img (26) | Kaggle avg (12, Day 12) | RWDF-23 mean score |
|---------------------|----------------------------|-------------------------|--------------------|
| human face          | 1.92                       | 1.33                    | 0.678              |
| person              | 2.08                       | 1.58                    | 0.678              |
| face . eyes . mouth | 6.42                       | 4.58                    | 0.526              |
| real face           | 1.92                       | 1.25                    | 0.713              |
| manipulated face    | 1.81                       | 1.17                    | 0.699              |

(RWDF-23 averages are higher mostly because more of the frames have several faces, e.g. a 9-face stacked edit.)

- **"Real face" / "manipulated face" still just track "human face" on genuine deepfakes.** Same box count as "human
  face" on 24/26 images for both. Matched boxes overlap almost exactly (48/50 "real face" and 40/47 "manipulated face"
  boxes at IoU > 0.9; the other 7 "manipulated face" boxes are still at IoU 0.71–0.90, the same face with slightly
  different box edges, and the 2 unmatched "real face" boxes are faces "human face" missed). Mean score shifts are small
  (+0.025, +0.000). On the same box, "manipulated face" scored *lower* than "real face" on 39 of 46 boxes (mean −0.033),
  on frames that are all from deepfake videos. On Kaggle it was lower on 14/14 (mean −0.074). This is a constant wording
  bias, not an authenticity signal.
- **Detection itself holds up on real deepfake frames.** By visual inspection, "human face" found 44 of the 46 faces
  physically in the scene. Misses: a face turned away and mostly covered by hair (rwdf_14_t006.6) and a child with her
  head bowed (rwdf_18_t003.0). Face-swap blending, watermarks ("Face swap made with Impressions app"), burned-in
  captions, news chyrons, and split-screen/stacked edits didn't disrupt detection.
- All 13 single-face frames gave exactly 1 "human face" box, scores 0.81–0.89 (mean 0.86).

### New failure modes

1. **Faces on screens and prints are detected as faces, sometimes above the real face.** 4 of 26 frames show someone
   holding a device or print with a face on it (rwdf_11 illustrated face on a tablet, rwdf_24 drawn face on a tablet,
   rwdf_36 two faces on a phone screen, rwdf_39 a photo on a tablet). All 5 such faces were boxed by every face prompt
   (0.37–0.83). In rwdf_39 the tablet face (0.82) outscored the real person holding it (0.70). In a 2-stage pipeline
   these crops go to the forensic classifier, which will likely call them "fake": false positives the localizer
   introduces. Holding up the "source" face showed up in 4 of the 20 videos sampled here.
2. **Prompt wording changes recall at the margin.** On the three-face frame (rwdf_18_t003.0), "real face" found the
   bowed child's face (0.38) that "human face" and "manipulated face" missed. On the 9-face stacked frame (rwdf_26),
   "manipulated face" found 7 vs. 9. On rwdf_14_t006.6, "human face" produced a duplicate (0.61 + 0.37 on the same face)
   that the other two didn't. The qualifier doesn't encode authenticity, but it shifts borderline faces across the 0.35
   threshold, so face counts can differ by prompt.
3. **Multi-face frames get much lower confidence.** "human face" averaged 0.62 on multi-face frames (range 0.36–0.82)
   vs. 0.86 on single-face frames, and faces other than the top-scoring one averaged 0.56. These are the first to drop
   under higher thresholds, consistent with the Day 12 threshold findings.
4. **"eyes" splits three ways:** on a clear frontal face "face . eyes . mouth" returns left eye, right eye, and a
   combined both-eyes box (exactly 3 "eyes" boxes on 12 of 13 single-face frames), which inflates box counts.

## Challenges Encountered

| # | Challenge                                                                                                                                                                                                                 | Fix                                                                                                                                                                                                                                             |
|---|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | yt-dlp 2025.08.11 failed on every YouTube link ("content is not available on this app") and on Reddit/Bilibili (auth required / HTTP 412)                                                                                 | `brew upgrade yt-dlp` → 2026.08.19 fixed Reddit and Bilibili; YouTube now asks for sign-in (bot check), which would need `--cookies-from-browser`. Decided not to use account cookies; kept sampling other platforms until 20 videos downloaded |
| 2 | `batch_inference.py` crashed on import: scipy 1.15.3's PyPI wheel fails to load on macOS 27 (`section '__DATA/__thread_bss' has a zero-fill section type, but offset field is not zero`). The env was built on macOS 26.6 | `conda install -c conda-forge "scipy=1.15" "numpy=2.2.6"` (conda-forge build loads fine; numpy pinned to keep it unchanged)                                                                                                                     |
| 3 | Wanted an independent face pre-filter for frame selection, but `cv2.CascadeClassifier` is missing in this env (OpenCV 5.0 moved Haar cascades out of the main module)                                                     | Selected frames visually from contact sheets instead                                                                                                                                                                                            |
| 4 | rwdf_34 (Bilibili) has corrupted HEVC packets (`Invalid NAL unit size`)                                                                                                                                                   | 3 of 6 candidate frames still decoded cleanly; used one of those                                                                                                                                                                                |

## References

RWDF-23: Cho, B., Le, B. M., Kim, J., Woo, S., Tariq, S., Abuadbba, A., & Moore, K. (2023). Towards Understanding of
Deepfake Videos in the Wild. *Proceedings of the 32nd ACM International Conference on Information and Knowledge
Management (CIKM)*, 4530–4537.
