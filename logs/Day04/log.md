# Day 04 — Monday, 8/3

**Focus:** Batch inference script — build, debug, and test (Task E); begin image curation (Task C)

## To-Do
- [ ✅ ] Curate 20+ test images: single frontal face, multiple faces, profile/occluded faces, real images
    - Note: Deepfake dataset frames are not yet available; will follow up on FaceForensics++ access requests
- [ ✅ ] Organize images into labeled folders by category
- [ ✅ ] Build initial batch-inference script (`src/batch_inference.py`) — accepts input folder, prompt, and thresholds as CLI args; saves annotated images + predictions.json; records per-image inference time
- [ ✅ ] Debug and resolve script errors
- [ ✅ ] Test script on a small known image before running full experiments

## Experiments conducted
- Wrote `src/batch_inference.py`, wrapping `load_model`, `load_image`, `predict`, and `annotate` from `groundingdino.util.inference` in a loop over an image folder, with `argparse` for `--input_folder`, `--output_folder`, `--prompt`, `--box_threshold`, `--text_threshold`, and `--cpu_only`.
- Set up smoke-test folder and ran initial test:
```bash
    mkdir -p data/test_mini
    cp Day03/sample_image_raw.jpeg data/test_mini/

    python src/batch_inference.py \
      --input_folder data/test_mini \
      --output_folder results/test_mini \
      --prompt "person" \
      --cpu_only
```
- Debugged two errors during the above run, re-ran after each fix until clean.
- Validated output:
```bash
    python -m json.tool results/test_mini/predictions.json

    python -c "
    import json
    data = json.load(open('results/test_mini/predictions.json'))
    for entry in data:
        print(entry['image'], entry['num_boxes'], entry['scores'], entry['inference_time_sec'])
    "
```
- Ran edge-case tests:
```bash
    # empty folder
    mkdir -p data/test_empty
    python src/batch_inference.py --input_folder data/test_empty --output_folder results/test_empty --prompt "person" --cpu_only

    # non-image files mixed in (.DS_Store etc. left in test_mini/)
    python src/batch_inference.py --input_folder data/test_mini --output_folder results/test_mini --prompt "person" --cpu_only

    # deliberately mismatched prompt -> expect 0 detections
    python src/batch_inference.py --input_folder data/test_mini --output_folder results/test_mismatch --prompt "bicycle" --cpu_only

    # re-run into same output folder with different prompt -> confirm overwrite behavior
    python src/batch_inference.py --input_folder data/test_mini --output_folder results/test_mini --prompt "face" --cpu_only
```
- Spot-checked annotated images in `results/test_mini/images/` and `results/test_mismatch/images/` visually for correct box placement and normal (non-BGR-swapped) colors.
- Curated and organized images for three of five categories (`single`, `multiple`, `profile`) into `data/` subfolders:
  - Attempted to source `single` faces from LFW (official `vis-www.cs.umass.edu` host):
```bash
    mkdir -p data/single
    cd data/single
    curl -O http://vis-www.cs.umass.edu/lfw/lfw.tgz
```
    — server was unresponsive, abandoned this approach.
  - Sourced `single` faces from the `logasja/lfw` dataset on Hugging Face instead, via the `datasets` library:
```bash
    pip install datasets
    python -c "
    from datasets import load_dataset
    ds = load_dataset('logasja/lfw', split='train')
    for i in range(10):
        ds[i]['image'].save(f'data/single/lfw_{i:02d}.jpg')
    "
```
  - Used stock portrait/group photos from Unsplash and Pexels for `multiple` and `profile` (sunglasses, hats, side angles, partial hand occlusion) — downloaded manually via browser.
- Final image count check across categories:
```bash
    python -c "
    from pathlib import Path
    for folder in Path('data').iterdir():
        if folder.is_dir():
            imgs = list(folder.glob('*.jp*g')) + list(folder.glob('*.png'))
            print(folder.name, len(imgs))
    "
```

## Results obtained
- All smoke tests and edge-case tests passed. Script reliably produces annotated images and a valid `predictions.json` with `image`, `category` (inferred from subfolder name), `prompt`, thresholds, `num_boxes`, `phrases`, `scores`, and `inference_time_sec` per entry.
- Script is ready to run against the curated set.
- Three of five image categories are curated and organized (`single`, `multiple`, `profile`). The remaining categories, deepfake dataset frames, are not yet available — see note below.

## Challenges Encountered
| # | Challenge | Fix |
|---|-----------|-----|
| 1 | `TypeError: load_model() got an unexpected keyword argument 'cpu_only'` | Assumed a `cpu_only` param by analogy to the CLI demo flag, but the actual function signature differs. Checked with `inspect.signature(load_model)` and switched to the correct `device` parameter. |
| 2 | `AssertionError: Torch not compiled with CUDA enabled` inside `predict()`, even after fixing `load_model` | `predict()` has its own independent `device` parameter defaulting to `"cuda"`, unrelated to the device passed into `load_model()`. Fixed by explicitly passing `device="cpu"` into the `predict()` call as well. Grepped `inference.py` for all `device` occurrences to confirm no other function had the same silent default. |
| 3 | LFW official host (`vis-www.cs.umass.edu`) unresponsive when attempting download | Used Unsplash/Pexels stock photos as an alternate source for the affected categories instead of pursuing an LFW mirror, since only a small number of images were needed. |

## Planned next steps
- Run the tested script across the four available categories with the five specified prompts and log per-category observations for Task C.
- Begin threshold sweep (Task D) on a representative subset of the available images.
- Follow up on FaceForensics++ access requests; incorporate deepfake frames once available.