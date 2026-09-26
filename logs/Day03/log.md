# Day 03 — Friday, 7/31

**Focus:** Finish setup + run official inference

## To-Do
- [ ✅ ] Successfully run the official image-inference example on a sample image
    - Notes:
```bash
        python demo/inference_on_a_image.py \
          -c groundingdino/config/GroundingDINO_SwinT_OGC.py \
          -p weights/groundingdino_swint_ogc.pth \
          -i ../logs/Day03/sample_image_input.jpeg \
          -o output_dir/ \
          -t "person" \
          --cpu-only
```
- [ ✅ ] PyTorch course: complete Section 2
    - Notes: Parts 14-39

## Experiments conducted
- Ran the official Grounding DINO inference demo on a single sample image (`sample_image_raw.jpeg`) using the Swin-T checkpoint, in `--cpu-only` mode, with the text prompt `"person"` and default box/text thresholds (0.3 / 0.25, since none were explicitly set).

## Results obtained
- Inference completed successfully end-to-end, producing an annotated output image in `output_dir/`, confirming the environment, checkpoint, and pipeline are all working correctly on my machine.

## Challenges Encountered
| # | Challenge | Fix |
|---|-----------|-----|
| 1 | Inference script errored with `argparse error: the following arguments are required: --output_dir/-o` | Caused by literally including `< >` placeholder brackets around the image path; bash interpreted them as shell input/output redirection instead of passing them as part of the argument. Fixed by passing the raw path with no brackets |
| 2 | `AttributeError: 'BertModel' object has no attribute 'get_head_mask'` when loading the model | Version mismatch — a newer `transformers` release removed a method GroundingDINO's code depends on. Fixed with `pip install transformers==4.37` |