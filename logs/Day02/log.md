# Day 02 — Wednesday, 7/29

**Focus:** Environment setup

## To-Do
- [ ✅ ] Install/Clone dependancies (Grounding DINO, Python, PyTorch, GPU, etc.)
    - Notes:
        ```bash
            xcode-select --install
            brew install miniforge
            conda create -n groundingdino python=3.10 -y
            conda init zsh
            source ~/.zshrc
            conda activate groundingdino
            pip install torch torchvision
            git clone https://github.com/IDEA-Research/GroundingDINO.git
            cd GroundingDINO/
            pip install -e . --no-build-isolation
            pip install transformers==4.37
        ```
- [ ✅ ] Record versions used
    - Notes:
        * Python: 3.10
        * PyTorch: 2.13.0
        * transformers: 4.37 (pinned down from whatever newer default pip installed)
        * CUDA: N/A (macOS, no NVIDIA GPU)
        * GPU/chip: Apple M2
        * MPS available: True (but ran inference in --cpu-only mode for reliability/accuracy
        * OS: macOS 26.6
- [ ✅ ] Document each error + fix
- [ ✅ ] Download pretrained Grounding DINO Swin-T checkpoint
    - Notes:
    ```bash
        mkdir weights && cd weights
        curl -L -o groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
        cd ..
    ```
- [ ✅ ] PyTorch Section 2
    - Notes: Parts 14-32

## Challenges Encountered
| # | Challenge | Fix |
|---|-------|-----|
| 1 | No NVIDIA GPU/CUDA toolkit available on macOS (Apple Silicon has no CUDA support) | Used PyTorch's built-in MPS (Metal) backend instead of CUDA — confirmed with `torch.backends.mps.is_available()` → `True`. However, ran inference with the `--cpu-only` flag rather than MPS, since GroundingDINO's MPS path has known issues (`torch.roll` unsupported without `PYTORCH_ENABLE_MPS_FALLBACK=1`, and `cumsum` doesn't support int64 on MPS, affecting accuracy) and CPU mode benchmarked as fast or faster in community tests |
| 2 | `CondaError: Run 'conda init' before 'conda activate'` | Ran `conda init zsh`, then `source ~/.zshrc` to reload the shell before retrying `conda activate groundingdino` |
| 3 | `pip install -e .` failed during build isolation (`ModuleNotFoundError: No module named 'torch'` / `No module named pip` inside pip's temp build env) | Ran `pip install -e . --no-build-isolation` so the build uses the already-installed torch in the active conda env instead of a fresh isolated one |
| 4 | `argparse error: the following arguments are required: --output_dir/-o` | Caused by literally including `< >` placeholder brackets around the image path; bash interpreted them as shell input/output redirection instead of passing them as part of the argument. Fixed by passing the raw path with no brackets |
| 5 | `AttributeError: 'BertModel' object has no attribute 'get_head_mask'` | Version mismatch — a newer `transformers` release removed a method GroundingDINO's code depends on. Fixed with `pip install transformers==4.37` |