"""
Thin, reusable wrapper around Grounding DINO for face/region localization.

This is the "stage 1" of the planned pipeline:

    image --> GroundingDinoDetector.detect() --> face crops --> forensic classifier

Scripts (batch_inference.py, threshold_sweep.py) and any future stage-2 code
should import from here instead of calling groundingdino.util.inference directly,
so model loading, device handling and box-format conversion live in one place.

Box formats
-----------
Grounding DINO's predict() returns boxes as *normalized* (cx, cy, w, h) in [0, 1].
Detection.box_cxcywh_norm keeps that raw output; Detection.box_xyxy converts it to
absolute pixel (x1, y1, x2, y2), which is what cropping / downstream models need.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DEFAULT_CHECKPOINT = REPO_ROOT / "weights/groundingdino_swint_ogc.pth"


def resolve_device(requested: str = "auto") -> str:
    """Pick a torch device string.

    "auto" -> cuda if available, else cpu. MPS is deliberately *not* chosen
    automatically: Grounding DINO's MPS path has known op gaps (torch.roll,
    int64 cumsum), see Day02 log. Pass "mps" explicitly to try it anyway.
    """
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS is not available")
        # Let unsupported ops fall back to CPU instead of crashing.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return requested


@dataclass
class Detection:
    box_xyxy: List[float]          # absolute pixels: x1, y1, x2, y2
    box_cxcywh_norm: List[float]   # raw model output, normalized cx, cy, w, h
    score: float
    phrase: str


@dataclass
class DetectionResult:
    image_path: str
    image_width: int
    image_height: int
    prompt: str
    box_threshold: float
    text_threshold: float
    inference_time_sec: float
    detections: List[Detection] = field(default_factory=list)

    @property
    def num_boxes(self) -> int:
        return len(self.detections)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["num_boxes"] = self.num_boxes
        return d


def cxcywh_norm_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """(N, 4) normalized cx,cy,w,h -> (N, 4) pixel x1,y1,x2,y2, clipped to the image."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    cx, cy, w, h = boxes[:, 0] * width, boxes[:, 1] * height, boxes[:, 2] * width, boxes[:, 3] * height
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)
    return xyxy


def crop(image_rgb: np.ndarray, box_xyxy, pad: float = 0.0) -> np.ndarray:
    """Crop a detection out of an HxWx3 image, optionally padding by a fraction of box size.

    Forensic classifiers usually want some context around the face, so pad≈0.2–0.3
    is a sensible starting point for stage 2.
    """
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(round(x1 - px))), max(0, int(round(y1 - py)))
    x2, y2 = min(w, int(round(x2 + px))), min(h, int(round(y2 + py)))
    return image_rgb[y1:y2, x1:x2]


class GroundingDinoDetector:
    """Load Grounding DINO once, then call detect() per image."""

    def __init__(
        self,
        config: str | Path = DEFAULT_CONFIG,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        device: str = "auto",
    ):
        from groundingdino.util.inference import load_model

        self.device = resolve_device(device)
        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint}. See README 'Installation' "
                "for the download command."
            )
        # NB: load_model() and predict() each take their own `device` argument and
        # predict() silently defaults to "cuda" -- always pass it to both (Day04 log).
        self.model = load_model(str(config), str(checkpoint), device=self.device)

    def detect(
        self,
        image_path: str | Path,
        prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        """Run one image. Returns (DetectionResult, image_rgb, raw) where raw is
        (boxes, logits, phrases) in the format groundingdino's annotate() expects."""
        from groundingdino.util.inference import load_image, predict

        image_rgb, image_tensor = load_image(str(image_path))
        height, width = image_rgb.shape[:2]

        t0 = time.perf_counter()
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        elapsed = time.perf_counter() - t0  # model forward + post-processing only

        raw_boxes = boxes.cpu().numpy() if len(boxes) else np.zeros((0, 4))
        xyxy = cxcywh_norm_to_xyxy(raw_boxes, width, height)
        detections = [
            Detection(
                box_xyxy=[round(float(v), 1) for v in xyxy[i]],
                box_cxcywh_norm=[round(float(v), 5) for v in raw_boxes[i]],
                score=round(float(logits[i]), 3),
                phrase=phrases[i],
            )
            for i in range(len(phrases))
        ]
        result = DetectionResult(
            image_path=str(image_path),
            image_width=width,
            image_height=height,
            prompt=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            inference_time_sec=round(elapsed, 3),
            detections=detections,
        )
        return result, image_rgb, (boxes, logits, phrases)
