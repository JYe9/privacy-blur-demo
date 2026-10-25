#!/usr/bin/env python3
"""Export face and license plate detectors to ONNX for Jetson Orin and desktop.

Exports:
  models/yolov11n-face_{640,1280}.onnx
  models/license-plate-v1n_{640,1280}.onnx

Contract: static 1x3xNxN, batch 1, FP32, opset 17, simplified, no NMS in graph.
Input: 'images' float32 RGB in [0, 1] letterboxed (pad 114).
Output: 'output0' (1, 5, N) = cx, cy, w, h, score in letterboxed canvas pixels.

Each model is parity-checked against the PyTorch checkpoint on real video frames.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from anonymizer import matrix_iou
from privacy_blur_onnx import ONNXDetector

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
VIDEOS_DIR = ROOT / "videos"
SIZES = (640, 1280)

# Real frames with both faces and license plates
CHECK_FRAMES = [
    ("street_dog_walking_crosswalk_pexels.mp4", 3.0),
    ("street_dog_walking_crosswalk_pexels.mp4", 15.0),
    ("toronto_crosswalk_pedestrians_pexels.mp4", 2.0),
    ("vienna_museum_crosswalk_pexels.mp4", 4.0),
]


def extract_frame(video_rel: str, timestamp_s: float) -> np.ndarray:
    video_path = VIDEOS_DIR / video_rel
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = round(timestamp_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame at {timestamp_s}s from {video_path}")
    return frame


def check_parity(
    pt_model: YOLO,
    onnx_path: Path,
    cls_name: str,
    conf: float,
    size: int,
) -> None:
    det = ONNXDetector(str(onnx_path), cls_name=cls_name, conf=conf, providers=["CPUExecutionProvider"])
    print(f"Parity check: {onnx_path.name} (conf={conf}, size={size})")
    for vname, t_sec in CHECK_FRAMES:
        frame = extract_frame(vname, t_sec)
        ref_res = pt_model(frame, conf=conf, imgsz=size, verbose=False)[0].boxes
        ref_boxes = ref_res.xyxy.cpu().numpy()

        onnx_boxes, _ = det(frame, apply_sanity=False)

        if len(ref_boxes) > 0 and len(onnx_boxes) > 0:
            ious = matrix_iou(ref_boxes, onnx_boxes)
            best_iou = ious.max(axis=1)
            min_iou = float(best_iou.min())
        elif len(ref_boxes) == 0 and len(onnx_boxes) == 0:
            min_iou = 1.0
        else:
            min_iou = 0.0

        print(
            f"  {vname:42s} t={t_sec:4.1f}s | "
            f"pytorch={len(ref_boxes):2d}  onnx={len(onnx_boxes):2d}  min IoU={min_iou:.3f}"
        )


def export_model(
    pt_path: Path,
    onnx_template: str,
    cls_name: str,
    conf: float,
) -> None:
    pt = YOLO(str(pt_path))
    for size in SIZES:
        out_path = MODELS_DIR / onnx_template.format(size=size)
        print(f"\nExporting {pt_path.name} -> {out_path.name} (imgsz={size}, opset=17)...")
        exported = Path(
            pt.export(
                format="onnx",
                imgsz=size,
                opset=17,
                simplify=True,
                dynamic=False,
                half=False,
                batch=1,
            )
        )
        if exported != out_path:
            exported.replace(out_path)

        size_mb = out_path.stat().st_size / 1e6
        print(f"Wrote {out_path} ({size_mb:.1f} MB)")
        check_parity(pt, out_path, cls_name=cls_name, conf=conf, size=size)


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    # 1. License plate model
    plate_pt = MODELS_DIR / "license-plate-finetune-v1n.pt"
    export_model(
        pt_path=plate_pt,
        onnx_template="license-plate-v1n_{size}.onnx",
        cls_name="plate",
        conf=0.30,
    )

    # 2. Face model
    face_pt = MODELS_DIR / "yolov11n-face.pt"
    export_model(
        pt_path=face_pt,
        onnx_template="yolov11n-face_{size}.onnx",
        cls_name="face",
        conf=0.25,
    )


if __name__ == "__main__":
    main()
