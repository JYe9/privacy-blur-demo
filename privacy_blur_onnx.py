#!/usr/bin/env python3
"""Standalone dual face and license plate anonymization with ONNX Runtime.

Runs on x86_64 with CUDA/TensorRT and on Jetson Orin with onnxruntime-gpu.
Only requires numpy, opencv-python and onnxruntime. No PyTorch or Ultralytics needed.

Usage:
  python3 privacy_blur_onnx.py --source in.mp4 --output out.mp4
  python3 privacy_blur_onnx.py --source 0 --show
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import glob
import os
import sys
import time

import cv2
import numpy as np

# Preload pip-installed nvidia CUDA/cuDNN libraries if present
for _sp in sys.path:
    _cudnn_lib = os.path.join(_sp, "nvidia", "cudnn", "lib")
    if os.path.isdir(_cudnn_lib):
        for _f in sorted(glob.glob(os.path.join(_cudnn_lib, "libcudnn*.so*"))):
            try:
                ctypes.CDLL(_f, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass

import onnxruntime as ort

from anonymizer import PrivacyBlurrer, filter_box_sanity, transcode_to_h264


class ONNXDetector:
    def __init__(
        self,
        model: str,
        cls_name: str,
        conf: float = 0.25,
        iou: float = 0.50,
        providers: list[str] | None = None,
    ):
        available = ort.get_available_providers()
        has_trt = bool(ctypes.util.find_library("nvinfer") or os.path.exists("/usr/lib/aarch64-linux-gnu/libnvinfer.so"))
        wanted = providers or (
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            if has_trt
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        opts = {
            "TensorrtExecutionProvider": {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./trt_cache",
            }
        }
        chosen = [(p, opts[p]) if p in opts else p for p in wanted if p in available]
        self.session = ort.InferenceSession(model, providers=chosen)
        if "TensorrtExecutionProvider" in available and \
                "TensorrtExecutionProvider" not in self.session.get_providers():
            chosen = [p for p in chosen if not (isinstance(p, tuple) and p[0] == "TensorrtExecutionProvider")]
            self.session = ort.InferenceSession(model, providers=chosen)

        self.input = self.session.get_inputs()[0]
        self.size = self.input.shape[2]
        self.cls_name = cls_name
        self.conf = conf
        self.iou = iou

    @property
    def provider(self) -> str:
        return self.session.get_providers()[0]

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = frame.shape[:2]
        r = self.size / max(h, w)
        nh, nw = round(h * r), round(w * r)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(blob), r, left, top

    def __call__(self, frame: np.ndarray, apply_sanity: bool = True) -> tuple[np.ndarray, np.ndarray]:
        blob, r, left, top = self.preprocess(frame)
        pred = self.session.run(None, {self.input.name: blob})[0][0]  # (5, N)
        scores = pred[4]
        keep = scores > self.conf
        if not keep.any():
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)

        cx, cy, bw, bh = pred[:4, keep]
        scores = scores[keep]
        boxes = np.stack([cx - bw / 2 - left, cy - bh / 2 - top, cx + bw / 2 - left, cy + bh / 2 - top], 1) / r

        idx = cv2.dnn.NMSBoxes(
            np.c_[boxes[:, :2], boxes[:, 2:] - boxes[:, :2]].tolist(),
            scores.tolist(),
            self.conf,
            self.iou,
        )
        if len(idx) == 0:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)

        idx = np.array(idx, dtype=int).reshape(-1)
        h, w = frame.shape[:2]
        boxes = boxes[idx].clip([0, 0, 0, 0], [w, h, w, h])
        scores = scores[idx]

        if apply_sanity:
            valid = [filter_box_sanity(b, self.cls_name, frame.shape) for b in boxes]
            if len(valid) > 0 and not all(valid):
                valid_arr = np.array(valid, dtype=bool)
                boxes = boxes[valid_arr]
                scores = scores[valid_arr]

        return boxes.astype(np.float32), scores.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, RTSP URL, or webcam index")
    ap.add_argument("--output", help="save anonymized MP4 video")
    ap.add_argument("--face-model", default="models/yolov11n-face_1280.onnx", help="face ONNX model")
    ap.add_argument("--plate-model", default="models/license-plate-v1n_1280.onnx", help="plate ONNX model")
    ap.add_argument("--face-conf", type=float, default=0.25)
    ap.add_argument("--plate-conf", type=float, default=0.30)
    ap.add_argument("--face-pad", type=float, default=0.15)
    ap.add_argument("--plate-pad", type=float, default=0.08)
    ap.add_argument("--mode", choices=["blur", "pixelate"], default="blur")
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--draw-boxes", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="force CPUExecutionProvider")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    providers = ["CPUExecutionProvider"] if args.cpu else None
    face_det = ONNXDetector(args.face_model, cls_name="face", conf=args.face_conf, providers=providers)
    plate_det = ONNXDetector(args.plate_model, cls_name="plate", conf=args.plate_conf, providers=providers)

    blur = PrivacyBlurrer(
        mode=args.mode,
        face_pad=args.face_pad,
        plate_pad=args.plate_pad,
        hold=args.hold,
        draw_boxes=args.draw_boxes,
    )

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"error: failed to open source: {args.source}", file=sys.stderr)
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = width * 2 if args.compare else width
    out_h = height
    writer = None
    temp_output_path = None
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        temp_output_path = args.output + ".temp.mp4"
        writer = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    print(f"face model {args.face_model} ({face_det.size}x{face_det.size}, {face_det.provider})")
    print(f"plate model {args.plate_model} ({plate_det.size}x{plate_det.size}, {plate_det.provider})")
    print(f"source {args.source} ({width}x{height} @ {fps:.1f} fps, {total_src_frames} frames)")

    t_start = time.perf_counter()
    t_infer = 0.0
    processed = 0

    try:
        while True:
            if args.max_frames and processed >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            t0 = time.perf_counter()
            f_boxes, f_scores = face_det(frame)
            p_boxes, p_scores = plate_det(frame)
            if processed > 0:
                t_infer += time.perf_counter() - t0

            faces = [(b, float(s)) for b, s in zip(f_boxes, f_scores)]
            plates = [(b, float(s)) for b, s in zip(p_boxes, p_scores)]
            blur.update(faces, plates)
            anon = blur.apply(frame)

            combined = blur.render_compare(frame, anon) if args.compare else anon

            if writer:
                writer.write(combined)
            if args.show:
                cv2.imshow("privacy blur onnx", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed += 1
            if processed % 30 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"frame {processed}/{total_src_frames} ({processed / elapsed:.1f} FPS)", end="\r")
                sys.stdout.flush()

    finally:
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    t_total = time.perf_counter() - t_start
    overall_fps = processed / t_total if t_total > 0 else 0.0
    infer_fps = (processed - 1) / t_infer if processed > 1 and t_infer > 0 else 0.0

    if args.output and temp_output_path and os.path.exists(temp_output_path):
        if transcode_to_h264(temp_output_path, args.output):
            os.remove(temp_output_path)
        else:
            os.replace(temp_output_path, args.output)

    print(f"\n{processed} frames in {t_total:.2f}s: {overall_fps:.1f} FPS end-to-end, {infer_fps:.1f} FPS dual inference")
    print(f"faces: {blur.total_face_detections} in {blur.frames_with_faces} frames, plates: {blur.total_plate_detections} in {blur.frames_with_plates} frames")


if __name__ == "__main__":
    main()
