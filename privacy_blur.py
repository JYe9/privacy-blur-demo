#!/usr/bin/env python3
"""Dual face and vehicle license plate anonymization pipeline using PyTorch and Ultralytics.

Usage:
  python3 privacy_blur.py --source in.mp4 --output out.mp4
  python3 privacy_blur.py --source 0 --show
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

from anonymizer import PrivacyBlurrer, filter_box_sanity, transcode_to_h264


class PyTorchPrivacyDetector:
    def __init__(
        self,
        face_model_path: str = "models/yolov11n-face.pt",
        plate_model_path: str = "models/license-plate-finetune-v1n.pt",
        face_conf: float = 0.25,
        plate_conf: float = 0.30,
        face_imgsz: int = 1280,
        plate_imgsz: int = 1280,
        tile: bool = False,
        device: str = "cuda:0",
    ):
        from ultralytics import YOLO

        self.face_model = YOLO(face_model_path)
        self.plate_model = YOLO(plate_model_path)
        self.face_conf = face_conf
        self.plate_conf = plate_conf
        self.face_imgsz = face_imgsz
        self.plate_imgsz = plate_imgsz
        self.tile = tile
        self.device = device

    def _run_single(
        self,
        model,
        frame: np.ndarray,
        imgsz: int,
        conf: float,
        cls_name: str,
    ) -> list[tuple[np.ndarray, float]]:
        h, w = frame.shape[:2]
        all_boxes: list[np.ndarray] = []
        all_scores: list[float] = []

        res_full = model(frame, imgsz=imgsz, conf=conf, device=self.device, verbose=False)[0]
        if res_full.boxes is not None and len(res_full.boxes) > 0:
            boxes = res_full.boxes.xyxy.cpu().numpy()
            scores = res_full.boxes.conf.cpu().numpy()
            for b, s in zip(boxes, scores):
                all_boxes.append(b)
                all_scores.append(float(s))

        if self.tile:
            x_mid, y_mid = w // 2, h // 2
            pad_x, pad_y = int(w * 0.10), int(h * 0.10)
            tile_coords = [
                (0, 0, x_mid + pad_x, y_mid + pad_y),
                (x_mid - pad_x, 0, w, y_mid + pad_y),
                (0, y_mid - pad_y, x_mid + pad_x, h),
                (x_mid - pad_x, y_mid - pad_y, w, h),
            ]
            tiles = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in tile_coords]
            tile_results = model(tiles, imgsz=min(640, imgsz), conf=conf, device=self.device, verbose=False)
            for (x1_off, y1_off, _, _), t_res in zip(tile_coords, tile_results):
                if t_res.boxes is not None and len(t_res.boxes) > 0:
                    t_boxes = t_res.boxes.xyxy.cpu().numpy()
                    t_scores = t_res.boxes.conf.cpu().numpy()
                    for tb, ts in zip(t_boxes, t_scores):
                        all_boxes.append(
                            np.array(
                                [
                                    tb[0] + x1_off,
                                    tb[1] + y1_off,
                                    tb[2] + x1_off,
                                    tb[3] + y1_off,
                                ],
                                dtype=np.float32,
                            )
                        )
                        all_scores.append(float(ts))

        if not all_boxes:
            return []

        all_boxes_np = np.array(all_boxes, dtype=np.float32)
        all_scores_np = np.array(all_scores, dtype=np.float32)

        xywh = np.c_[all_boxes_np[:, :2], all_boxes_np[:, 2:] - all_boxes_np[:, :2]].tolist()
        indices = cv2.dnn.NMSBoxes(xywh, all_scores_np.tolist(), conf, 0.50)
        if len(indices) == 0:
            return []

        indices = np.array(indices, dtype=int).reshape(-1)

        valid_items: list[tuple[np.ndarray, float]] = []
        for idx in indices:
            box = all_boxes_np[idx]
            score = float(all_scores_np[idx])
            if filter_box_sanity(box, cls_name, frame.shape):
                valid_items.append((box, score))

        return valid_items

    def detect(self, frame: np.ndarray) -> tuple[list[tuple[np.ndarray, float]], list[tuple[np.ndarray, float]]]:
        faces = self._run_single(
            self.face_model,
            frame,
            imgsz=self.face_imgsz,
            conf=self.face_conf,
            cls_name="face",
        )
        plates = self._run_single(
            self.plate_model,
            frame,
            imgsz=self.plate_imgsz,
            conf=self.plate_conf,
            cls_name="plate",
        )
        return faces, plates


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, RTSP URL, or webcam index")
    ap.add_argument("--output", default=None, help="save anonymized MP4 video")
    ap.add_argument("--face-model", default="models/yolov11n-face.pt", help="face detector weights (.pt)")
    ap.add_argument("--plate-model", default="models/license-plate-finetune-v1n.pt", help="plate detector weights (.pt)")
    ap.add_argument("--face-conf", type=float, default=0.25, help="confidence threshold for face detector")
    ap.add_argument("--plate-conf", type=float, default=0.30, help="confidence threshold for plate detector")
    ap.add_argument("--face-imgsz", type=int, default=1280, help="input size for face detector")
    ap.add_argument("--plate-imgsz", type=int, default=1280, help="input size for plate detector")
    ap.add_argument("--imgsz", type=int, default=None, help="global input size override")
    ap.add_argument("--tile", action="store_true", help="enable 2x2 batched overlapping tiled inference")
    ap.add_argument("--face-pad", type=float, default=0.15, help="relative padding around face boxes")
    ap.add_argument("--plate-pad", type=float, default=0.08, help="relative padding around plate boxes")
    ap.add_argument("--pad", type=float, default=None, help="global padding override")
    ap.add_argument("--mode", choices=["blur", "pixelate"], default="blur", help="anonymization technique")
    ap.add_argument("--device", default="cuda:0", help="inference device ('cuda:0' or 'cpu')")
    ap.add_argument("--hold", type=int, default=3, help="frames to persist boxes for temporal smoothing")
    ap.add_argument("--draw-boxes", action="store_true", help="draw bounding boxes and labels")
    ap.add_argument("--compare", action="store_true", help="render side-by-side original vs anonymized comparison")
    ap.add_argument("--show", action="store_true", help="display preview window")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after processing N frames")
    args = ap.parse_args()

    face_imgsz = args.imgsz if args.imgsz is not None else args.face_imgsz
    plate_imgsz = args.imgsz if args.imgsz is not None else args.plate_imgsz
    face_pad = args.pad if args.pad is not None else args.face_pad
    plate_pad = args.pad if args.pad is not None else args.plate_pad

    detector = PyTorchPrivacyDetector(
        face_model_path=args.face_model,
        plate_model_path=args.plate_model,
        face_conf=args.face_conf,
        plate_conf=args.plate_conf,
        face_imgsz=face_imgsz,
        plate_imgsz=plate_imgsz,
        tile=args.tile,
        device=args.device,
    )

    blur = PrivacyBlurrer(
        mode=args.mode,
        face_pad=face_pad,
        plate_pad=plate_pad,
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

    print(f"face model {args.face_model} ({face_imgsz}x{face_imgsz}, {args.device})")
    print(f"plate model {args.plate_model} ({plate_imgsz}x{plate_imgsz}, {args.device})")
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
            faces, plates = detector.detect(frame)
            if processed > 0:
                t_infer += time.perf_counter() - t0

            blur.update(faces, plates)
            anon = blur.apply(frame)

            combined = blur.render_compare(frame, anon) if args.compare else anon

            if writer:
                writer.write(combined)
            if args.show:
                cv2.imshow("privacy blur", combined)
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
