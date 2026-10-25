from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AnonymizedBox:
    box: np.ndarray
    cls_name: str
    conf: float
    ttl: int


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def matrix_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def filter_box_sanity(box: np.ndarray, cls_name: str, frame_shape: tuple[int, int]) -> bool:
    h_img, w_img = frame_shape[:2]
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    if cls_name == "plate":
        if w < 12 or h < 6:
            return False
        if w > w_img * 0.22 or h > h_img * 0.12:
            return False
        ar = w / max(1.0, h)
        return 1.1 <= ar <= 6.0

    if cls_name == "face":
        if w < 8 or h < 8:
            return False
        if w > w_img * 0.45 or h > h_img * 0.55:
            return False
        ar = h / max(1.0, w)
        return 0.65 <= ar <= 2.2

    return True


class PrivacyBlurrer:
    def __init__(
        self,
        mode: str = "blur",
        face_pad: float = 0.15,
        plate_pad: float = 0.08,
        hold: int = 3,
        draw_boxes: bool = False,
    ):
        self.mode = mode
        self.face_pad = face_pad
        self.plate_pad = plate_pad
        self.hold = max(0, hold)
        self.draw_boxes = draw_boxes
        self.active_boxes: list[AnonymizedBox] = []

        self.total_face_detections = 0
        self.total_plate_detections = 0
        self.frames_with_faces = 0
        self.frames_with_plates = 0
        self.total_frames = 0

    def update(
        self,
        faces: list[tuple[np.ndarray, float]],
        plates: list[tuple[np.ndarray, float]],
    ) -> None:
        self.total_frames += 1
        new_items: list[tuple[np.ndarray, str, float]] = []
        for b, c in faces:
            new_items.append((b, "face", c))
        for b, c in plates:
            new_items.append((b, "plate", c))

        if faces:
            self.frames_with_faces += 1
            self.total_face_detections += len(faces)
        if plates:
            self.frames_with_plates += 1
            self.total_plate_detections += len(plates)

        if self.hold == 0:
            self.active_boxes = [
                AnonymizedBox(box=b, cls_name=cls, conf=c, ttl=0)
                for b, cls, c in new_items
            ]
            return

        matched_active = set()
        matched_new = set()

        for n_idx, (new_box, new_cls, new_conf) in enumerate(new_items):
            best_iou = 0.0
            best_a_idx = -1
            for a_idx, active in enumerate(self.active_boxes):
                if a_idx in matched_active or active.cls_name != new_cls:
                    continue
                iou = compute_iou(new_box, active.box)
                if iou > 0.25 and iou > best_iou:
                    best_iou = iou
                    best_a_idx = a_idx

            if best_a_idx >= 0:
                self.active_boxes[best_a_idx].box = new_box
                self.active_boxes[best_a_idx].conf = new_conf
                self.active_boxes[best_a_idx].ttl = self.hold
                matched_active.add(best_a_idx)
                matched_new.add(n_idx)

        surviving: list[AnonymizedBox] = []
        for a_idx, active in enumerate(self.active_boxes):
            if a_idx not in matched_active:
                active.ttl -= 1
            if active.ttl >= 0:
                surviving.append(active)

        for n_idx, (new_box, new_cls, new_conf) in enumerate(new_items):
            if n_idx not in matched_new:
                surviving.append(
                    AnonymizedBox(box=new_box, cls_name=new_cls, conf=new_conf, ttl=self.hold)
                )

        self.active_boxes = surviving

    def apply(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        h, w = frame.shape[:2]

        for item in self.active_boxes:
            x1, y1, x2, y2 = item.box
            pad = self.face_pad if item.cls_name == "face" else self.plate_pad

            bw = x2 - x1
            bh = y2 - y1
            px = bw * pad
            py = bh * pad

            cx1 = int(max(0, np.floor(x1 - px)))
            cy1 = int(max(0, np.floor(y1 - py)))
            cx2 = int(min(w, np.ceil(x2 + px)))
            cy2 = int(min(h, np.ceil(y2 + py)))

            if cx2 - cx1 < 3 or cy2 - cy1 < 3:
                continue

            roi = out[cy1:cy2, cx1:cx2]

            if self.mode == "pixelate":
                rw = cx2 - cx1
                rh = cy2 - cy1
                bw_px = max(1, rw // 10)
                bh_px = max(1, rh // 10)
                small = cv2.resize(roi, (bw_px, bh_px), interpolation=cv2.INTER_LINEAR)
                out[cy1:cy2, cx1:cx2] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
            else:
                k = max(7, (max(cx2 - cx1, cy2 - cy1) // 2) | 1)
                out[cy1:cy2, cx1:cx2] = cv2.GaussianBlur(roi, (k, k), 0)

            if self.draw_boxes:
                color = (0, 255, 0) if item.cls_name == "face" else (0, 215, 255)
                cv2.rectangle(out, (cx1, cy1), (cx2, cy2), color, 2)
                label = f"{item.cls_name} {item.conf:.2f}"
                t_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    out,
                    (cx1, max(0, cy1 - t_size[1] - 4)),
                    (cx1 + t_size[0] + 4, cy1),
                    color,
                    -1,
                )
                cv2.putText(
                    out,
                    label,
                    (cx1 + 2, max(cy1 - 2, t_size[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        return out

    def render_compare(self, orig: np.ndarray, anon: np.ndarray) -> np.ndarray:
        orig_disp = orig.copy()
        cv2.putText(orig_disp, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        anon_disp = anon.copy()
        cv2.putText(anon_disp, f"Anonymized ({self.mode})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        return np.hstack([orig_disp, anon_disp])


def transcode_to_h264(input_path: str, output_path: str) -> bool:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return False

    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel", "error",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "20",
        output_path,
    ]
    res = subprocess.run(cmd, check=False)
    return res.returncode == 0
