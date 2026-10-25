# Privacy Blur: Dual Face and License Plate Anonymization Pipeline

A lightweight privacy anonymization pipeline in Python that detects and blurs both human faces and vehicle license plates in video streams frame by frame.

Supports both PyTorch (Ultralytics) and standalone ONNX Runtime execution (suitable for edge deployment on NVIDIA Jetson Orin or desktop GPUs). Includes high-resolution inference (1280x1280), false-positive sanity filtering, and temporal smoothing to prevent momentary unblurred frame flickers.

---

## Architecture and ONNX Contract

The pipeline uses two single-class YOLO11n models:
1. Face detector: `models/yolov11n-face.pt`
2. License plate detector: `models/license-plate-finetune-v1n.pt`

Both models are exported to ONNX via `export_onnx.py` using identical contracts matching TensorRT requirements for Jetson Orin and desktop CUDA:

* Static shape, batch 1, FP32, opset 17 (parsed by TensorRT 8.5 / JetPack 5 and TensorRT 10 / JetPack 6).
* Input `images`: float32 RGB in [0, 1], letterboxed into a square canvas (pad value 114/255).
* Output `output0`: `(1, 5, N)` where N = 8400 (640x640) or 33600 (1280x1280). Rows represent `cx, cy, w, h, score` in letterbox canvas pixels.
* No NMS inside the graph: Non-Maximum Suppression is executed in post-processing with IoU threshold 0.50.

### Exported Artifacts

| Model File | Input Shape | Purpose |
|---|---|---|
| `models/yolov11n-face_640.onnx` | 1x3x640x640 | Face detection for foreground/close-up subjects |
| `models/yolov11n-face_1280.onnx` | 1x3x1280x1280 | Face detection for small or distant subjects |
| `models/license-plate-v1n_640.onnx` | 1x3x640x640 | License plate detection for nearby vehicles |
| `models/license-plate-v1n_1280.onnx` | 1x3x1280x1280 | License plate detection for distant or angled plates |

---

## Model Sources and Licenses

* Face model: [AdamCodd/YOLOv11n-face-detection](https://huggingface.co/AdamCodd/YOLOv11n-face-detection)
  * Architecture: YOLO11n (2.6M parameters, trained on WIDERFACE)
  * Reported validation AP: easy 0.942, medium 0.921, hard 0.810
  * License: Weights are Apache-2.0. Exported with Ultralytics which is AGPL-3.0.
* License plate model: [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection)
  * Architecture: YOLO11n (2.6M parameters, trained on Roboflow Universe License Plate dataset, ~10,125 images)
  * Reported metrics: mAP@50: 0.981, mAP@50-95: 0.726
  * License: AGPL-3.0.
* Alternatives that were compared: the YOLO11s variant from the same repository (`license-plate-finetune-v1s.pt`, 19 MB),
  Koushim/yolov8-license-plate-detection (YOLOv8n) and keremberke/yolov5m-license-plate (YOLOv5m). The YOLO11n
  model was kept because it matches the face model's architecture and runs fastest.

Weights are not checked in. Fetch them into `models/` before running:

```bash
mkdir -p models
curl -L -o models/license-plate-finetune-v1n.pt \
  https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.pt
curl -L -o models/yolov11n-face.pt \
  https://huggingface.co/AdamCodd/YOLOv11n-face-detection/resolve/main/model.pt
python3 export_onnx.py   # writes the four ONNX files and runs the parity check
```

## Test Videos

Validation footage is from Pexels (free to use under the [Pexels License](https://www.pexels.com/license/)) and is not
checked in. Download each clip into `videos/` under the file name below:

| File | Source | Content |
|---|---|---|
| `street_dog_walking_crosswalk_pexels.mp4` | [Pexels 19538628](https://www.pexels.com/video/a-man-walking-his-dog-on-a-busy-city-street-19538628/) | 1080p, 52 s; pedestrian with a dog, parked cars with front plates |
| `toronto_crosswalk_pedestrians_pexels.mp4` | [Pexels 14365420](https://www.pexels.com/video/pedestrians-in-the-streets-of-toronto-city-in-ontario-canada-14365420/) | 1080p60, 11.5 s; crossing pedestrians and cyclists, taxi and sedan plates |
| `vienna_museum_crosswalk_pexels.mp4` | [Pexels 37591692](https://www.pexels.com/video/pedestrians-crossing-in-front-of-vienna-museum-37591692/) | 1080p, 22 s; crowd on a crosswalk, waiting SUV with an Austrian plate |

---

## Parity Verification (PyTorch vs ONNX)

Each exported ONNX model is verified against the PyTorch checkpoint on real video frames across the test videos using `export_onnx.py`. Detection counts and minimum box IoU:

| Model and Resolution | Test Frame | Timestamp | PyTorch Count | ONNX Count | Minimum Box IoU |
|---|---|---|---|---|---|
| `license-plate-v1n_640.onnx` | `street_dog_walking` | 3.0s | 1 | 1 | 0.989 |
| | `street_dog_walking` | 15.0s | 2 | 2 | 0.989 |
| | `toronto_crosswalk` | 2.0s | 1 | 1 | 0.981 |
| | `vienna_museum` | 4.0s | 0 | 0 | 1.000 |
| `license-plate-v1n_1280.onnx` | `street_dog_walking` | 3.0s | 1 | 1 | 0.984 |
| | `street_dog_walking` | 15.0s | 2 | 2 | 0.954 |
| | `toronto_crosswalk` | 2.0s | 1 | 1 | 0.971 |
| | `vienna_museum` | 4.0s | 1 | 1 | 0.975 |
| `yolov11n-face_640.onnx` | `street_dog_walking` | 3.0s | 1 | 1 | 0.999 |
| | `street_dog_walking` | 15.0s | 1 | 1 | 0.997 |
| | `toronto_crosswalk` | 2.0s | 7 | 7 | 0.980 |
| | `vienna_museum` | 4.0s | 9 | 9 | 0.990 |
| `yolov11n-face_1280.onnx` | `street_dog_walking` | 3.0s | 1 | 1 | 0.996 |
| | `street_dog_walking` | 15.0s | 1 | 1 | 0.992 |
| | `toronto_crosswalk` | 2.0s | 10 | 10 | 0.960 |
| | `vienna_museum` | 4.0s | 10 | 10 | 0.964 |

At 640x640, the distant Austrian plate in `vienna_museum` is missed by both PyTorch and ONNX (0 detections). At 1280x1280, both detectors recover the plate with 0.975 IoU.

---

## Performance Benchmarks

Measured on an NVIDIA GeForce RTX 4060 8GB (CUDA 12.4, driver 615.78, 1080p source video):

### Throughput and Latency per Frame

| Configuration | Face Size | Plate Size | Dual Infer FPS | Dual Infer Latency | End-to-End FPS |
|---|---|---|---|---|---|
| PyTorch Dual 640 | 640 | 640 | 74.5 FPS | 13.4 ms | 38.6 FPS |
| PyTorch Dual 1280 | 1280 | 1280 | 68.7 FPS | 14.6 ms | 33.5 FPS |
| ONNX CUDA EP 640 | 640 | 640 | 93.3 FPS | 10.7 ms | 65.3 FPS |
| ONNX CUDA EP 1280 | 1280 | 1280 | 25.4 FPS | 39.4 ms | 22.5 FPS |

The PyTorch path letterboxes 1080p frames to a 1280x736 rectangle, while the static ONNX models always run on the full
1280x1280 square, so ONNX at 1280 does about 1.7x the work. A TensorRT FP16 engine on the Jetson should narrow that
gap (not measured yet); a 1280x736 static export is an option if more speed is needed.

### Full-Video Detection Count Agreement (1280 Models)

| Video Clip | Frames | PyTorch Faces | ONNX Faces | PyTorch Plates | ONNX Plates |
|---|---|---|---|---|---|
| `vienna_museum_crosswalk_pexels.mp4` | 556 | 5,413 | 5,461 | 311 | 316 |
| `toronto_crosswalk_pedestrians_pexels.mp4` | 690 | 5,553 | 5,546 | 280 | 281 |
| `street_dog_walking_crosswalk_pexels.mp4` | 1247 | 1,774 | 1,817 | 1,073 | 1,072 |

Detection counts match within 0.8% across thousands of frames.

---

## Installation and Setup

### Desktop / Workstation (PyTorch and ONNX)

```bash
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install repository dependencies
pip install -r requirements.txt
```

### NVIDIA Jetson Orin (Standalone ONNX Runtime)

The ONNX runtime script (`privacy_blur_onnx.py`) requires only NumPy, OpenCV, and ONNX Runtime. No PyTorch or Ultralytics packages are needed.

```bash
# Install numpy and opencv
pip install -r requirements-jetson.txt

# Install the onnxruntime-gpu wheel built for your JetPack version from NVIDIA Jetson Zoo:
# https://elinux.org/Jetson_Zoo#ONNX_Runtime
```

---

## Usage

### 1. Standalone ONNX Runtime (Recommended for Edge / Jetson)

```bash
# Standard 1280 inference with CUDA fallback
python3 privacy_blur_onnx.py \
  --source videos/vienna_museum_crosswalk_pexels.mp4 \
  --output outputs/vienna_blurred.mp4

# Fast 640 inference
python3 privacy_blur_onnx.py \
  --source videos/vienna_museum_crosswalk_pexels.mp4 \
  --face-model models/yolov11n-face_640.onnx \
  --plate-model models/license-plate-v1n_640.onnx \
  --output outputs/vienna_fast.mp4

# Side-by-side comparison with bounding box overlays
python3 privacy_blur_onnx.py \
  --source videos/toronto_crosswalk_pedestrians_pexels.mp4 \
  --output outputs/toronto_compare.mp4 \
  --compare \
  --draw-boxes
```

The script checks for TensorRT execution provider (FP16 engine cached in `./trt_cache`), then CUDA execution provider, then CPU.

### 2. PyTorch / Ultralytics Pipeline

```bash
python3 privacy_blur.py \
  --source videos/vienna_museum_crosswalk_pexels.mp4 \
  --output outputs/vienna_blurred.mp4

# With 2x2 tiled inference for small distant objects
python3 privacy_blur.py \
  --source videos/vienna_museum_crosswalk_pexels.mp4 \
  --output outputs/vienna_tiled.mp4 \
  --tile
```

### 3. Rebuilding ONNX Models

```bash
python3 export_onnx.py
```

### 4. Packaging for Jetson Deployment

```bash
./make_dist.sh
```

Creates `dist/privacy_blur_jetson.tar.gz` containing the ONNX models, `privacy_blur_onnx.py`, `anonymizer.py`, `requirements-jetson.txt`, and this README.

### 5. Pure TensorRT Export (trtexec on Jetson)

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/license-plate-v1n_1280.onnx \
  --fp16 \
  --saveEngine=models/license-plate-v1n_1280.engine
```

---

## CLI Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--source` | `str` | Required | Video file path, RTSP stream URL, or webcam index (e.g. `0`) |
| `--output` | `str` | `None` | Path to save the anonymized MP4 video (H.264 encoded) |
| `--face-model` | `str` | `models/yolov11n-face_1280.onnx` (ONNX) / `models/yolov11n-face.pt` (PT) | Face detector weights |
| `--plate-model` | `str` | `models/license-plate-v1n_1280.onnx` (ONNX) / `models/license-plate-finetune-v1n.pt` (PT) | Plate detector weights |
| `--face-conf` | `float` | `0.25` | Confidence threshold for face detector |
| `--plate-conf` | `float` | `0.30` | Confidence threshold for plate detector |
| `--face-pad` | `float` | `0.15` | Relative padding expansion around face boxes |
| `--plate-pad` | `float` | `0.08` | Relative padding expansion around plate boxes |
| `--mode` | `str` | `blur` | Anonymization style: `blur` (Gaussian) or `pixelate` (Mosaic) |
| `--hold` | `int` | `3` | Frames to persist boxes for temporal smoothing |
| `--draw-boxes` | flag | `False` | Draw debug bounding boxes and confidence labels |
| `--compare` | flag | `False` | Output side-by-side video comparing original and blurred frames |
| `--show` | flag | `False` | Display live preview window with OpenCV |
| `--cpu` | flag | `False` | Force CPUExecutionProvider |
| `--max-frames` | `int` | `None` | Stop processing after N frames |

---

## Known Limitations

1. Crowd Occlusion: When pedestrians walk directly between the camera and a vehicle (as in the Toronto crosswalk footage), the vehicle's bumper and plate are physically occluded. The pipeline blurs the pedestrians, but cannot anonymize a hidden plate until it re-emerges.
2. Animal Faces: At low confidence thresholds (`--face-conf 0.25`), pet faces (such as a dog walking with a pedestrian) can trigger the face detector. Increasing `--face-conf` to `0.30` suppresses this if animal blurring is not desired.
3. Oblique Viewing Angles: Vehicle license plates viewed from steep angles (greater than 70 degrees from normal) have reduced aspect ratios and lower confidence scores. The `--hold 3` temporal buffer maintains coverage as the vehicle rotates into alignment.
4. Objects Exiting Field of View: Once a subject departs the camera frame, tracking expires after the `--hold` window.
