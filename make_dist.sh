#!/usr/bin/env bash
set -euo pipefail

# Package standalone ONNX runtime and models for Jetson Orin deployment.
DIST_DIR="dist"
ARCHIVE="${DIST_DIR}/privacy_blur_jetson.tar.gz"

mkdir -p "${DIST_DIR}"

echo "Creating ${ARCHIVE}..."
tar -czvf "${ARCHIVE}" \
  models/yolov11n-face_640.onnx \
  models/yolov11n-face_1280.onnx \
  models/license-plate-v1n_640.onnx \
  models/license-plate-v1n_1280.onnx \
  privacy_blur_onnx.py \
  anonymizer.py \
  requirements-jetson.txt \
  README.md

echo "Created ${ARCHIVE} ($(du -h "${ARCHIVE}" | cut -f1))"
