#!/bin/bash
# DRDO ID26053 — Phase 6: TensorRT Engine Compilation
# Run ONLY on NVIDIA Jetson AGX Orin (JetPack 5.x ships CUDA 11.4 / TRT 8.5; JetPack 6 ships CUDA 12.x / TRT 8.6+)
# Required: trtexec is at /usr/src/tensorrt/bin/trtexec (standard JetPack path)

set -e

TRTEXEC=/usr/src/tensorrt/bin/trtexec
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ONNX_PATH="${ONNX_PATH:-$REPO_ROOT/models/minkunet18_drdo.onnx}"
ENGINE_PATH="${ENGINE_PATH:-$REPO_ROOT/models/minkunet18_drdo_fp16.engine}"

echo "[TRT] Compiling ONNX -> TensorRT FP16 engine..."
echo "[TRT] ONNX:    $ONNX_PATH"
echo "[TRT] Engine:  $ENGINE_PATH"

$TRTEXEC \
    --onnx=$ONNX_PATH \
    --saveEngine=$ENGINE_PATH \
    --fp16 \
    --minShapes=feats:1x4 \
    --optShapes=feats:65536x4 \
    --maxShapes=feats:131072x4

echo "[TRT] Engine written to: $ENGINE_PATH"
echo "[TRT] Done."
