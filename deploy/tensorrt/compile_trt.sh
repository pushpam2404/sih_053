#!/bin/bash
# DRDO ID26053 — Phase 6: TensorRT Engine Compilation
# Run ONLY on NVIDIA Jetson AGX Orin (JetPack 5.x, CUDA 12.x)
# Required: trtexec is at /usr/src/tensorrt/bin/trtexec (standard JetPack path)

set -e

TRTEXEC=/usr/src/tensorrt/bin/trtexec
ONNX_PATH=/workspace/sih/phase6/tensorrt/minkunet18_drdo.onnx
ENGINE_PATH=/workspace/sih/phase6/tensorrt/minkunet18_drdo_fp16.engine

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
echo "[STEP P6.2.2a COMPLETE (on Jetson)]"
