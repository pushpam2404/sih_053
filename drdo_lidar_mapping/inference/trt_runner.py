#!/usr/bin/env python3
"""
DRDO ID26053 — Phase 6: TensorRT High-Performance Inference Runner
Provides TrtInferenceRunner to execute MinkUNet-18 FP16 TensorRT engines on NVIDIA Jetson AGX Orin.
Includes host-side CPU fallback stub for platforms without TensorRT/CUDA.
"""

import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import time
from typing import Optional
import numpy as np

# Check TensorRT and PyCUDA availability
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


class TrtInferenceRunner:
    """
    Manages loading and executing a serialized TensorRT engine for 3D point cloud semantic segmentation.
    Supports dynamic batch sizes up to 131,072 points per frame.
    """
    MAX_POINTS = 131072   # matches --maxShapes in deploy/tensorrt/compile_trt.sh

    def __init__(self, engine_path: str, allow_stub: bool = False) -> None:
        """
        Initializes the TensorRT inference runner.
        Args:
            engine_path: Filepath to the serialized .engine file.
            allow_stub: Use the z-threshold CPU stub when TensorRT or the engine is missing.
                Default False: previously a missing engine on the vehicle silently produced fake
                "obstacle if z > 1 m" logits.
        """
        self.engine_path: str = os.path.expanduser(engine_path)
        self.num_classes: int = 8
        self.in_channels: int = 4
        self.use_stub: bool = not TRT_AVAILABLE or not os.path.exists(self.engine_path)

        if not self.use_stub:
            print(f"[TRT] Loading TensorRT engine from: {self.engine_path}")
            self.logger = trt.Logger(trt.Logger.WARNING)
            with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError(f"TensorRT failed to deserialize {self.engine_path} (built for another TRT/GPU?)")
            self.context = self.engine.create_execution_context()
            # Allocate once for the max profile. Per-frame cuda.mem_alloc + pageable numpy copies
            # fragment memory and add avoidable latency; page-locked buffers make the async copies real.
            self.stream = cuda.Stream()
            self.h_input = cuda.pagelocked_empty((self.MAX_POINTS, self.in_channels), dtype=np.float32)
            self.h_output = cuda.pagelocked_empty((self.MAX_POINTS, self.num_classes), dtype=np.float32)
            self.d_input = cuda.mem_alloc(self.h_input.nbytes)
            self.d_output = cuda.mem_alloc(self.h_output.nbytes)
            print("[TRT] Engine deserialized successfully.")
        elif allow_stub:
            print("[WARN] TensorRT engine unavailable — using CPU STUB (fake logits, not a model)")
        else:
            reason = "tensorrt/pycuda not importable" if not TRT_AVAILABLE else f"engine not found: {self.engine_path}"
            raise RuntimeError(f"[TRT] Cannot run inference: {reason}. Pass allow_stub=True for host tests only.")

    def infer(self, feats: np.ndarray) -> np.ndarray:
        """
        Executes semantic segmentation inference on LiDAR point features.
        Args:
            feats: np.ndarray of shape (N, 4) with dtype float32 [x, y, z, intensity]
        Returns:
            logits: np.ndarray of shape (N, 8) with dtype float32
        """
        assert feats.ndim == 2 and feats.shape[1] == self.in_channels, (
            f"Expected input shape (N, {self.in_channels}), got {feats.shape}"
        )
        n_points = feats.shape[0]

        if self.use_stub:
            # Deterministic/fast host fallback for syntactic and functional validation
            logits = np.zeros((n_points, self.num_classes), dtype=np.float32)
            # Default to GROUND (class 0) with high confidence, perturbed by elevation
            logits[:, 0] = 2.0
            logits[:, 4] = np.where(feats[:, 2] > 1.0, 3.5, -1.0)  # Obstacle if z > 1.0m
            return logits

        if n_points > self.MAX_POINTS:
            raise ValueError(f"{n_points} points exceeds the engine max profile ({self.MAX_POINTS})")

        # Native TensorRT execution path on Jetson (TensorRT 8.x binding API: binding 0 = feats,
        # binding 1 = logits). TensorRT 10 removed this API — port to set_input_shape/execute_async_v3.
        self.context.set_binding_shape(0, (n_points, self.in_channels))
        in_view = self.h_input[:n_points]
        out_view = self.h_output[:n_points]
        np.copyto(in_view, feats, casting="same_kind")
        cuda.memcpy_htod_async(self.d_input, in_view, self.stream)
        ok = self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)],
                                           stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(out_view, self.d_output, self.stream)
        self.stream.synchronize()
        if not ok:
            raise RuntimeError("[TRT] execute_async_v2 failed")
        return out_view.copy()


def main() -> None:
    """Benchmark and validate TrtInferenceRunner."""
    default_engine = os.path.join(REPO_ROOT, "models", "minkunet18_drdo_fp16.engine")
    runner = TrtInferenceRunner(engine_path=default_engine)

    n_test = 2048
    dummy_feats = np.random.randn(n_test, 4).astype(np.float32)

    # Initial inference
    out_logits = runner.infer(dummy_feats)
    assert out_logits.shape == (n_test, 8), (
        f"Expected output shape ({n_test}, 8), got {out_logits.shape}"
    )
    print(f"[TRT] Output logits shape: {out_logits.shape}")

    # Benchmark 10 iterations
    n_iters = 10
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = runner.infer(dummy_feats)
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / n_iters) * 1000.0
    print(f"[TRT] Benchmark latency: {avg_ms:.2f} ms ({1000.0 / avg_ms:.1f} Hz throughput)")

    print("[STEP P6.2.2 COMPLETE]")


if __name__ == "__main__":
    main()
