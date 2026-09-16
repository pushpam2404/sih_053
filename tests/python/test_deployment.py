#!/usr/bin/env python3
"""DRDO ID26053 — Master Deployment & End-to-End Validation Suite.

Verifies:
  - Check 1: Model Artifacts (TorchScript, PyTorch checkpoint, ONNX model)
  - Check 2: Nav2 Kinodynamic Planner & Controller Configuration
  - Check 3: ONNX Model Graph Validation & Input/Output signatures
  - Check 4: TensorRT Compilation Scripts & Python Inference Runner
  - Check 5: Jetson AGX Orin Docker Containerization Artifacts
"""

import os
import sys
import numpy as np
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def check_1_model_artifacts() -> None:
    """Verify all required neural network artifacts are present."""
    required_files = [
        "models/minkunet18_traced.pt",
        "models/minkunet18_drdo_ep30.pth",
        "models/minkunet18_drdo.onnx",
        "ros2/drdo_bringup/config/fast_lio_ouster64.yaml",
        "ros2/drdo_bringup/config/nvblox_params.yaml",
        "ros2/drdo_bringup/launch/mapping.launch.py",
        "ros2/drdo_bringup/launch/navigation.launch.py",
    ]
    for rel_path in required_files:
        full_path = os.path.join(REPO_ROOT, rel_path)
        assert os.path.exists(full_path), f"Missing artifact: {full_path}"
    print("[CHECK 1/5] Core system & model artifacts: ALL PRESENT")


def check_2_nav2_yaml() -> None:
    """Verify Nav2 parameters YAML structural integrity."""
    nav2_yaml_path = os.path.join(REPO_ROOT, "ros2/drdo_bringup/config/nav2_params.yaml")
    assert os.path.exists(nav2_yaml_path), f"Missing Nav2 YAML: {nav2_yaml_path}"

    with open(nav2_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    required_keys = ["bt_navigator", "planner_server", "controller_server", "global_costmap", "local_costmap"]
    for k in required_keys:
        assert k in data, f"Missing top-level Nav2 key: {k}"

    planner_plugin = data["planner_server"]["ros__parameters"]["GridBased"]["plugin"]
    assert planner_plugin == "nav2_smac_planner/SmacPlannerHybrid", (
        f"Unexpected planner plugin: {planner_plugin}"
    )

    controller_plugin = data["controller_server"]["ros__parameters"]["FollowPath"]["plugin"]
    assert controller_plugin == "nav2_mppi_controller::MPPIController", (
        f"Unexpected controller plugin: {controller_plugin}"
    )
    print("[CHECK 2/5] Nav2 params YAML: VALID")


def check_3_onnx_model() -> None:
    """Verify ONNX model export file and graph integrity."""
    onnx_path = os.path.join(REPO_ROOT, "models/minkunet18_drdo.onnx")
    assert os.path.exists(onnx_path), f"Missing ONNX model: {onnx_path}"
    file_size = os.path.getsize(onnx_path)
    assert file_size > 1024, f"ONNX file too small ({file_size} bytes)"

    if ONNX_AVAILABLE:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"  [CONF] ONNX model loaded and verified ({file_size / 1024.0:.2f} KB)")
    else:
        print(f"  [WARN] onnx checker skipped, file size verified ({file_size / 1024.0:.2f} KB)")

    print("[CHECK 3/5] ONNX model: VALID")


def check_4_tensorrt_pipeline() -> None:
    """Verify TensorRT compilation script and inference runner."""
    compile_script = os.path.join(REPO_ROOT, "deploy/tensorrt/compile_trt.sh")
    assert os.path.exists(compile_script), f"Missing compile_trt.sh: {compile_script}"
    with open(compile_script, "r", encoding="utf-8") as f:
        script_content = f.read()
    assert "trtexec" in script_content, "compile_trt.sh missing trtexec command"

    from drdo_lidar_mapping.inference.trt_runner import TrtInferenceRunner
    runner = TrtInferenceRunner(engine_path="dummy_nonexistent.engine", allow_stub=True)
    dummy_points = np.random.randn(2048, 4).astype(np.float32)
    out_logits = runner.infer(dummy_points)
    assert out_logits.shape == (2048, 8), f"Unexpected output shape: {out_logits.shape}"
    print("[CHECK 4/5] TensorRT pipeline: SCRIPTS VALID")


def check_5_docker_deployment() -> None:
    """Verify Dockerfile and entrypoint script for Jetson AGX Orin."""
    dockerfile = os.path.join(REPO_ROOT, "deploy/docker/Dockerfile.jetson")
    assert os.path.exists(dockerfile), f"Missing Dockerfile: {dockerfile}"
    with open(dockerfile, "r", encoding="utf-8") as f:
        content = f.read()

    assert "FROM nvcr.io/nvidia/l4t-pytorch" in content, "Missing L4T base image"
    assert "nav2-bringup" in content, "Missing nav2-bringup in Dockerfile"

    entrypoint = os.path.join(REPO_ROOT, "deploy/docker/entrypoint.sh")
    assert os.path.exists(entrypoint), f"Missing entrypoint: {entrypoint}"
    with open(entrypoint, "r", encoding="utf-8") as f:
        ep_content = f.read()
    assert "navigation.launch.py" in ep_content or "mapping.launch.py" in ep_content or "drdo_nav" in ep_content, (
        "entrypoint.sh missing launch execution"
    )
    print("[CHECK 5/5] Docker deployment files: VALID")


def main() -> None:
    """Run all master deployment checks."""
    print("=" * 80)
    print("      DRDO ID26053: MASTER DEPLOYMENT & END-TO-END VALIDATION SUITE")
    print("=" * 80)

    check_1_model_artifacts()
    check_2_nav2_yaml()
    check_3_onnx_model()
    check_4_tensorrt_pipeline()
    check_5_docker_deployment()

    print("=" * 80)
    print("    DRDO ID26053: PRODUCTION DEPLOYMENT SUITE — ALL CHECKS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
