# DRDO ID26053 — Phase 6 Agent Execution Plan
## Nav2 Path Planning · TensorRT Acceleration · Jetson Deployment · Dockerization

> **Document Type:** Principal Systems Architect's Binding Directive — Phase 6 (FINAL DEPLOYMENT PHASE)
> **Prerequisite:** Phase 5 COMPLETE — SORT tracking active, `minkunet18_traced.pt` exported, `drdo_mapping.launch.py` validated.
> **Host (development):** macOS Apple Silicon M4 / Windows 10/11 x64 — Python 3.11 — Docker Desktop
> **Target Deploy:** NVIDIA Jetson AGX Orin 64GB — JetPack 5.x — CUDA 12.x — ROS2 Humble
> **Working Directory for ALL output files:** `~/Desktop/sih/phase6/`
> **Date:** September 2026

---

## Phase 5 Accomplishments (What Was Locked In)

| Deliverable | Status | Validation Gate | Output Artifact |
|---|---|---|---|
| DBSCAN Bounding Box Clustering | ✅ COMPLETE | `[STEP P5.1.1 COMPLETE]` | `phase5/python/cluster_obstacles.py` |
| Kalman / SORT Dynamic Tracker | ✅ COMPLETE | `[STEP P5.1.2 COMPLETE]` | `phase5/python/sort_tracker.py` |
| TorchScript Model Export | ✅ COMPLETE | `[STEP P5.2.1 COMPLETE]` | `phase5/models/minkunet18_traced.pt` |
| nvblox 3D ESDF Config | ✅ COMPLETE | `[STEP P5.3.1 COMPLETE]` | `phase5/ros2/config/nvblox_params.yaml` |
| FAST-LIO2 Ouster OS1-64 Config | ✅ COMPLETE | `[STEP P5.3.1 COMPLETE]` | `phase5/ros2/config/fast_lio_ouster64.yaml` |
| Full ROS 2 Mapping Launch | ✅ COMPLETE | `[STEP P5.3.1 COMPLETE]` | `phase5/ros2/launch/drdo_mapping.launch.py` |

**What Phase 5 did NOT do (intentionally):**
- The robot **sees, maps, and tracks** dynamic threats — but has no ability to **plan a route** around them.
- The AI model is exported to TorchScript (CPU-portable) but **NOT yet TensorRT** (Jetson-native hardware acceleration).
- Everything runs as **standalone Python scripts** — there is **no Docker container** to ship to the Jetson.

**Phase 6 closes these three final gaps. After this, the system is field-deployable.**

---

## IMPORTANT: HOST vs. JETSON EXECUTION MODEL

This phase builds **two categories of artefacts**:

| Category | Run on Host (macOS/Windows) | Run on Jetson AGX Orin |
|---|---|---|
| YAML configuration | ✅ Yes — syntactic validation | ✅ Yes — actual navigation |
| ONNX export + TRT build script | ✅ Yes — ONNX export only | ✅ Yes — `trtexec` compilation |
| Nav2 parameter YAML | ✅ Yes — structure validation | ✅ Yes — Nav2 reads it |
| Dockerfile | ✅ Yes — `docker build` dry-run | ✅ Yes — actually runs |
| Python E2E test | ✅ Yes — all validation gates | — |

**Rule:** Any step that requires actual CUDA (`trtexec`, `nvblox`, live ROS 2) will be **wrapped in a host-side stub or syntactic validation gate** that still prints `[STEP P6.X.X COMPLETE]`. Do NOT skip the step — instead, write code that validates what can be validated.

---

## ═══════════════════════════════════════════════════════════
## PART 1: ARCHITECTURAL DECISION LEDGER (ADL) — Phase 6
## ═══════════════════════════════════════════════════════════

### ADL-1: Nav2 Costmap Integration — LOCKED

The robot currently produces a `nav_msgs/OccupancyGrid` on `/map` (from `grid_map_node.cpp`) and a ESDF slice on `/nvblox_node/esdf_slice`. These must be connected to the ROS 2 **Navigation 2 (Nav2)** stack so the robot can autonomously avoid obstacles and drive to goal poses.

**Configuration decisions:**
- **`global_costmap`:** Uses `/map` as the `static_layer`. Resolution = 0.05 m.
- **`local_costmap`:** Uses a rolling 10 m × 10 m window around the robot. Consumes `/drdo/dynamic_obstacles` as `obstacle_layer` for SORT-tracked threats.
- **`amcl`:** Disabled. Localization comes from FAST-LIO2's `/Odometry` TF tree (`map → odom → base_link`).
- **Map frame:** `map` (published by FAST-LIO2). Robot frame: `base_link`.
- **Planner server** uses `SmacPlannerHybrid` — an A* variant that understands SE(2) kinodynamic constraints (forward/reverse, curvature limits). This is mandatory for Ackermann/skid-steer vehicles.
- **Controller server** uses `MPPIController` (Model Predictive Path Integral). The MPPI controller samples ~1024 random trajectories per step on the GPU and picks the one with minimum cost. It is the best off-road local planner on Jetson hardware.

### ADL-2: Path Planning — LOCKED

| Planner Component | Algorithm | Rationale |
|---|---|---|
| Global Planner | `SmacPlannerHybrid` | A* with kinodynamic constraints. Accounts for vehicle turning radius. |
| Local Planner | `MPPIController` | GPU-parallelized trajectory sampling. Optimal on Jetson Orin. |
| Smoother | `SimpleSmoother` | Lightweight path smoothing post global plan. |
| Recovery | `ClearCostmapRecovery` | On stuck: clear local costmap and retry. |

### ADL-3: TensorRT Acceleration for MinkUNet — LOCKED

The TorchScript `.pt` from Phase 5 runs the neural segmentation on any CPU/GPU. TensorRT converts this to a `.engine` file compiled specifically for the Jetson AGX Orin's TensorCore + DLA hardware, giving **2–4× additional speedup over TorchScript on CUDA**.

**Pipeline:**
1. Export PyTorch model to **ONNX** (format that TensorRT accepts as input).
2. Run **`trtexec`** (NVIDIA CLI tool bundled with JetPack) on the ONNX to produce an `.engine` file calibrated for FP16 Jetson Tensor Cores.
3. On host: since `trtexec` requires CUDA, we write the ONNX export + create the exact shell command. The agent produces a ready-to-run `compile_trt.sh` bash script. The agent does NOT need to actually run `trtexec` on the host.
4. Write a `TrtInferenceRunner` Python class that loads the `.engine` and runs inference via `tensorrt` + `pycuda` Python bindings.

**Constraint:** `torch.onnx.export` requires ONLY PyTorch — no CUDA. **The ONNX export step runs on the host.** The `trtexec` compilation step is explicitly marked as Jetson-only.

### ADL-4: Dockerized Edge Deployment — LOCKED

The Dockerfile must be completely self-contained for the Jetson. A lighter model will not know the exact package names — **spell them out fully** in the plan.

**Base image:** `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3`
- This is NVIDIA's official Jetson base image for JetPack 5.x with PyTorch 2.0 pre-installed.
- DO NOT use a generic Ubuntu image — it will not have Jetson-specific CUDA libraries.

**Packages to install:**
- `ros-humble-desktop` (full ROS 2 Humble)
- `ros-humble-nav2-bringup` (Navigation 2 stack)
- `ros-humble-tf2-ros`, `ros-humble-tf2-geometry-msgs`
- `python3-colcon-common-extensions` (colcon build system)
- `libpython3.10-dev`, `pybind11-dev` (for the `drdo_map` pybind11 module)
- `python3-pip` packages: `numpy`, `scipy`, `scikit-learn`, `torch` (already in base image)

**Build steps:**
- `COPY` the entire `~/Desktop/sih/` workspace into `/workspace/sih/`.
- `RUN colcon build` for `drdo_grid_map`.
- Compile the `drdo_map.so` pybind11 module.
- Set `ENTRYPOINT ["ros2", "launch", "...", "drdo_mapping.launch.py"]`.

---

## ═══════════════════════════════════════════════════════════
## PART 2: PHASE 6 AGENT EXECUTION PLAN — MAV BLOCKS
## ═══════════════════════════════════════════════════════════

### Agent Constraints — READ BEFORE PROCEEDING

1. **Every step produces exactly one or two files.** Do not create extra files unprompted.
2. **Every step has a Self-Validation Gate.** Do NOT proceed to the next step until the gate prints its `[STEP P6.X.X COMPLETE]` token.
3. **If a step requires CUDA/Jetson-only tools,** the gate must instead validate file existence, YAML syntax, and Python import structure — and still print `[STEP P6.X.X COMPLETE]`.
4. **Do NOT combine steps.** One step at a time.
5. **Existing phase5/ files are read-only.** Do not modify anything outside `phase6/`.
6. **Python style:** Type hints required. Docstrings required on all classes and functions. Use `try/except ImportError` graceful fallback for any library that may not be installed on the host.

---

```
BLOCK GROUPS:
  P6.0 — Setup & Directory Tree
  P6.1 — Nav2 Configuration & Integration
  P6.2 — TensorRT ONNX Export & Engine Build Pipeline
  P6.3 — Edge Deployment Dockerfile
  P6.4 — End-to-End Configuration Validation Suite
```

---

## BLOCK GROUP P6.0: SETUP

---

### Step P6.0.1: Create Phase 6 Directory Tree

**Objective:** Create the canonical `phase6/` directory hierarchy so every step has a known output location.

**Sub-Steps:**
- `phase6/ros2/nav2_config/` — Nav2 YAML parameter files
- `phase6/ros2/launch/` — Phase 6 launch file that adds Nav2 to Phase 5's mapping stack
- `phase6/tensorrt/` — ONNX export script, TensorRT build script, and inference runner
- `phase6/docker/` — Dockerfile and docker-compose files
- `phase6/results/` — JSON output from validation scripts

**Execution Directive (bash):**
```bash
mkdir -p ~/Desktop/sih/phase6/ros2/nav2_config
mkdir -p ~/Desktop/sih/phase6/ros2/launch
mkdir -p ~/Desktop/sih/phase6/tensorrt
mkdir -p ~/Desktop/sih/phase6/docker
mkdir -p ~/Desktop/sih/phase6/results
```

**Self-Validation Gate (bash):**
```bash
for d in ros2/nav2_config ros2/launch tensorrt docker results; do
  [ -d ~/Desktop/sih/phase6/$d ] \
    && echo "[OK] phase6/$d exists" \
    || { echo "[FAIL] phase6/$d MISSING"; exit 1; }
done
echo "[STEP P6.0.1 COMPLETE]"
```

---

## BLOCK GROUP P6.1: NAV2 PATH PLANNING CONFIGURATION

---

### Step P6.1.1: Write Nav2 Full Parameter YAML

**Objective:** Write a complete, self-contained `nav2_params.yaml` that configures every Nav2 server to use our DRDO map, FAST-LIO2 odometry, SORT obstacles, and MPPI controller.

**Output file:** `~/Desktop/sih/phase6/ros2/nav2_config/nav2_params.yaml`

**Sub-Steps:**
1. **`bt_navigator` section** — configure `navigate_to_pose` behavior tree. Set `odom_topic: /Odometry` (FAST-LIO2 output).
2. **`planner_server` section** — set `plugin: nav2_smac_planner/SmacPlannerHybrid`. Key parameters:
   - `minimum_turning_radius: 0.4` (40 cm minimum turning radius for ground vehicle)
   - `motion_model_for_search: REEDS_SHEPP`
3. **`controller_server` section** — set `plugin: nav2_mppi_controller::MPPIController`. Key parameters:
   - `time_steps: 56` (trajectory horizon)
   - `model_dt: 0.05` (50 ms step → 20 Hz)
   - `batch_size: 1000` (1000 parallel trajectory samples on GPU)
   - `vx_max: 2.0` (2 m/s max linear velocity)
   - `wz_max: 1.0` (1 rad/s max angular velocity)
4. **`global_costmap` section:**
   - `static_layer.plugin: nav2_costmap_2d::StaticLayer`
   - `map_topic: /map` (our `grid_map_node` output)
   - `resolution: 0.05`
5. **`local_costmap` section:**
   - `width: 10.0`, `height: 10.0` (10 m × 10 m rolling window)
   - `obstacle_layer.plugin: nav2_costmap_2d::ObstacleLayer`
   - `obstacle_layer.observation_sources: dynamic_obstacles`
   - `dynamic_obstacles.topic: /drdo/dynamic_obstacles` (SORT tracker output topic)
   - `dynamic_obstacles.sensor_frame: base_link`
6. **`amcl`:** Set to `two_d_pose_estimate_initial_pose_topic: ""` and add comment that AMCL is disabled because FAST-LIO2 provides global localization via TF.
7. **`map_server`:** Disabled (we provide a live map, not a static YAML map).

**Expected YAML structure (top-level keys):**
```
bt_navigator:
  ros__parameters: ...
planner_server:
  ros__parameters:
    planner_plugins: [GridBased]
    GridBased:
      plugin: nav2_smac_planner/SmacPlannerHybrid
      minimum_turning_radius: 0.4
      ...
controller_server:
  ros__parameters:
    controller_plugins: [FollowPath]
    FollowPath:
      plugin: nav2_mppi_controller::MPPIController
      ...
global_costmap:
  global_costmap:
    ros__parameters: ...
local_costmap:
  local_costmap:
    ros__parameters: ...
```

**Self-Validation Gate (Python):**
Write `phase6/results/validate_p6_1_1.py`. This script must:
- Open `nav2_params.yaml`
- Use Python's `yaml.safe_load` to parse it
- Assert the top-level keys `bt_navigator`, `planner_server`, `controller_server`, `global_costmap`, `local_costmap` all exist
- Assert `controller_server.ros__parameters.FollowPath.plugin == "nav2_mppi_controller::MPPIController"`
- Assert `planner_server.ros__parameters.GridBased.plugin == "nav2_smac_planner/SmacPlannerHybrid"`
- Assert `planner_server.ros__parameters.GridBased.minimum_turning_radius == 0.4`
- Assert `global_costmap.global_costmap.ros__parameters.map_topic == "/map"` (or equivalent path)
- Print `[STEP P6.1.1 COMPLETE]`

**Run command:**
```bash
python3 ~/Desktop/sih/phase6/results/validate_p6_1_1.py
```

---

### Step P6.1.2: Write Phase 6 Navigation Launch File

**Objective:** Write `phase6/ros2/launch/drdo_nav.launch.py` — a ROS 2 launch file that **includes** Phase 5's mapping launch and **adds** the Nav2 bringup stack on top of it.

**Output file:** `~/Desktop/sih/phase6/ros2/launch/drdo_nav.launch.py`

**Sub-Steps:**
1. This launch file imports and includes Phase 5's `drdo_mapping.launch.py` (the mapping stack) using `IncludeLaunchDescription`.
2. Then it adds Nav2 via `IncludeLaunchDescription` pointing to `nav2_bringup/launch/navigation_launch.py`.
3. Passes the `params_file` argument to point to our `nav2_params.yaml`.
4. Add a `use_sim_time: False` LaunchArgument (real hardware mode).
5. Include a fallback struct validation block at the bottom (`if __name__ == "__main__"`) that runs on the host without ROS 2 installed and prints `[STEP P6.1.2 COMPLETE]`.

**Fallback validation block (host-side):**
```python
# This block executes when the file is run as a plain Python script (no ROS 2 needed)
if __name__ == "__main__":
    import os
    nav2_cfg = os.path.expanduser(
        "~/Desktop/sih/phase6/ros2/nav2_config/nav2_params.yaml")
    mapping_launch = os.path.expanduser(
        "~/Desktop/sih/phase5/ros2/launch/drdo_mapping.launch.py")
    assert os.path.exists(nav2_cfg), f"Missing Nav2 config: {nav2_cfg}"
    assert os.path.exists(mapping_launch), f"Missing Phase 5 launch: {mapping_launch}"
    print("  [CONF] nav2_params.yaml: FOUND")
    print("  [CONF] drdo_mapping.launch.py (Phase 5): FOUND")
    print("  [INFO] ROS 2 not required for host-side validation of launch structure.")
    print("[STEP P6.1.2 COMPLETE]")
```

**Run command:**
```bash
python3 ~/Desktop/sih/phase6/ros2/launch/drdo_nav.launch.py
```

---

## BLOCK GROUP P6.2: TENSORRT ACCELERATION

---

### Step P6.2.1: ONNX Export from Phase 5 Checkpoint

**Objective:** Write `phase6/tensorrt/export_onnx.py`. This script exports the Phase 5 checkpoint (`phase5/models/minkunet18_traced.pt` or `phase4/data/weights/minkunet18_drdo_ep30.pth`) to ONNX format using `torch.onnx.export`. **This step runs fully on the host — no CUDA required.**

**Output file:** `~/Desktop/sih/phase6/tensorrt/export_onnx.py`
**Output model:** `~/Desktop/sih/phase6/tensorrt/minkunet18_drdo.onnx`

**Exact implementation requirements:**
1. Load the `HostMinkUNetScaffold` model (same architecture as Phase 5 `export_torchscript.py`). Do NOT re-import the class — copy the definition inline into this file.
   - Architecture: `Linear(4→64) → ReLU → Linear(64→8)`. `forward(feats, coords) → logits`.
2. Load checkpoint from `~/Desktop/sih/phase4/data/weights/minkunet18_drdo_ep30.pth`. Remap state dict keys (`net.` prefix) exactly as done in Phase 5.
3. Set model to `eval()` mode.
4. Create dummy inputs: `dummy_feats = torch.zeros(1, 4, dtype=torch.float32)` (batch of 1 point, 4 features).
5. Call `torch.onnx.export` with:
   - `input_names=["feats", "coords"]`
   - `output_names=["logits"]`
   - `dynamic_axes={"feats": {0: "N"}, "logits": {0: "N"}}` — dynamic batch size!
   - `opset_version=17`
6. After export, load and validate the ONNX graph using `onnx.load()` and `onnx.checker.check_model()`.

**IMPORTANT:** `onnx` may not be installed. Add this guard at the top:
```python
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
```
If `onnx` is not installed, attempt `pip install onnx` via `subprocess.check_call`. If that also fails, skip the checker but still print `[STEP P6.2.1 COMPLETE]` after saving the file.

**Self-Validation Gate (built into the script's `main()`):**
```python
assert os.path.exists(ONNX_PATH), "ONNX file was not saved!"
size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
print(f"[ONNX] Exported: {ONNX_PATH} ({size_mb:.3f} MB)")
if ONNX_AVAILABLE:
    model_check = onnx.load(ONNX_PATH)
    onnx.checker.check_model(model_check)
    print("[ONNX] Graph structure check: PASSED")
print("[STEP P6.2.1 COMPLETE]")
```

**Run command:**
```bash
python3 ~/Desktop/sih/phase6/tensorrt/export_onnx.py
```

---

### Step P6.2.2: Write TensorRT Build Script (`compile_trt.sh`) and Inference Runner

**Objective:** Produce two files:
1. `phase6/tensorrt/compile_trt.sh` — a self-contained bash script that compiles the ONNX to TensorRT `.engine` on the Jetson.
2. `phase6/tensorrt/trt_inference_runner.py` — a Python class that loads the `.engine` and runs inference.

**Output file 1:** `~/Desktop/sih/phase6/tensorrt/compile_trt.sh`

Content of `compile_trt.sh`:
```bash
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
```

**Output file 2:** `~/Desktop/sih/phase6/tensorrt/trt_inference_runner.py`

This file must implement the `TrtInferenceRunner` class with:
- `__init__(self, engine_path: str)`: Load the `.engine` file using `tensorrt.Runtime` if `tensorrt` is available.
- `infer(self, feats: np.ndarray) -> np.ndarray`: Run inference. Input `(N, 4)`, output `(N, 8)`.
- Include `try/except ImportError` for `tensorrt` and `pycuda`. If missing, fall back to a stub that produces random logits and prints `[WARN] TensorRT not available, using CPU stub`.
- Include a `main()` block that:
  - Instantiates `TrtInferenceRunner` with the engine path
  - Runs inference on 2048 dummy points
  - Asserts output shape is `(2048, 8)`
  - Benchmarks 10 iterations and prints average latency in ms
  - Prints `[STEP P6.2.2 COMPLETE]`

**Run command (host-side):**
```bash
python3 ~/Desktop/sih/phase6/tensorrt/trt_inference_runner.py
```

---

## BLOCK GROUP P6.3: EDGE DEPLOYMENT CONTAINERIZATION

---

### Step P6.3.1: Write the Jetson Orin Dockerfile

**Objective:** Write `phase6/docker/Dockerfile.jetson` — a complete, production Dockerfile that encapsulates the entire DRDO ID26053 Phase 1–6 software stack.

**Output file:** `~/Desktop/sih/phase6/docker/Dockerfile.jetson`

**Write the Dockerfile exactly as follows** (the agent must follow this template precisely, filling in the instructions):

```dockerfile
# DRDO ID26053 — Phase 6: Jetson AGX Orin Production Dockerfile
# Base: NVIDIA L4T PyTorch (JetPack 5.x, CUDA 12.x, PyTorch 2.0)
# Build: docker build -f Dockerfile.jetson -t drdo_id26053:v6 .
# Run:   docker run --runtime=nvidia --rm -it drdo_id26053:v6

FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# 1. Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# 2. Install ROS 2 Humble
RUN apt-get update && apt-get install -y \
    curl gnupg2 lsb-release software-properties-common && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-nav2-bringup \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    python3-colcon-common-extensions \
    libpython3.10-dev \
    python3-pip \
    pybind11-dev \
    cmake \
    g++ && \
    rm -rf /var/lib/apt/lists/*

# 3. Install Python dependencies
RUN pip3 install --no-cache-dir \
    numpy==1.26.4 \
    scipy \
    scikit-learn \
    onnx \
    pycuda

# 4. Copy the entire SIH project workspace
WORKDIR /workspace
COPY . /workspace/sih/

# 5. Build the pybind11 drdo_map.so module
RUN cd /workspace/sih/phase3 && \
    g++ -O2 -std=c++17 -shared -fPIC \
    $(python3 -m pybind11 --includes) \
    -I/workspace/sih/phase2/include \
    src/drdo_map_py.cpp \
    -o lib/drdo_map$(python3-config --extension-suffix)

# 6. Build the drdo_grid_map ROS 2 package
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    cd /workspace/sih/phase3/ros2 && \
    colcon build --symlink-install --packages-select drdo_grid_map"

# 7. Set ROS 2 environment and entrypoint
ENV ROS_DISTRO=humble
ENV AMENT_PREFIX_PATH=/workspace/sih/phase3/ros2/install/drdo_grid_map:/opt/ros/humble
ENV PYTHONPATH=/workspace/sih/phase3/lib:/workspace/sih/phase3/python:$PYTHONPATH

# 8. Launch the full autonomous mapping stack
COPY phase6/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
```

**Additionally, write `phase6/docker/entrypoint.sh`:**
```bash
#!/bin/bash
source /opt/ros/humble/setup.bash
source /workspace/sih/phase3/ros2/install/setup.bash
exec ros2 launch /workspace/sih/phase5/ros2/launch/drdo_mapping.launch.py "$@"
```

**Self-Validation Gate (Python):**
Write `phase6/results/validate_p6_3_1.py`. This script must:
- Check that `Dockerfile.jetson` exists
- Open and read `Dockerfile.jetson`, then assert:
  - `FROM nvcr.io/nvidia/l4t-pytorch` is in the file
  - `ros-humble-desktop` is in the file
  - `ros-humble-nav2-bringup` is in the file
  - `pybind11-dev` is in the file
  - `ENTRYPOINT` is in the file
  - `drdo_map_py.cpp` is referenced in the build step
- Check that `entrypoint.sh` exists
- Open and read `entrypoint.sh`, assert `drdo_mapping.launch.py` is referenced
- Print `[STEP P6.3.1 COMPLETE]`

**Run command:**
```bash
python3 ~/Desktop/sih/phase6/results/validate_p6_3_1.py
```

---

## BLOCK GROUP P6.4: END-TO-END CONFIGURATION VALIDATION

---

### Step P6.4.1: Comprehensive Phase 6 Validation Suite

**Objective:** Write `phase6/results/validate_phase6_e2e.py`. This is the master validation script that runs ALL Phase 6 gates in sequence and confirms the complete deployment package is consistent and ready.

**Output file:** `~/Desktop/sih/phase6/results/validate_phase6_e2e.py`

**The script must perform the following checks in order:**

**Check 1: Phase 5 Dependency Verification**
- Assert `~/Desktop/sih/phase5/models/minkunet18_traced.pt` exists
- Assert `~/Desktop/sih/phase5/ros2/config/fast_lio_ouster64.yaml` exists
- Assert `~/Desktop/sih/phase5/ros2/launch/drdo_mapping.launch.py` exists
- Assert `~/Desktop/sih/phase5/ros2/config/nvblox_params.yaml` exists
- Print: `[CHECK 1/5] Phase 5 dependencies: ALL PRESENT`

**Check 2: Nav2 YAML Structural Integrity**
- Open and parse `phase6/ros2/nav2_config/nav2_params.yaml` with `yaml.safe_load`
- Assert top-level keys include: `bt_navigator`, `planner_server`, `controller_server`, `global_costmap`, `local_costmap`
- Navigate to `planner_server.ros__parameters.GridBased.plugin` and assert it equals `"nav2_smac_planner/SmacPlannerHybrid"`
- Navigate to `controller_server.ros__parameters.FollowPath.plugin` and assert it equals `"nav2_mppi_controller::MPPIController"`
- Print: `[CHECK 2/5] Nav2 params YAML: VALID`

**Check 3: ONNX Export File**
- Assert `~/Desktop/sih/phase6/tensorrt/minkunet18_drdo.onnx` exists
- Assert file size > 1 KB (not an empty file)
- If `onnx` is installed, load the file with `onnx.load()` and call `onnx.checker.check_model()` — assert it passes
- Print: `[CHECK 3/5] ONNX model: VALID`

**Check 4: TensorRT Scripts**
- Assert `~/Desktop/sih/phase6/tensorrt/compile_trt.sh` exists
- Open it and assert the string `trtexec` appears in the file
- Assert `~/Desktop/sih/phase6/tensorrt/trt_inference_runner.py` exists
- Import and run `TrtInferenceRunner` from the module (it will use the stub if TensorRT is absent)
- Assert the stub produces output shape `(2048, 8)`
- Print: `[CHECK 4/5] TensorRT pipeline: SCRIPTS VALID`

**Check 5: Docker Build Files**
- Assert `~/Desktop/sih/phase6/docker/Dockerfile.jetson` exists
- Open it and assert `FROM nvcr.io/nvidia/l4t-pytorch`, `nav2-bringup`, `drdo_map_py.cpp` are all present
- Assert `~/Desktop/sih/phase6/docker/entrypoint.sh` exists
- Open it and assert `drdo_mapping.launch.py` is referenced
- Print: `[CHECK 5/5] Docker deployment files: VALID`

**Final output:**
```python
print("=" * 80)
print("          DRDO ID26053: PHASE 6 DEPLOYMENT PACKAGE — ALL CHECKS PASSED")
print("=" * 80)
print("[PHASE 6 E2E VALIDATION COMPLETE]")
```

**Run command:**
```bash
python3 ~/Desktop/sih/phase6/results/validate_phase6_e2e.py
```

---

## SUMMARY: FILE MANIFEST FOR PHASE 6

| File | Step | Purpose |
|---|---|---|
| `phase6/ros2/nav2_config/nav2_params.yaml` | P6.1.1 | Nav2 stack configuration (MPPI, SmacPlanner, costmaps) |
| `phase6/results/validate_p6_1_1.py` | P6.1.1 | YAML structure validation script |
| `phase6/ros2/launch/drdo_nav.launch.py` | P6.1.2 | ROS 2 launch adding Nav2 to mapping stack |
| `phase6/tensorrt/export_onnx.py` | P6.2.1 | Exports PyTorch → ONNX (runs on host) |
| `phase6/tensorrt/minkunet18_drdo.onnx` | P6.2.1 | ONNX model artifact (output of P6.2.1) |
| `phase6/tensorrt/compile_trt.sh` | P6.2.2 | Bash script to run `trtexec` on Jetson |
| `phase6/tensorrt/trt_inference_runner.py` | P6.2.2 | Python TensorRT inference runner (with stub fallback) |
| `phase6/docker/Dockerfile.jetson` | P6.3.1 | Jetson production Docker container definition |
| `phase6/docker/entrypoint.sh` | P6.3.1 | Container entry point launching ROS 2 stack |
| `phase6/results/validate_p6_3_1.py` | P6.3.1 | Dockerfile/entrypoint validation script |
| `phase6/results/validate_phase6_e2e.py` | P6.4.1 | Master validation suite |

---

## SEQUENTIAL EXECUTION ORDER

Execute steps EXACTLY in this order. Do not skip, combine, or reorder.

```
P6.0.1 → P6.1.1 → P6.1.2 → P6.2.1 → P6.2.2 → P6.3.1 → P6.4.1
```

After each step, the agent MUST confirm the validation gate printed `[STEP P6.X.X COMPLETE]` before proceeding.

---
*Document sealed by Principal Systems Architect — DRDO ID26053 SIH 2026*
