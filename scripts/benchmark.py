import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.expanduser("~/Desktop/sih/phase3/lib"))
sys.path.insert(0, os.path.expanduser("~/Desktop/sih/phase3/python"))
import drdo_map
from mink_inference import MinkUNetInference

CKPT = os.path.expanduser("~/Desktop/sih/phase4/data/weights/minkunet18_drdo_ep30.pth")
RESULTS = os.path.expanduser("~/Desktop/sih/phase4/results/e2e_benchmark.json")

def benchmark():
    print("[E2E BENCHMARK] Initializing model and map...")
    model = MinkUNetInference(checkpoint_path=CKPT, device="auto")
    drdo_map.reset_map()

    N_FRAMES = 20
    N_POINTS = 500  # realistic point cloud scan per frame for end-to-end timing

    frame_times = []
    infer_times = []
    insert_times = []
    classify_times = []

    for frame in range(1, N_FRAMES + 1):
        robot_x = (frame - 1) * 0.2
        robot_y = 0.0

        # Generate synthetic frame points
        angles = np.random.uniform(-np.pi, np.pi, N_POINTS)
        radii  = np.random.uniform(1.0, 40.0, N_POINTS)
        x = robot_x + radii * np.cos(angles)
        y = robot_y + radii * np.sin(angles)
        z = np.random.uniform(-0.1, 1.5, N_POINTS)
        intensity = np.random.uniform(0.0, 1.0, N_POINTS)
        pts = np.stack([x, y, z, intensity], axis=1).astype(np.float32)

        t0 = time.perf_counter()

        # Step 1: Semantic Inference
        t_inf_start = time.perf_counter()
        labeled = model.infer(pts)
        t_inf = (time.perf_counter() - t_inf_start) * 1000.0
        infer_times.append(t_inf)

        # Step 2: C++ Map Engine Point Insertion
        t_ins_start = time.perf_counter()
        for i in range(len(labeled)):
            drdo_map.insert_point(
                float(labeled[i, 0]), float(labeled[i, 1]), float(labeled[i, 2]),
                int(labeled[i, 3]), robot_x, robot_y, frame
            )
        t_ins = (time.perf_counter() - t_ins_start) * 1000.0
        insert_times.append(t_ins)

        # Step 3: Classify & score + decay
        t_cls_start = time.perf_counter()
        drdo_map.classify_and_score_all(0.0)
        if frame % 5 == 0:
            drdo_map.decay_kernel(frame)
        t_cls = (time.perf_counter() - t_cls_start) * 1000.0
        classify_times.append(t_cls)

        total_frame_ms = (time.perf_counter() - t0) * 1000.0
        frame_times.append(total_frame_ms)

    mean_total = float(np.mean(frame_times))
    p95_total  = float(np.percentile(frame_times, 95))
    mean_inf   = float(np.mean(infer_times))
    mean_ins   = float(np.mean(insert_times))
    mean_cls   = float(np.mean(classify_times))

    report = {
        "n_frames": N_FRAMES,
        "points_per_frame": N_POINTS,
        "mean_frame_ms": mean_total,
        "p95_frame_ms": p95_total,
        "mean_inference_ms": mean_inf,
        "mean_insert_ms": mean_ins,
        "mean_classify_decay_ms": mean_cls,
        "target_budget_ms": 100.0,
        "passed": mean_total <= 100.0
    }

    print("\n--- End-to-End Pipeline Latency Profile ---")
    print(f"  Total Latency per Frame: Mean = {mean_total:.2f} ms | P95 = {p95_total:.2f} ms (Budget <= 100 ms)")
    print(f"  Breakdown:")
    print(f"    - MinkUNet Inference:  {mean_inf:.2f} ms")
    print(f"    - C++ Map Insertion:    {mean_ins:.2f} ms")
    print(f"    - Classify & Decay:     {mean_cls:.2f} ms")

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(report, f, indent=2)

    assert mean_total <= 100.0, f"Frame latency {mean_total:.2f}ms exceeds 100ms budget!"
    print("[STEP P4.7.1 COMPLETE]")

if __name__ == "__main__":
    benchmark()
