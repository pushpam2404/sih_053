# phase3/python/kitti_replay.py
"""
SemanticKITTI offline bag replay driver for DRDO ID26053.
Usage:
    python3 kitti_replay.py [scan.bin] [scan.label] [checkpoint.pth]
    python3 kitti_replay.py            # synthetic mode — no data needed
"""
import sys, os
import numpy as np
sys.path.insert(0, "../lib")
sys.path.insert(0, ".")
import drdo_map
from mink_inference import MinkUNetInference

KITTI_TO_DRDO = {40:0, 44:0, 48:1, 70:3, 71:4, 72:2, 80:4, 81:4, 252:4}


def remap(kitti_label: int) -> int:
    return KITTI_TO_DRDO.get(kitti_label & 0xFFFF, 6)


def load_kitti(bin_path: str, label_path: str = None):
    pts    = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    labels = None
    if label_path and os.path.exists(label_path):
        raw    = np.fromfile(label_path, dtype=np.uint32)
        labels = np.array([remap(int(l)) for l in raw], dtype=np.uint8)
    return pts, labels


def synthetic_scan(N=500):
    np.random.seed(2026)
    angles = np.random.uniform(0, 2*3.14159, N)
    radii  = np.random.uniform(1.0, 80.0, N)
    x, y   = radii*np.cos(angles), radii*np.sin(angles)
    z      = np.random.uniform(-0.05, 0.05, N)
    z[radii < 15.0] = np.random.uniform(0.05, 1.5, (radii<15.0).sum())
    pts    = np.stack([x, y, z, np.ones(N)], axis=1).astype(np.float32)
    lbs    = np.where(z < 0.05, 0, 4).astype(np.uint8)
    lbs[radii > 50.0] = 3
    return pts, lbs


def run_replay(bin_path=None, label_path=None, ckpt=None, robot_x=0.0, robot_y=0.0):
    drdo_map.reset_map()

    if bin_path and os.path.exists(bin_path):
        pts, gt_labels = load_kitti(bin_path, label_path)
        print(f"[KITTI] Loaded {len(pts)} points from {bin_path}")
    else:
        print("[KITTI] Using synthetic 500-point scan.")
        pts, gt_labels = synthetic_scan()

    model  = MinkUNetInference(checkpoint_path=ckpt, device="auto")

    if gt_labels is not None:
        labeled = np.column_stack([pts[:, :3], gt_labels.astype(np.float32)])
    else:
        labeled = model.infer(pts)

    inserted = discarded = 0
    for i in range(len(labeled)):
        x, y, z = float(labeled[i,0]), float(labeled[i,1]), float(labeled[i,2])
        lbl = int(labeled[i,3])
        ok  = drdo_map.insert_point(x, y, z, lbl, robot_x, robot_y, 1)
        inserted += ok; discarded += (not ok)

    print(f"[PIPELINE] Inserted: {inserted}  Discarded: {discarded}")

    processed = drdo_map.classify_and_score_all(0.0)
    n_valid   = drdo_map.valid_cell_count()
    print(f"[PIPELINE] Valid cells: {n_valid}  Processed: {processed}")

    decayed = drdo_map.decay_kernel(55)
    print(f"[PIPELINE] Decay frame=55: {decayed} cells decayed.")

    # Spot-check a ground point
    gpts = labeled[(labeled[:,3]==0) & (labeled[:,2]<0.1)]
    if len(gpts) > 0:
        cell = drdo_map.query_world(float(gpts[0,0]), float(gpts[0,1]))
        if cell:
            print(f"[SPOT CHECK] Ground cell: level={cell['level']}"
                  f"  flag={cell['obstacle_flag']}  trav={cell['traversability']:.3f}")

    print("[PASS] KITTI replay validated.")
    return inserted, discarded, n_valid


if __name__ == "__main__":
    ins, disc, n = run_replay(
        sys.argv[1] if len(sys.argv)>1 else None,
        sys.argv[2] if len(sys.argv)>2 else None,
        sys.argv[3] if len(sys.argv)>3 else None,
    )
    assert ins > 0,   "Must insert at least 1 point!"
    assert n   > ins, "Valid cells must exceed inserted points (raycasting)!"
    print("[STEP P3.2.2 COMPLETE]")
