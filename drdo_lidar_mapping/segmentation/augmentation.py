import numpy as np

def lasermix(pts_A: np.ndarray, lbs_A: np.ndarray,
             pts_B: np.ndarray, lbs_B: np.ndarray,
             split_angle: float = None):
    """LaserMix 3D data augmentation: combines scans across azimuth angles."""
    if split_angle is None:
        split_angle = np.random.uniform(-np.pi, np.pi)
    az_A = np.arctan2(pts_A[:, 1], pts_A[:, 0])
    az_B = np.arctan2(pts_B[:, 1], pts_B[:, 0])
    mask_A = az_A <  split_angle
    mask_B = az_B >= split_angle
    mixed_pts = np.concatenate([pts_A[mask_A], pts_B[mask_B]], axis=0)
    mixed_lbs = np.concatenate([lbs_A[mask_A], lbs_B[mask_B]], axis=0)
    return mixed_pts, mixed_lbs

def augment_pointcloud(pts: np.ndarray,
                       yaw_range=(-np.pi, np.pi),
                       scale_range=(0.95, 1.05),
                       jitter_std=0.01) -> np.ndarray:
    """Random yaw rotation, isotropic scaling, and coordinate jitter."""
    yaw = np.random.uniform(*yaw_range)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rot_mat = np.array([[cos_y, -sin_y, 0],
                        [sin_y,  cos_y, 0],
                        [0,      0,     1]], dtype=np.float32)

    aug_pts = pts.copy()
    aug_pts[:, :3] = aug_pts[:, :3] @ rot_mat.T
    scale = np.random.uniform(*scale_range)
    aug_pts[:, :3] *= scale
    if jitter_std > 0:
        aug_pts[:, :3] += np.random.normal(0, jitter_std, size=aug_pts[:, :3].shape)
    return aug_pts
