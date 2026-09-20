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


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Image-space augmentation — what training actually consumes
# ═════════════════════════════════════════════════════════════════════════════════════════════
# The two functions above operate on point clouds. Training does not see point clouds: it reads
# the (C, 64, 1024) range-image cache built by scripts/build_range_cache.py, because re-projecting
# 131k points per sample in the dataloader costs more CPU than the forward pass costs GPU. So an
# augmentation that lives in point space has to run inside the cache builder, i.e. be baked in
# once and be identical for all 50 epochs — which is not an augmentation, it is a fixed
# preprocessing choice. These operate on the cached image and are re-rolled every epoch.
#
# WHAT IS DELIBERATELY ABSENT, and why:
#
#   scaling (the 0.95-1.05 in augment_pointcloud above). The network gets `range` as an explicit
#   input channel. Scaling multiplies a real, physically meaningful measurement by a random
#   number, so the net is asked to learn that a 10 m return and a 10.5 m return of the same
#   object are the same thing — but on the OS1-64 that difference is genuine information (beam
#   divergence, point density, intensity falloff all change with range). It trains the model to
#   discard a feature it needs at 50 m, where PERSON already has ~1 pixel to work with.
#
#   coordinate jitter. Same argument: the x/y/z channels are reconstructed as
#   direction_table * range, so jitter that is not consistent with the (fixed) direction table
#   produces an image no sensor can ever emit. It is noise dressed as regularisation.
#
#   yaw rotation. Not absent — it IS random_roll(). A rotation about z is exactly a shift along
#   the azimuth axis, which in the image is a horizontal roll: free, exact, and it needs no
#   re-projection. This is the entire reason the point-space rotation is not missed.

IMAGE_Y_CHANNEL = 2       # [range, x, y, z, intensity] — RellisRangeDataset.CHANNELS


def _is_torch(a) -> bool:
    return hasattr(a, "device") and hasattr(a, "dtype") and not isinstance(a, np.ndarray)


def _roll_w(a, shift: int):
    """Roll the last axis. Same call for a torch tensor and a numpy array so the augmentations
    can run either inside a Dataset (numpy) or on an already-collated batch (torch)."""
    if _is_torch(a):
        import torch
        return torch.roll(a, shifts=int(shift), dims=-1)
    return np.roll(a, int(shift), axis=-1)


def _flip_w(a):
    if _is_torch(a):
        import torch
        return torch.flip(a, dims=[-1])
    return np.ascontiguousarray(a[..., ::-1])


def _rng(rng):
    return np.random if rng is None else rng


def random_roll(img, label=None, rng=None, shift: int = None):
    """Random horizontal (azimuth) roll of a range image. EXACTLY a random yaw rotation.

    Column u of the image is yaw = (2u/W - 1) * pi, so rolling by k columns rotates the whole
    scan by 2*pi*k/W about the sensor z axis — no resampling, no interpolation, no new empty
    pixels, and the x/y/z channels stay consistent with their own pixels because each pixel
    carries its own reconstructed xyz and moves with it.

    This only stays exact because every conv in SalsaNextLite pads W circularly (see
    model.RangeConv2d). With zero padding on W, a roll would move real data across a seam the
    network treats as a scene boundary and the augmentation would be teaching the seam.
    """
    W = img.shape[-1]
    if shift is None:
        shift = int(_rng(rng).randint(0, W))
    img = _roll_w(img, shift)
    if label is None:
        return img
    return img, _roll_w(label, shift)


def random_flip(img, label=None, y_channel: int = IMAGE_Y_CHANNEL, y_mean: float = 0.0,
                y_std: float = 1.0, rng=None, do_flip: bool = None):
    """Mirror the azimuth axis AND negate the y channel.

    THE NEGATION IS NOT OPTIONAL. Mirroring W reflects the scene about the sensor's x axis, so a
    point that was 3 m to the left is now 3 m to the right. The image carries y as an input
    channel; flip the pixels without negating y and every sample says "this geometry is on the
    left" while its own y channel says "right". Nothing crashes, no shape changes, the loss still
    goes down — the net simply learns to ignore y, and the one channel that disambiguates a
    pedestrian stepping in from the left versus the right stops carrying signal. That failure is
    invisible in training curves, which is exactly why it is worth this paragraph.

    x, z and range are unchanged by the reflection; intensity is a scalar property of the return.

    Normalisation caveat: RellisRangeDataset hands out (y - mean)/std, and negating a normalised
    value is only the same as normalising a negated value when mean == 0. A full 360 deg sweep is
    left-right symmetric so the cache's y mean is ~0 by construction, and the defaults below
    assume it. If a future cache reports a non-zero y mean, pass it as y_mean/y_std (from
    configs/norm_stats.json) and the exact correction is applied instead.
    """
    if do_flip is None:
        do_flip = _rng(rng).rand() < 0.5
    if not do_flip:
        return img if label is None else (img, label)

    img = _flip_w(img)
    if y_channel is not None and img.shape[-3] > y_channel:
        if y_mean == 0.0:
            img[..., y_channel, :, :] = -img[..., y_channel, :, :]
        else:
            # y_norm' = (-y - mean)/std = -y_norm - 2*mean/std
            img[..., y_channel, :, :] = -img[..., y_channel, :, :] - 2.0 * y_mean / max(y_std, 1e-6)
    if label is None:
        return img
    return img, _flip_w(label)


def random_dropout(img, label=None, p: float = 0.05, ignore_index: int = None, rng=None):
    """Drop pixels to simulate missing returns.

    Real OS1-64 scans lose returns constantly: retroreflective and very dark surfaces, rain and
    dust, beams that hit nothing inside the 100 m range gate. In the cache those pixels are
    exactly zero across all channels (dataset.py multiplies the features by the return mask) and
    carry IGNORE_INDEX as their label. This reproduces that distribution rather than inventing a
    new one — a dropped pixel is zeroed in EVERY channel, not just range, and its label becomes
    IGNORE_INDEX so the loss is not asked to name a class for a pixel with no measurement.

    p=0.05 is a deliberate default: RELLIS scans already come in at roughly 10-20% empty after
    projection, so 5% is a perturbation of that rate, not a second, larger source of holes.
    """
    if ignore_index is None:
        from .taxonomy import IGNORE_INDEX       # lazy: keeps the taxonomy import out of module scope
        ignore_index = IGNORE_INDEX
    if p <= 0.0:
        return img if label is None else (img, label)

    H, W = img.shape[-2], img.shape[-1]
    drop = _rng(rng).rand(H, W) < p
    if _is_torch(img):
        import torch
        drop_t = torch.from_numpy(drop).to(img.device)
        img = img.clone()
        img[..., drop_t] = 0.0
    else:
        img = img.copy()
        img[..., drop] = 0.0
    if label is None:
        return img
    if _is_torch(label):
        import torch
        label = label.clone()
        label[..., torch.from_numpy(drop).to(label.device)] = ignore_index
    else:
        label = label.copy()
        label[..., drop] = ignore_index
    return img, label


def lasermix_image(a, b, num_bands: int = 4, rng=None):
    """LaserMix on the range image: splice alternating ROW bands from two scans.

    `a` and `b` are (img, label) pairs; returns a new (img, label) pair.

    Rows are inclination. That is the whole point of LaserMix (Kong et al., CVPR 2023): in a
    LiDAR scan the semantic content is strongly determined by pitch — the bottom rows are road
    and grass, the middle rows are vehicles, people and trunks, the top rows are canopy and sky.
    Mixing along inclination therefore produces a scan whose layout is still physically
    plausible, and the network is forced to use local evidence instead of "row 58 is always
    ground".

    The existing point-space `lasermix()` above splits on AZIMUTH instead. That mixes the left
    half of one scene with the right half of another, which is a valid scan only by accident and
    destroys no prior in particular. It is kept because other code imports it; this is the one to
    use for range-image training.

    Row bands are also strictly cheaper here: this is two slice assignments on a contiguous
    array, versus the point version's two arctan2 calls plus a concatenate over 131k points.
    """
    img_a, lab_a = a
    img_b, lab_b = b
    if img_a.shape != img_b.shape or lab_a.shape != lab_b.shape:
        raise ValueError(f"lasermix_image needs matching shapes, got {tuple(img_a.shape)} / "
                         f"{tuple(img_b.shape)}")
    H = img_a.shape[-2]
    r = _rng(rng)
    num_bands = max(2, int(num_bands))
    # Random band boundaries rather than a fixed stride: a fixed stride makes the mix predictable
    # and the network can learn the band pattern instead of the terrain.
    cuts = np.sort(r.choice(np.arange(1, H), size=min(num_bands - 1, H - 1), replace=False))
    edges = [0, *cuts.tolist(), H]

    if _is_torch(img_a):
        out_img, out_lab = img_a.clone(), lab_a.clone()
    else:
        out_img, out_lab = img_a.copy(), lab_a.copy()

    take_b = bool(r.rand() < 0.5)          # which parity starts from scan B
    for i in range(len(edges) - 1):
        if take_b:
            lo, hi = edges[i], edges[i + 1]
            out_img[..., lo:hi, :] = img_b[..., lo:hi, :]
            out_lab[..., lo:hi, :] = lab_b[..., lo:hi, :]
        take_b = not take_b
    return out_img, out_lab
