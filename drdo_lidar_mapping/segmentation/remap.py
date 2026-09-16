import numpy as np

# RELLIS-3D ID → DRDO-8 ADL-1 ID
# 0: GROUND, 1: GRAVEL_DIRT, 2: GRASS_LOW, 3: VEGETATION_DENSE,
# 4: OBSTACLE_HARD, 5: WATER_MUD, 6: UNKNOWN, 7: SKY_NOISE (discard)
RELLIS_TO_DRDO = {
    0:  7,   # void/sky       → SKY_NOISE   (discard)
    1:  1,   # dirt           → GRAVEL_DIRT
    3:  2,   # grass          → GRASS_LOW
    4:  3,   # tree           → VEGETATION_DENSE
    5:  4,   # pole           → OBSTACLE_HARD
    6:  5,   # water          → WATER_MUD
    7:  7,   # sky            → SKY_NOISE   (discard)
    8:  4,   # vehicle        → OBSTACLE_HARD
    9:  4,   # object         → OBSTACLE_HARD
    10: 0,   # asphalt        → GROUND
    12: 4,   # building       → OBSTACLE_HARD
    15: 4,   # log            → OBSTACLE_HARD
    17: 4,   # person         → OBSTACLE_HARD
    18: 4,   # fence          → OBSTACLE_HARD
    19: 3,   # bush           → VEGETATION_DENSE
    23: 0,   # concrete       → GROUND
    27: 4,   # barrier        → OBSTACLE_HARD
    29: 5,   # puddle         → WATER_MUD
    31: 5,   # mud            → WATER_MUD
    33: 3,   # rubberChips    → VEGETATION_DENSE
}

def remap_rellis(label_array: np.ndarray) -> np.ndarray:
    """Remap raw RELLIS-3D class labels to DRDO 8-class traversability taxonomy."""
    out = np.full_like(label_array, 6, dtype=np.uint8)  # default UNKNOWN (6)
    for rellis_id, drdo_id in RELLIS_TO_DRDO.items():
        out[label_array == rellis_id] = drdo_id
    return out
