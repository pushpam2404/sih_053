"""DRDO ID26053 — backward-compatibility shim over taxonomy.py.

THIS FILE NO LONGER OWNS A MAPPING. `taxonomy.py` is the single source of truth for every
label table in the project; everything here is derived from it at import time.

Why the file was gutted rather than fixed in place: the table that used to live here was wrong
against RELLIS-3D's own ontology (benchmarks/.../config/labels/rellis.yaml) in three ways, and
the third was a safety bug, not a cosmetic one:

  * 29 was called "puddle" — 29 is a grass variant; puddle is 31.
  * 31 was called "mud"    — 31 is puddle.
  * 33 was called "rubberChips" and routed to VEGETATION_DENSE — 33 is MUD. That single entry
    handed a hazard a traversability of 0.15 (drivable vegetation) instead of 0.0, i.e. the
    planner was told it could drive into mud.
  * ids 30, 32 and 34 were absent entirely and fell to the default UNKNOWN.

Two copies of a table drift; one of them was already wrong for an unknown number of commits.
So the table exists once, in taxonomy.py, and this module only re-exports it under the old
names so existing importers (`segmentation/__init__.py`, `scripts/train.py`) keep working.

One behavioural change you must know about: void(0) and sky(7) used to map to ADL-1 7
(SKY_NOISE, which the engine DISCARDS on insert). They now map to ADL-1 6 (UNKNOWN,
traversability 0.40) because taxonomy.fine_to_adl1() refuses to delete a point from the map
just because the network could not name it — an unlabelled return is still an occupied cell.
Class 7 stays reserved for the geometric dust filter. See taxonomy.fine_to_adl1's docstring.
"""
from typing import Dict

import numpy as np

from .taxonomy import (  # noqa: F401  (re-exported for callers that import from here)
    ADL1_NAMES,
    ADL1_UNKNOWN,
    IGNORE_INDEX,
    NUM_ADL1,
    RAW_TO_FINE,
    fine_to_adl1,
    remap_raw_to_fine,
)

__all__ = ["remap_rellis", "remap_rellis_fine", "RELLIS_TO_DRDO",
           "remap_raw_to_fine", "fine_to_adl1"]


def remap_rellis(label_array: np.ndarray) -> np.ndarray:
    """Raw RELLIS-3D ids -> ADL-1 uint8 (0..7), the taxonomy the C++ grid engine consumes.

    Composed as fine_to_adl1(remap_raw_to_fine(...)) so the RAW->FINE->ADL-1 chain is the same
    one training and evaluation use. Unmapped raw ids become UNKNOWN, never a real terrain class.
    """
    return fine_to_adl1(remap_raw_to_fine(label_array))


def remap_rellis_fine(label_array: np.ndarray) -> np.ndarray:
    """Raw RELLIS-3D ids -> the 12 FINE training ids (255 = ignore). Thin alias for symmetry."""
    return remap_raw_to_fine(label_array)


# Kept only because `segmentation/__init__.py` re-exports it. DERIVED — do not hand-edit; edit
# taxonomy.RAW_TO_FINE and this follows. Raw ids that taxonomy sends to IGNORE_INDEX appear here
# as ADL1_UNKNOWN, matching what remap_rellis() actually returns for them.
RELLIS_TO_DRDO: Dict[int, int] = {
    int(_raw): int(fine_to_adl1(np.array([_fine], dtype=np.uint8))[0])
    for _raw, _fine in RAW_TO_FINE.items()
}
