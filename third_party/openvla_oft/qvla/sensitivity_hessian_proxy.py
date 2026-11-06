"""Deprecated filename: OFT calibration now uses native-action Jacobians."""

import sys
import warnings
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qvla.calibrate_actions import main

if __name__ == "__main__":
    warnings.warn(
        "Use `python -m qvla.calibrate_actions`; this alias uses its arguments and "
        "computes action Jacobians, not an input-Gram Hessian.",
        FutureWarning,
        stacklevel=2,
    )
    main()
