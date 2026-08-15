"""DriftSense V2 localization pipeline.

Reference -> preprocessing -> scale/rotation handling -> candidate
generation -> feature extraction -> candidate scoring -> ranking ->
subpixel refinement -> final coordinate -> confidence/ambiguity score.

Never reads ground truth. See localize.py for the full pipeline and
reports/V2_ARCHITECTURE_PLAN.md section 5 for the design rationale.
"""

from .localize import LocalizationResult, localize

__all__ = ["localize", "LocalizationResult"]
