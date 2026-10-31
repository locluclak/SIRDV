"""DV-score selective inference package exports."""

from .blas_threads import pin_blas_threads as _pin_blas_threads

# Small-matrix BLAS calls in the parametric trace are dominated by OpenBLAS
# thread spawn/join overhead; pin to a single BLAS thread by default.
_pin_blas_threads()

from .lava_function import calc_lava_score
from .si_dv import SelectionAdapter, SelectiveInference, SI_DV, NaiveTest, FSNaiveTest, CGSelectionAdapter, DataSplittingTest
from .si_parametric import parametric_intervals_FS_nested, cg_parametric_intervals_FS_nested
from .si_lasso_path import walk_lasso_support_regions
from .si_seqfs import forward_selection_path, fs_conditioning_interval, interval_FS
from .si_types import SIConfig, SIDecomposition, SIResult

__all__ = [
    "calc_lava_score",
    "SelectionAdapter",
    "CGSelectionAdapter",
    "SelectiveInference",
    "SI_DV",
    "NaiveTest",
    "FSNaiveTest",
    "DataSplittingTest",
    "SIConfig",
    "SIDecomposition",
    "SIResult",
    "parametric_intervals_FS_nested",
    "cg_parametric_intervals_FS_nested",
    "walk_lasso_support_regions",
    "forward_selection_path",
    "fs_conditioning_interval",
    "interval_FS",
]
