"""Pin the BLAS/OpenBLAS thread count for the SI pipeline.

The parametric LP trace (`si_parametric.trace_parametric_bases`) calls
``np.linalg.inv``/``np.linalg.solve`` on small (``nt+nv``)-sized matrices a
few hundred times, and ``H[:, non_basis]`` gathers.  On this environment
(OpenBLAS ``MAX_THREADS=128, NO_AFFINITY=1, USE_OPENMP=0``) a 214x214
``np.linalg.inv`` costs ~12 ms single-shot and ~72 ms under the two trace
worker threads, purely from BLAS thread spawn/join overhead -- the actual
work is ~0.2 MFLOP.  Forcing a single BLAS thread drops that to ~1-2 ms and
takes the n=200 lava demo end-to-end from ~21 s to ~6 s with an identical
p-value.

``threadpoolctl`` cannot see this build (its ``threadpool_info()`` is empty),
so this module sets the ``*_NUM_THREADS`` environment variables *and* pokes
the already-loaded OpenBLAS/MKL shared library through ctypes, which is the
only mechanism that works after numpy has imported the BLAS.

Set ``SI_BLAS_THREADS=0`` to disable, or to a positive integer to choose the
count (default 1).
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys

_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_PINNED = False


def _blas_dll_candidates() -> list[str]:
    """Best-effort list of shared BLAS libraries loaded or loadable here."""
    import numpy

    exe_dir = os.path.dirname(sys.executable)
    bases = [
        os.path.dirname(numpy.__file__),
        exe_dir,
        os.path.join(exe_dir, "..", "Library", "bin"),
        os.path.join(exe_dir, "..", "Library", "lib"),
    ]
    out: list[str] = []
    for base in bases:
        for pat in ("**/*openblas*.dll", "**/mkl_rt*.dll", "**/*blas*.so*"):
            out += glob.glob(os.path.join(base, pat), recursive=True)
    return [p for p in dict.fromkeys(out) if os.path.exists(p)]


def _set_num_threads(n: int) -> None:
    """Call ``openblas_set_num_threads``/``MKL_Set_Num_Threads`` if present."""
    for path in _blas_dll_candidates():
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for sym in ("openblas_set_num_threads", "openblas_set_num_threads64_",
                    "goto_set_num_threads"):
            fn = getattr(lib, sym, None)
            if fn is None:
                continue
            try:
                fn(ctypes.c_int(n))
            except Exception:  # noqa: BLE001
                pass
        for sym in ("MKL_Set_Num_Threads", "mkl_set_num_threads"):
            fn = getattr(lib, sym, None)
            if fn is None:
                continue
            try:
                fn(ctypes.c_int(n))
            except Exception:  # noqa: BLE001
                pass


def pin_blas_threads(n: int | None = None) -> None:
    """Force the process BLAS thread count (idempotent).

    ``n`` defaults to ``SI_BLAS_THREADS`` (1); ``0`` disables pinning.
    Environment variables are set for libraries that have not loaded yet,
    then the currently loaded OpenBLAS/MKL is poked directly.
    """
    global _PINNED
    if _PINNED:
        return
    if n is None:
        raw = os.environ.get("SI_BLAS_THREADS", "1")
        try:
            n = int(raw)
        except ValueError:
            n = 1
    if n and n > 0:
        for var in _ENV_VARS:
            os.environ.setdefault(var, str(n))
        try:
            _set_num_threads(n)
        except Exception:  # noqa: BLE001
            pass
    _PINNED = True
