"""Temporary wall-clock profiler for the parametric_nested pipeline.

Every timer is a single-line pair::

    _t = tp.start("name")
    <work>
    tp.stop(_t)

plus optional counters::

    tp.tick("name")

``tp.report()`` prints an accumulated breakdown (seconds, call count,
percent of total measured time).  ``tp.reset()`` clears before a fresh run.

To remove the instrumentation later: delete this file and every line that
mentions ``tp.`` (the import plus the ``start``/``stop``/``tick`` lines and
any ``@tp.timed`` decorator lines).  No indentation is touched, so the
erase is purely mechanical.
"""

from __future__ import annotations

import time

from collections import defaultdict

_TIMES: dict[str, list[float, int]] = defaultdict(lambda: [0.0, 0])
_COUNTS: dict[str, int] = defaultdict(int)


def reset() -> None:
    _TIMES.clear()
    _COUNTS.clear()


def start(name: str):
    return name, time.perf_counter()


def stop(tok) -> None:
    name, t0 = tok
    _TIMES[name][0] += time.perf_counter() - t0
    _TIMES[name][1] += 1


def tick(name: str, n: int = 1) -> None:
    _COUNTS[name] += n


def timed(name: str):
    """Decorator timing every call of a function (safe on exceptions)."""
    def deco(fn):
        def wrapper(*args, **kwargs):
            tok = start(name)
            try:
                return fn(*args, **kwargs)
            finally:
                stop(tok)
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return deco


def report() -> None:
    total = sum(secs for secs, _ in _TIMES.values())
    print("\n================ TIMING REPORT ================")
    print(f"{'section':<46} {'seconds':>10} {'calls':>8} {'%total':>7}")
    for name, (secs, n) in sorted(_TIMES.items(), key=lambda kv: -kv[1][0]):
        pct = 100.0 * secs / total if total > 0 else 0.0
        print(f"{name:<46} {secs:10.3f} {n:8d} {pct:7.2f}")
    print(f"{'TOTAL':<46} {total:10.3f} {'':>8} {100.0:7.2f}")
    if _COUNTS:
        print("\ncounters:")
        for name, n in sorted(_COUNTS.items()):
            print(f"  {name}: {n}")
    print("================================================")