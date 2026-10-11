"""Temporary wall-clock profiler for the parametric_nested pipeline.

Every timer is a single-line pair::

    _t = tp.start("name")
    <work>
    tp.stop(_t)

plus optional counters::

    tp.tick("name")

``tp.report()`` prints an accumulated breakdown plus a reconciliation with
the wall-clock span from ``tp.reset()`` to the report::
- ``excl`` is each section's *exclusive* time (its own wall time with any
      nested instrumented work removed), so the exclusive rows sum to the
      wall time actually covered by timers.

    - ``unaccounted`` is the remainder (``WALL - measured``): the
      sequential code not covered by any timer, plus any instrumented overlap.

    - Together they sum to ``WALL`` (``reset`` -> ``report``), the program's
      real elapsed time, so the report now reconciles with the final runtime.


Nesting is handled with a per-thread stack, so a section's time is attributed
to its enclosing timer in the same thread.  Concurrent spans (e.g., the two
worker threads of the threaded OT trace) are *not* nested, so their times
overlap: the exclusive sum then exceeds ``WALL`` and the ``unaccounted`` row goes
negative — a deliberate, visible flag of that overlap (the wrapper ``L+R``
timer and the per-call ``fn`` timers measure the same wall time).  The ``%wall``
column is each section's share of the final runtime (100% is the whole run).

To remove the instrumentation later: delete this file and every line that
mentions ``tp.`` (the import plus the ``start``/``stop``/``tick`` lines and
any ``@tp.timed`` decorator lines).  No indentation is touched, so the
erase is purely mechanical.
"""

from __future__ import annotations

import threading
import time

from collections import defaultdict

_TIMES: dict[str, list[float, int]] = defaultdict(lambda: [0.0, 0])   # inclusive secs, call count
_EXCLUSIVE: dict[str, float] = defaultdict(float)
_COUNTS: dict[str, int] = defaultdict(int)
_LOCAL = threading.local()
_WALL0: float | None = None


def reset() -> None:
    global _WALL0
    _TIMES.clear()
    _EXCLUSIVE.clear()
    _COUNTS.clear()
    _WALL0 = time.perf_counter()
    _LOCAL.stack = []


def start(name: str):
    stack = getattr(_LOCAL, "stack", None)
    if stack is None:
        stack = _LOCAL.stack = []
    stack.append(name)
    return name, time.perf_counter()


def stop(tok) -> None:
    name, t0 = tok
    d = time.perf_counter() - t0
    _TIMES[name][0] += d
    _TIMES[name][1] += 1
    stack = getattr(_LOCAL, "stack", [])
    if stack and stack[-1] == name:
        stack.pop()
        if stack:   # enclosing timer in this thread
            _EXCLUSIVE[stack[-1]] -= d
    _EXCLUSIVE[name] += d


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
    wall = (time.perf_counter() - _WALL0) if _WALL0 is not None else 0.0
    measured = sum(_EXCLUSIVE.values())
    print("\n================ TIMING REPORT ================")
    print(f"{'section':<46} {'excl':>10} {'incl':>10} {'calls':>8} {'%wall':>7}")
    for name, (secs, n) in sorted(_TIMES.items(), key=lambda kv: -_EXCLUSIVE[kv[0]]):
        excl = _EXCLUSIVE.get(name, 0.0)
        pct = 100.0 * excl / wall if wall > 0 else 0.0
        print(f"{name:<46} {excl:10.3f} {secs:10.3f} {n:8d} {pct:7.2f}")
    m_pct = 100.0 * measured / wall if wall > 0 else 0.0
    print(f"{'--- measured (exclusive, no nesting)':<46} {measured:10.3f} {'':>10} {'':>8} {m_pct:7.2f}")
    unacc = wall - measured
    print(f"{'--- unaccounted (sequential / overlap)':<46} {unacc:10.3f}")
    print(f"{'WALL (reset -> report)':<46} {wall:10.3f} {'':>10} {'':>8} {100.0:7.2f}")
    if _COUNTS:
        print("\ncounters:")
        for name, n in sorted(_COUNTS.items()):
            print(f"  {name}: {n}")
    print("================================================")