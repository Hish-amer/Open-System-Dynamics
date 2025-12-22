from __future__ import annotations 
    # keeps type hints are stored as strings and evaluated later.

import time
from dataclasses import dataclass
from typing import Callable, TypeVar, Any, Optional
import statistics

# =============================================================================

T = TypeVar("T") 
    # filler type to placehold in the benchmark function for the output of
    # some arbitrary function that the benchmark is taking in as an arg.
    # we could call it anything, but "T" is convention, standing for TYPE.

@dataclass(frozen=True)
class BenchStats:
    repeats: int
    min_sec: float
    median_sec: float
    max_sec: float
    mean_sec: float

# =============================================================================

def time_call(
        fn: Callable[..., T],
        /,
            # anything above this "/" is a positional argument to force 
            # it into first pstn and avoid it being called as a keyword arg.
        *args: Any,
            # allows the function to take any set of positional arguments
        **kwargs: Any
            # allows the function to take keywrod args as well to deal with
            # mixed arg scenarios.
    ) -> tuple[T, float]:
            # returns whatever T, i.e. the function tye is and seconds in float

    """
    Run fn(*args, **kwargs) once and return (result, elapsed_seconds).
    Works for any callable.
    """
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0)

# =============================================================================
def benchmark(
        fn: Callable[..., T],
        /, *args: Any,
        repeats_bnch: int = 30,
        warmup_bnch: int = 3,
        **kwargs: Any
    ) -> tuple[T, BenchStats]:

    """
    Run fn multiple times and return (last_result, timing_stats).
    Warmup helps avoid first-call effects, so this takes an average
    and median of how long the function takes given it is repeated
    "repeats_bnch" times, which is more faithful than just running once
     as we do in the time_call function.
    """
    last: Optional[T] = None
        # so last can be type T, so placeholder for the return typeof the fn 
        # being fed into the arg, or none.
    times: list[float] = []
    
    for _ in range(warmup_bnch):
        last = fn(*args, **kwargs)

    for _ in range(repeats_bnch):
        t0 = time.perf_counter()
        last = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
            # so this loop stores how long each iteration of the function
            # took and appends it to a times list, outside of the first "warmer"
            # cycles that make sure first call effects are ignored.
            # it then returns an average, median, min and max of that time set.

    assert last is not None

    return last, BenchStats(
        repeats=repeats_bnch,
        min_sec=min(times),
        median_sec=statistics.median(times),
        max_sec=max(times),
        mean_sec=statistics.mean(times),
    )
