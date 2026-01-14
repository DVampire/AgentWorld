from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError


def mutliprocess(
    tasks: Sequence[Tuple[Callable[..., Any], tuple, dict]],
    *,
    max_workers: int = 8,
    timeout_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Run tasks concurrently using threads.
    - No retry.
    - Fail fast: stop as soon as any task fails.
    - Ordered output: results are returned in input order.

    Args:
        tasks: List of (fn, args, kwargs).
        max_workers: Number of worker threads.
        timeout_s: Per-task timeout (None means no timeout).

    Returns:
        results: Ordered list of completed task results.
        Each element is a dict:
            {
              "index": int,
              "ok": bool,
              "value": any,        # present if ok == True
              "error": str,        # present if ok == False
              "error_type": str,   # present if ok == False
              "elapsed_s": float
            }

        Note:
            Once a failure occurs, remaining unfinished tasks are cancelled.
            Tasks already running may still finish, but their results are ignored.
    """
    if max_workers <= 0:
        raise ValueError("max_workers must be greater than 0")

    def _task_wrapper(
        index: int,
        fn: Callable[..., Any],
        args: tuple,
        kwargs: dict,
    ) -> Dict[str, Any]:
        """Execute a single task and measure elapsed time."""
        start = time.perf_counter()
        value = fn(*args, **kwargs)
        return {
            "index": index,
            "ok": True,
            "value": value,
            "elapsed_s": time.perf_counter() - start,
        }

    results_by_index: Dict[int, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index: Dict[Any, int] = {}

        # Submit all tasks
        for index, (fn, args, kwargs) in enumerate(tasks):
            future = executor.submit(_task_wrapper, index, fn, args, kwargs)
            future_to_index[future] = index

        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result(timeout=timeout_s)
            except TimeoutError as e:
                result = {
                    "index": index,
                    "ok": False,
                    "error": repr(e),
                    "error_type": type(e).__name__,
                    "elapsed_s": 0.0,
                }
                # Fail fast: cancel remaining tasks
                for f in future_to_index:
                    if f is not future:
                        f.cancel()
                results_by_index[index] = result
                break
            except BaseException as e:
                result = {
                    "index": index,
                    "ok": False,
                    "error": repr(e),
                    "error_type": type(e).__name__,
                    "elapsed_s": 0.0,
                }
                # Fail fast: cancel remaining tasks
                for f in future_to_index:
                    if f is not future:
                        f.cancel()
                results_by_index[index] = result
                break

            results_by_index[index] = result

    # Return results in the original input order (only completed ones)
    return [
        results_by_index[i]
        for i in range(len(tasks))
        if i in results_by_index
    ]


# ---------------- Example ----------------
if __name__ == "__main__":
    import time

    def work(x: int) -> int:
        time.sleep(0.05)
        if x % 7 == 0:
            raise RuntimeError(f"bad input: {x}")
        return x * x

    task_list = [(work, (i,), {}) for i in range(1, 21)]
    output = mutliprocess(task_list, max_workers=8, timeout_s=1.0)

    print(output)
