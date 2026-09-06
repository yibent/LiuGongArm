"""One fast attempt, then one model attempt; cancellation never escalates."""
from dataclasses import replace
import time


class FastPathFailure(RuntimeError):
    pass


def run_cascade(request, fast, enhanced, recover, event, *, attempts=None):
    # The caller retains the history even when recovery or model execution fails.
    attempts = [] if attempts is None else attempts

    def attempt(backend, function, task):
        started = time.perf_counter()
        row = {"backend": backend}
        attempts.append(row)
        event("attempt_started", backend=backend)
        try:
            value = function(task)
            row["ok"] = bool(value["physical_success"])
            return value
        except Exception as error:
            row.update(ok=False, error=str(error))
            raise
        finally:
            row["elapsed_s"] = time.perf_counter() - started
            event("attempt_finished", **row)

    if request.enhanced:
        value = attempt("models", enhanced, request)
        return value, request.route(), attempts
    try:
        value = attempt("official_pick_place", fast, request)
        if value["physical_success"]:
            return value, request.route(), attempts
        raise FastPathFailure("Fast task failed physical evaluation")
    except InterruptedError:
        raise
    except RuntimeError as error:
        event("model_fallback", reason=str(error))
        # Recovery chooses re-observation/regrasp versus placement-only based on
        # the physical task state. Never release a successfully held part here.
        recovered = recover(request)
        task = replace(request, mode="enhanced")
        value = attempt("models", recovered or enhanced, task)
        route = task.route()
        if recovered: route["grasp"] = "official_pick_place"
        return value, route, attempts
