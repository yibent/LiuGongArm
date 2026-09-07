"""One fast attempt, then one model attempt; cancellation never escalates."""
from dataclasses import replace
import time
from mr_liu.arena.failure import PlacementSpaceUnavailable, LocalizationFailure


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
            if not row['ok']: row['evaluation'] = value
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
        error = FastPathFailure("Fast task failed physical evaluation")
        error.evaluation = value
        raise error
    except (InterruptedError, PlacementSpaceUnavailable, LocalizationFailure):
        raise
    except RuntimeError as error:
        if request.mode == "basic":
            raise
        if getattr(error,'evaluation',{}).get('released'):
            # A released, misplaced part needs a new semantic observation.
            # Preserve its physical failure for supervision; don't immediately
            # replay its pregrasp visual reference and hide it behind not-found.
            event('placement_review_required',evaluation=error.evaluation)
            return error.evaluation, request.route(), attempts
        event("model_fallback", reason=str(error))
        # Recovery chooses re-observation/regrasp versus placement-only based on
        # the physical task state. Never release a successfully held part here.
        recovered = recover(request)
        task = replace(request, mode="enhanced")
        value = attempt("models", recovered or enhanced, task)
        route = task.route()
        if recovered: route["grasp"] = "official_pick_place"
        return value, route, attempts
