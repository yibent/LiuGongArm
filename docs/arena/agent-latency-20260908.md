# Agent latency audit — 2026-09-08

Evidence: server BusAgent events at 10:46–10:54 CST, downloaded before subsequent scene resets. Successful model responses include planning and focused visual calls, not speech synthesis. Usage is provider-reported token totals, including repeated/cached input; it is not a billing estimate.

| User input | Successful model requests | Reported tokens | Time summed across those requests |
|---|---:|---:|---:|
| Place red block into red container | 3 | 21,548 | 18.5 s |
| Then place other blocks in blue container | 28 | 258,876 | 149.1 s |
| What about my subsequent task? | 2 | 14,624 | 31.8 s |
| `。` (punctuation-only ASR output) | 16 | 151,921 | 75.8 s |

The second task also had one failed provider request. These figures cannot all be attributed to model autonomy: the record includes controller restarts and obsolete visual references.

## Confirmed defects and changes

- Punctuation-only ASR final text became a new intent and the planner inferred a manipulation from history. Filter before STT publishes an intent, with defensive filtering in task intake and dialogue.
- Colloquial status and upcoming-queue questions missed the immediate route and entered the full robot tool loop. Answer these from the durable queue, including work still in intake planning. Mixed queries containing new commands continue to reach the planner.
- Planning observations used `/api/command` and were rejected with `Panda is busy`. `/api/observe` now enqueues only the sensor capture on the simulation thread; inference runs on the HTTP worker, without claiming or changing the physical command, target or holding state. Persistent tracking retains its original skill path.
- Invalid arguments, ambiguous selectors and stale references were retried verbatim. Reject missing selector information before dispatch and return evidence-changing failures for repair rather than repeat the same physical request.
- Model telemetry now includes returned tool names. The task request counter also counts failed provider attempts rather than silently counting only successful responses.

## Validation

- Backend: 357 tests passed, 1 database integration test skipped. After the request-count change, the 58 relevant task-engine/Mastra tests passed and the backend build completed.
- Python: 7 service/read-only observation tests passed.
- GPUFree public URL: three status queries returned in 662, 406 and 507 ms. All used no LLM and created no goals. Punctuation-only input produced no reply and no goal.
- During a live physical command, concurrent red-block and blue-container collection queries completed in 0.869 s and 1.698 s without `Panda is busy`. One found a red block; the other returned zero visible instances. The physical command remained running, demonstrating queue independence rather than guaranteed detection accuracy.

## Limits of acceptance

The first natural-language pick/place attempt produced a complete plan in one successful response, after an 8-second primary-provider timeout and a 14.8-second secondary response. Execution then failed because the rebuilt vision environment lacked a CUDA NVRTC library. The environment-maintenance task repaired that library and tested SAM3 inference.

Later grasp/place tests overlapped user scene switches and a perception cache reset. They encountered missing observation files and are invalid for end-to-end performance acceptance. The acceptance goal was cancelled; no further physical test or scene reset was initiated by this task. A full uninterrupted manipulation-speed acceptance is still required. External model channel latency/timeouts also remain measurable limitations.
