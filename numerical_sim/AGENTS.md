# Testing policy

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for setup, implementation conventions and
the checks required before review. Run commands from `numerical_sim/`.

- During iteration, run focused tests for the changed behavior. Do not rerun the
  full suite after every edit. Documentation-only changes need documentation and
  whitespace checks rather than runtime tests.
- Once the patch stabilizes, run the affected subsystem tests. For numerical
  changes, cover eager and CUDA-graph execution, finite state, environment isolation
  and masked reset where relevant.
- Changes to shared solvers, batching, CUDA capture, stream ordering or reset
  require the full runtime suite before review.
- Before handing off a finished patch, complete the applicable contribution checks.
  For code changes, run the full configured suite once after the final relevant edit:
  `I4H_DATA="$PWD/data" uv run pytest tests/runtime tests/packages -q`.
  Set `I4H_DATA` to the actual production asset directory if it differs.
- Run rendering, recording, Isaac Lab, teleoperation, parity and performance checks
  when the change affects those paths. Physical devices require separate hardware
  qualification; fake-device checks do not establish hardware readiness.
- Reuse the pinned environment and compilation caches. Avoid concurrent GPU test
  runs that introduce contention, especially for performance measurements.
- Report passed, failed, skipped and unavailable checks clearly. Missing CUDA or
  production assets can skip substantial coverage; report those limits explicitly.
  Do not weaken assertions or numerical tolerances to make tests pass.

After checks pass, rerun or broaden them only when subsequent changes, failures or
unresolved concerns justify it. Keep timing estimates out of this policy; use
`pytest --durations=20` to identify slow tests when needed.
