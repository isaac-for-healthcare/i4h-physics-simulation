# Standalone split validation

The coordinated changes use source baseline `37f03006e68b6e6a1a1ac4d07c952d82ab468f56`.
Detailed local evidence is in the sibling standalone repository's
`docs/validation/results.md`, including raw logs, archive hashes and source/test relocation maps.

The Lab-enabled upstream full suite recorded 1226 passed, 4 failed and 14 skipped. One failure
was a CPU/CUDA test placement mismatch; after an explicit CPU load its complete recording/replay
and named-view test passed. The three remaining numerical observations are baseline clipping,
scene excitation and the 40 mm mounted-reset error. Standalone recorded 236 passed, 1 baseline
hook failure and 4 XR skips; bridge/Versius recorded 37 passed and 1 XR skip.

No physics, domain, assessment, production-asset bytes, physical tolerances or frozen
qualification reports were changed. Final standalone Git pins and lock require the separately
authorized publication of the upstream host change. This is not a release.
