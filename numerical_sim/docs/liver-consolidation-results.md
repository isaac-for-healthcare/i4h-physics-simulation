# Manager-env and liver consolidation

The manager-env branch incorporates liver source `7edecea49204539ca54cf9da736b5bcf597dadff` by merge, preserving the separate Versius consolidation `cfd45a2`. The package-data conflict retains both liver and Versius cases. Source recordings and experiment-local files were not added.

The bridge now owns graph-safe post-physics callback registration and selected-world host reset/restore hooks previously carried as a Workflows patch. Numerical sampling precedes the host reducer at each physics step. Ordinary IDs receive physical reset, Lab events and history/procedure initialization. Restored IDs receive Lab bookkeeping and explicit host restoration without overwriting restored numerical history. Terminal hooks run before reset bookkeeping.

The optional grasp adapter imports the surgical domain only when requested and reports a missing optional package explicitly. This fixes the package-boundary regression introduced with the liver source.

## Validation on 2026-11-19

- Liver grasp, hosted lifecycle and Versius tests excluding the PPO module: **55 passed** in the numerical workspace environment, with production assets supplied through `I4H_DATA`.
- Versius PPO tests in `.venvs/isaaclab-current`: **2 passed**. The base numerical environment does not install RSL-RL and cannot collect this module.
- Package boundaries, hosted lifecycle, continuation, runtime bindings/components, volumetric grasp and instruments: **86 passed, 3 skipped** in the numerical workspace environment. The skips retain existing unsupported-world-layout/optional-host exclusions.
- All pre-commit hooks passed for the bridge and lifecycle-test changes.

An initial broader package/runtime run was interrupted after 171 passes and three skips. It reported the eager surgical import above; the corrected boundary test passed in the completed targeted run. This is not a completed full-suite qualification or a repeat of Stage C's frozen reference campaign.

## Remaining boundaries

The [Stage C closure](stage-c-closure.md) retains its two deferred P2 defects and its documented exclusions. The liver mechanics remain uncalibrated. Historical example training commands still require their separately retained datasets/recordings; those artifacts are not distributed by this source merge. Historical policy results do not establish successful full-cycle PPO training in Workflows.

This branch and PR target the internal numerical repository. Public distribution requires a reviewed source sync onto the separately rooted public repository, a public commit, and a coordinated Workflows dependency repin. The Workflows bridge patch remains a compatibility measure for its previous pin until that repin.

## GitHub CI follow-up

After the source update, lint passed. CI identified missing copyright headers in the liver/Versius additions and links to their untracked local artifacts; the follow-up commit adds the headers and labels those artifact locations as local paths.

The numerical smoke job was already failing at the previous public PR head `b921eca`. The current CPU-runner job invokes package tests that also request CUDA and production assets; it reported 256 failures, 356 passes and 16 skips. This infrastructure/suite split requires separate repair and is not hidden by new skips or weaker assertions. The markdown-link job also checks frozen Stage C reports whose `.tmp_tests` evidence lives outside Git. Their bytes and evidence hashes remain unchanged; an archive/publication policy for those references remains open.
