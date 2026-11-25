# Historical Versius qualification artifacts

The standalone runtime and launchers have been replaced by canonical surgical
cases and shared Isaac Lab tooling. Use the [Versius guide](../../isaaclab/versius.md)
for setup, validation, training, playback and teleoperation commands.

`RESULTS.md`, `results.json`, `stack.json`, `requirements.lock`, `reuse.json`,
`life_size_reference.json`, `NOTICE` and `LICENSE` preserve historical results,
source attribution and stack evidence. Commands inside `RESULTS.md` describe the
historical application and are retained as evidence. They are no longer launch
entrypoints. `replay.json` remains a reusable raw-position input fixture.

New qualification evidence is kept separately; the historical summary follows.

## Historical consolidation results and limits

The [2026-11-17 consolidation report](https://github.com/isaac-for-healthcare/i4h-physics-simulation/blob/8020858babd4ad3c2d1cebff7c75bcb2f2bd904c/numerical_sim/docs/versius-consolidation-results.md)
records 1,377 passed and 46 skipped configured repository tests and 61 focused
passes. Block and original-chole scripted fixtures each achieved 10/10 in both
eager and graph execution on two worlds. Life-size chole remained **0/10**;
its matching scripted gate is still required before training. One PPO update and
50 playback steps per profile established smoke coverage, not a trained policy.

Selected reset preserved peer state and command/history bytes. Independent chole
rollouts showed 0.00403 tissue drift; that observation was not a passed tight
equality gate. GL replay/recording passed on that stack; physical Follou and
OpenHaptics operation was checked only with fakes and still needs hardware
qualification.

Matched warmed measurements reported graph physics block 6.405 → 6.780 ms
(+5.9%) and original chole 12.136 → 11.837 ms (−2.5%). Offscreen replay total
loops changed 11.148 → 9.838 ms and 17.965 → 16.470 ms respectively. These
measurements exclude real devices and windowed interaction; they establish no
50 Hz guarantee. Newton mesh/shape host transfers and video framebuffer readback
remain. The campaign used a locally patched Warp binary and external Lab/Newton
sources; these historical passes do not qualify stock Warp HashGrid capture or
current source. Detailed failures, stack identity and measurements remain in the
pinned report; frozen example files above retain their original bytes.
