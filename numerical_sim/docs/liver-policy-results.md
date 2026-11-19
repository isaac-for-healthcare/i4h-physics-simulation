# Experimental liver policy results

Best learned policy: **bc_best, 94% full-cycle success** over 100 reserved episodes, 20 per target. All policies used identical starts verified by exact tissue/action hashes. The 80% initial experimental milestone is judged on the complete physical cycle.

| Policy | Full-cycle success |
| --- | ---: |
| controller | 100% (100/100) |
| bc_best | 94% (94/100) |
| ppo_best | 0% (0/100) |
| ppo_latest | 0% (0/100) |

Collected 1,000 complete successful demonstrations (200 per target) in 131.7 seconds; split 800/200 by complete episode. BC selected the lowest validation action MSE. PPO used RSL-RL 5.5.1 with the approved rollout and optimization parameters. The first 25-minute PPO segment and subsequent continuation are archived; reward v2 corrects wrong-patch acquisition bonuses and saturates lift potential at 20 mm. Those fixes leave the demonstration rewards, BC action labels, physical evaluator and tissue mechanics unchanged.

The policy controls Cartesian increments and grip; each target retains its demonstrated orientation. A +20 mm destination is accepted only after one continuous second within the preserved 2 mm tolerance. The original-particle retention, release/relaxation and 20-second deadline rules remain enforced. The numerical mechanics already contain inverted tetrahedra and large soft-attachment drift; all measured diagnostics are retained. This experiment establishes task success on the current mechanics, not absolute-force validation or generalization to a whole organ edge.

Validation: **30 focused tests passed; 58 regression tests passed, 3 skipped**. The skips are the unavailable Isaac Lab host and two unsupported Newton global-WORLD checks. All 1,000 demonstration files passed independent timing, action-bound, episode-end and reserved-seed integrity checks. The isolated dependency setup and checkpoint/resume commands were exercised. Both Slang videos passed full decode.

- Detailed report, per-target results and physical diagnostics (local: `artifacts/liver-policy-20261119/results.md`)
- Selected best policy (local: `artifacts/liver-policy-20261119/best_policy.pt`)
- BC checkpoint (local: `artifacts/liver-policy-20261119/bc_best.pt`) and PPO checkpoint (local: `artifacts/liver-policy-20261119/ppo_best.pt`)
- Dataset manifest (local: `artifacts/liver-policy-20261119/dataset/manifest.json`)
- Training curves (local: `artifacts/liver-policy-20261119/training_curves.png`)
- BC Slang video (local: `artifacts/liver-policy-20261119/bc_best_slang.mp4`) and PPO Slang video (local: `artifacts/liver-policy-20261119/ppo_best_slang.mp4`)
- [Setup, resume and interactive playback commands](../examples/liver_grasp/README.md)

All implementation changes remain uncommitted. Original frozen qualification evidence and other worktrees were preserved.

These artifacts are retained locally and are not distributed in Git. A public download location has not been assigned.
