# Experimental state-based liver grasp policy

Run from `numerical_sim/` in the `i4h-liver-grasp` worktree. Training uses the current
numerical mechanics, not an Isaac Lab application. The adapter implements the
RSL-RL 5.5.1 `VecEnv`/`TensorDict` interface. Existing simulation environments and
package pins stay untouched. `setup_training.sh` creates a local overlay using the
preserved numerical interpreter and explicitly pinned added dependencies.

```bash
examples/liver_grasp/setup_training.sh
examples/liver_grasp/train.sh benchmark --worlds 16
examples/liver_grasp/train.sh benchmark --worlds 8
examples/liver_grasp/train.sh benchmark --worlds 1
examples/liver_grasp/train.sh qualify --worlds 16 --episodes 25
examples/liver_grasp/train.sh collect --worlds 16 --episodes 1000 --minutes 25
examples/liver_grasp/train.sh bc --minutes 10
examples/liver_grasp/train.sh evaluate --checkpoint artifacts/liver-policy-20261119/bc_best.pt
examples/liver_grasp/train.sh ppo --worlds 16 --minutes 60
examples/liver_grasp/train.sh resume --worlds 16 --minutes 30
examples/liver_grasp/train.sh evaluate
examples/liver_grasp/train.sh evaluate --checkpoint artifacts/liver-policy-20261119/ppo_best.pt
examples/liver_grasp/train.sh play --worlds 1 --minutes 60 \
  --checkpoint artifacts/liver-policy-20261119/bc_best.pt
examples/liver_grasp/train.sh video --checkpoint artifacts/liver-policy-20261119/bc_best.pt
```

`--data` selects the production assets; `--output` selects a separate experiment
folder. Defaults point to the preserved sibling numerical assets and local
`artifacts/liver-policy-20261119/`. Leave the original frozen grasp campaign intact.
To resume BC, pass `--checkpoint .../bc_latest.pt` to the `bc` stage.

The Cartesian action is clipped to [-1,1] and its three-vector length is capped at
one; it increments the commanded root by at most 1 mm per 30 Hz tick. Action four
maps [-1,1] to jaw grip [0,1]. Quaternion remains the target's demonstrated
orientation; energy/fire remain zero. Acquisition is sampled after both 60 Hz
physics steps, preserving original particle/sphere/rest identity and sticky loss.

The 57 observations contain current relative target/jaw geometry, the surface
normal, motion since the preceding control tick, destination error, instrument
quaternion/grip, actual jaw angle/held flag, acquisition/retention counts and radial
residuals, mechanism counts, previous action and physical task phases/timers.
Geometry uses metres times 100, displacement uses metres times 3000, radial
residual uses metres times 1000, counts divide by 64 and timers divide by the task
deadline (hold streak divides by 60). These fixed physical scales are the complete
normalization; no running statistics or teacher commands enter observations.

Reset restores every declared numerical continuation array from a fresh, strictly
identified settled bundle for the actual environment count, then clears only that
row's probe/task buffers. The instrument is placed 30 mm in front of the demonstrated
capture geometry, with a seeded +/-5 mm position offset per axis. Masked initial
kinematics also populate previous sphere positions, avoiding artificial velocity.
Tissue is never moved to fake a start. The GUI controller uses this same adapter.

The scorer is separate from the controller and reward. It requires intended-patch
acquisition, half-second initial retention, a 20 mm destination, one continuous
second inside a 2 mm destination sphere, clearing both counts and active bindings
within half a second of opening, then a full second open without reacquisition.
The complete attempt, including relaxation, has a 20 s deadline. Timeout is an
absorbing task failure, so RSL-RL's `time_outs` is false rather than bootstrapping
through a failed attempt. Inversions and attachment drift are reported; mechanics
are unchanged and remain experimental.

Demonstrations retain complete successful and failed episodes. Only successful
complete episodes enter the stratified 80/20 episode split. `dataset/source/`
freezes collection scripts; the manifest records examples, seeds, source/dependency
identity and the canonical snapshot. Requested/applied actions and terminal next
observations are separate NPZ arrays; termination reasons are in the manifest.

BC selects the lowest held-out action MSE. PPO uses two 128-unit ELU layers, 64
steps, four epochs/four minibatches, LR 3e-4, clip .2, gamma .99 and GAE .95. The
initial normalized action standard deviation is .02, to avoid immediately opening
an acquired grasp at the demonstrated .52 grip. No teacher action is mixed into
learned rollout or evaluation actions. Checkpoints retain actor, optimizer,
configuration, fixed normalization, provenance and Python/NumPy/Torch/CUDA RNG.
PPO also retains critic and complete algorithm optimizer state. Resume currently
starts fresh canonical episodes at the saved schedule cursor; it does not claim
bitwise reproduction of the interrupted rollout.

Final evaluation assigns each reserved episode to a fixed canonical environment
row. `*_starts.json` records exact tissue/action hashes so controller, BC and PPO
starts can be compared. PPO validation uses separate seeds (88201); final comparison
uses the 100 reserved seeds (981304), 20 per target. Best BC, best PPO validation
and latest PPO checkpoints remain separate even if learning regresses.

`resume` archives the starting checkpoint and previous evidence under `sessions/`, then merges prior and new curves, validation scores and completed training episodes. The underlying `ppo --checkpoint` command restores training state directly; use `resume` to retain history automatically.

Reward revision v2 rejects acquisition bonuses for wrong-patch captures and saturates lift-progress potential at 20 mm. Physical scoring is unchanged. Equal PPO validation scores retain the more recent checkpoint. The initial v1 PPO segment is preserved in the experiment history.

`audit_dataset.py --verify-starts` reconstructs canonical row IDs from the exact seeded dispatch order and saved control lengths. `dataset/provenance_rows.json` retains that mapping without changing the collection manifest or action labels; all 1,000 completions were matched, and ten restored start observations matched bit-for-bit.

Final evaluation records attachment and inversion diagnostics after every 60 Hz physics step. Demonstration and video diagnostics are sampled at control ticks where any vectorized episode ends; their geometric maxima have that sampling limit. Acquisition/retention and success scoring still run after every physics step in both modes. Release latency is quantized to physics samples.

Equivalent explicit `uv` commands, from `numerical_sim/`:

```bash
export PYTHONPATH="$PWD/physics/src:$PWD/surgical/src:$PWD/standalone/src:$PWD/isaaclab/src:$PWD/assessment/src:$PWD/endoluminal/src"

# Continue PPO with archived history, optimizer and RNG restoration.
uv run --no-project --python examples/liver_grasp/.venv-train/bin/python \
  python examples/liver_grasp/resume.py --worlds 16 --minutes 30

# Interactive Slang playback of the selected best learned policy.
uv run --no-project --python examples/liver_grasp/.venv-train/bin/python \
  python examples/liver_grasp/train.py play --worlds 1 --minutes 60 \
  --checkpoint artifacts/liver-policy-20261119/best_policy.pt
```
