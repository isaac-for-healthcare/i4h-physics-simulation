# i4h-isaaclab

Version 0.9.0, import `i4h_isaaclab`, Python `>=3.12,<3.13`.
Dependencies: i4h-physics, Torch and Pillow; compatible Isaac Lab runtime. The adapter (`newton_manager`, `physics`, `views`, `env`) builds and steps physics cases with nothing else; cases with a `procedure:` (rewards, terminations, the procedure tasks) need i4h-assessment, the `apps` extra (`uv pip install './numerical_sim/isaaclab[apps]'`). Add the `surgical`/`endoluminal` extras for those domains. Install locally with `uv pip install ./numerical_sim/isaaclab`
or use the workspace lockfile for development.

```python
import i4h_isaaclab.tasks  # registers the preserved Gym task IDs
# Import within a compatible Isaac Lab process.
```

## Isaac Lab compatibility

Isaac Lab is not a pip dependency; install it from source. This package targets
`isaac-sim/IsaacLab` tag `v3.0.0` (GA, on branch `release/3.0.0`) at `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`
(torch 2.11; `i4h_isaaclab.lab.QUALIFIED`), with Newton 1.6.1 and Warp 1.17.0. The runtime test suite passes on that
revision; the physics qualification (numerical parity, including the Newton 1.6.0 -> 1.6.1 change) is
still pending on it. Inside Lab, install the domain packages without their `usd` extra: Lab's `usd-exchange`
provides `pxr`, and `usd-core` would overwrite it (both ship `pxr`). It relies on Lab APIs that older
3.0 checkouts lack: `NewtonManager._invalidate_graph`, `isaaclab_newton.cloner.newton_builder_world_hook`,
`DirectRLEnvCfg.compute_final_obs` and `DirectRLEnv._reset_envs_from_buffer`. Importing
`i4h_isaaclab.env` or `i4h_isaaclab.newton_manager` against a Lab without them fails with an error
naming the target revision. Other Lab revisions are untested until the runtime suite and the
physics qualification are rerun on them.

## Custom tasks

Subclass `I4hEnvCfg` (case, `observe`, `camera`, `decimation`, `episode_length_s`, `gravity_from_case`, ...) and
register it with `entry_point="i4h_isaaclab.env:I4hEnv"`:

```python
import gymnasium as gym
from i4h_isaaclab.env import I4hEnvCfg
from isaaclab.utils.configclass import configclass

@configclass
class MyTaskCfg(I4hEnvCfg):
    case: str = "chole_tet_med"
    decimation: int = 4

gym.register("My-I4h-Task-v0", entry_point="i4h_isaaclab.env:I4hEnv",
             kwargs={"env_cfg_entry_point": f"{__name__}:MyTaskCfg"})  # as i4h_isaaclab.tasks registers its own

# inside an Isaac Lab process:
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
env = gym.make("My-I4h-Task-v0", cfg=load_cfg_from_registry("My-I4h-Task-v0", "env_cfg_entry_point"))
```

For custom rewards, terminations or resets, subclass `I4hEnv` and register that class instead. Overridden
`_get_rewards`/`_get_dones` read the zero-copy views (`obs_views`, `state_views`, `task_rows`) directly: the physics
manager orders torch after every physics call. Code outside the step calls `env.sync_views()` first. An overridden
`_reset_idx` runs on automatic resets too. Isaac Lab reset events (`EventCfg`, `mode="reset"`) run after the physical
reset, and the procedure resets after them.

## Manager-based env

Every task id also exists as `-Manager` (`I4h-ReachTarget-Manager`, ...): `i4h_isaaclab.manager.I4hManagerBasedRLEnv`,
an Isaac Lab `ManagerBasedRLEnv` over the same case. It has the same `policy` and `task` observations, the same
normalised actions (`workspace` 0.5), the same reward and dones, and uses the same PPO configs, so a checkpoint
trained on `-Direct` runs on `-Manager`. `tests/runtime/test_manager_env.py` checks this parity: bitwise for the
rod reach rollouts, and within the direct env's own run-to-run spread for tet tissue, whose float-atomic contact
sums differ between any two runs.

The pieces are ordinary manager terms for a host task that composes its own, such as a robot beside the tissue or
an Arena embodiment:

- `I4hActionTermCfg`;
- the observation functions `i4h_policy` and `i4h_task`;
- `procedure_reward`;
- the termination functions `procedure_terminated` and `procedure_timeout`;
- `I4hHostMixin`, mixed in front of any `ManagerBasedRLEnv` subclass. It constructs the env inside the component and
  resets physics, Lab's reset events, then the procedure, as `I4hEnv` does.

The scene needs the `I4H_ROOT` entry (an authored env root, no spawn), so that Lab replicates the worlds that hold
the particles. Use `-Direct` for the tiled camera (`rgb`/`depth`), the `i4h` fallback backend, staggered
episodes and session replay; the manager env has none of these.

Isaac Lab's manager-based step calls `reset_buf.nonzero()` (a host sync) every step, which `I4hEnv` avoids with its mask
reset. At 64 envs the measured difference is below 2% (`docs/performance.md`).

Teleoperation devices are not configured by default; `i4h_standalone.devices.configure_teleop(cfg)`
installs the `i4h` device (requires `i4h-standalone`).

Production assets are supplied separately with `--data` or `I4H_DATA`; missing assets
fail explicitly. No production geometry or native device binaries are packaged.
See the numerical workspace README and architecture document for setup and package boundaries.
Licensed under Apache-2.0; see this package's LICENSE and NOTICE.
