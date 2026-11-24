# i4h-isaaclab

Version 0.9.0, import `i4h_isaaclab`, Python `>=3.12,<3.13`.
Dependencies: i4h-physics, Torch and Pillow; compatible Isaac Lab runtime. The adapter (`newton_manager`, `newton_solver` / `lab30`, `views`, `env`) builds and steps physics cases with nothing else; cases with a `procedure:` (rewards, terminations, the procedure tasks) need i4h-assessment, the `apps` extra (`uv pip install './numerical_sim/isaaclab[apps]'`). Add the `surgical`/`endoluminal` extras for those domains. Install locally with `uv pip install ./numerical_sim/isaaclab`
or use the workspace lockfile for development.

```python
import i4h_isaaclab.tasks  # registers the preserved Gym task IDs
# Import within a compatible Isaac Lab process.
```

## Isaac Lab compatibility

Isaac Lab is installed from source. The qualified stack is Newton
`7f0c13af6e4dd75253709c31dc65c0edcaf22b44` (1.9.0.dev0), Warp 1.17.0,
MuJoCo/MuJoCo-Warp 3.17.0, Lab main `4cfdfcfa52d7cb64eb83ab293b2fcfc92452aaf1`,
and patched Lab 3.0 `e53cd288727a5ca3dae7187d1d415b4d32b67e11`; the exact pins are in
`qualification/c30-{core,main,lab30}.json`.
The package supplies main pure/coupled solver adapters and a narrow release 3.0
manager bridge over the same authoring, binding and numerical core. Lab owns
allocation, stepping and capture; component bindings own views and episode
restoration. The alternate backend and `NewtonSim` have been removed.

`I4hComponent.save(state, control)` and `restore(state, control)` copy the i4h-owned rows of the live State and
Control to and from another pair of the host's Model. Rigid body state, host time and assessment are the host's to
restore. Restore last: write the host's own state, call the simulation's `forward()`, then restore the i4h rows,
because Lab runs the solver's cache reset for worlds written through its asset API at the next forward or step.

Patched 3.0 additionally needs the explicit
[cloner compatibility delta](../docs/newton18-lab30-cloner.patch): Newton uses public
`surface_label` / `volume_label` and their corresponding private particle starts.
The supplied checkout remains unchanged; qualification uses an isolated worktree
based on e53cd288 with that patch. See [adapter evidence](../docs/stage-b-host.md)
and [Stage B status](../docs/stage-b.md) for current checks and remaining gates.
Inside Lab use its `usd-exchange` provider; do not install a competing `usd-core`
package over its `pxr` module.

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
the particles. Use `-Direct` for the tiled camera (`rgb`/`depth`), staggered
episodes and session replay; the manager env has none of these.

Isaac Lab's manager-based step calls `reset_buf.nonzero()` (a host sync) every step, which `I4hEnv` avoids with its mask
reset. At 64 envs the measured difference is below 2% (`docs/performance.md`).

Teleoperation devices are not configured by default; `i4h_isaaclab_standalone.devices.configure_teleop(cfg)`
installs the `i4h` device (requires `i4h-standalone`).

Production assets are supplied separately with `--data` or `I4H_DATA`; missing assets
fail explicitly. No production geometry or native device binaries are packaged.
See the numerical workspace README and architecture document for setup and package boundaries.
Licensed under Apache-2.0; see this package's LICENSE and NOTICE.

Native Versius robot tasks and their scripted, PPO and teleoperation commands are described in
[the Versius guide](../docs/versius.md). They use surgical contact-case YAMLs and the shared launchers.
