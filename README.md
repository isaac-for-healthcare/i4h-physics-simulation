# Isaac for Healthcare - Medical Physics Simulation

Isaac Sim–compatible simulation tools for modeling anatomy and healthcare robotics, powered by NVIDIA [Newton](https://github.com/newton-physics/newton), [Warp](https://github.com/NVIDIA/warp), and generative video models.

This repository holds numerical simulators under [`numerical_sim/`](./numerical_sim), a generative simulator under [`neural_sim/`](./neural_sim), and the deprecated catheter solver under [`physics_simulation/`](./physics_simulation). Use them to prototype endoluminal and laparoscopic procedures, train robotics policies, and generate synthetic surgical video.

## Available Components

### Numerical Simulation

**Status:** Available in [`numerical_sim/`](./numerical_sim) as six Python packages, version 0.9.0, sharing one lockfile.

| Package | Import | Provides |
| ------- | ------ | -------- |
| i4h-physics | `i4h_physics` | Scene, bodies, staged systems, solver and stepper; physical binding and reset |
| i4h-surgical | `i4h_surgical` | Surgical cases, instruments, appearance and grasp measurements |
| i4h-endoluminal | `i4h_endoluminal` | Vessel anatomy, catheter control and binding, reach measurements |
| i4h-assessment | `i4h_assessment` | Shared executor, history and reward products |
| i4h-isaaclab | `i4h_isaaclab` | Isaac Lab hosts, gym tasks and tiled cameras |
| i4h-standalone | `i4h_standalone` | The `i4h-sim` app: rendering, recording, XR, devices and panels |

Available in-tree today:

- Tetrahedral soft tissue with XPBD deformation, grasp, contact, stretch/breaking and thermal stages
- Deformable organ surfaces and organ–organ contact
- Particle-based fluids (PBF) with suction
- YAML-authored cases: cholecystectomy (`chole_tet_med`, `chole_grasp`), `liver`, `tissue_contact` and the two-world demo `two_worlds`
- Endoluminal catheter cases in an aorta (`aorta_static`, `aorta_shell`, `aorta_centreline` and their `_reach` variants), with an endoscope view from the catheter tip
- Isaac Lab tasks `I4h-TwoWorlds-Direct`, `I4h-Aorta-Direct`, `I4h-ReachTarget-Direct`, `I4h-GraspTarget-Direct` and `I4h-Chole-Direct`
- Render backends `noop` (headless), `slang`, `gl` and `xr` (Meta Quest 3 through CloudXR; see the [XR guide](./numerical_sim/docs/xr.md))
- Optional haptic devices (MiniMou, and Follou through its separately installed SDK), and controller vibration in XR

Not available yet:

- Bronchoscope and other non-catheter endoluminal devices
- Airway and colon cases

Production assets (meshes, textures, centrelines and USD) are included under `numerical_sim/data/` and licensed under Apache-2.0. `scripts/fetch_assets.py` can also install them from a copy you supply and checks each file against [`docs/asset-manifest.json`](./numerical_sim/docs/asset-manifest.json); see [numerical_sim/data/README.md](./numerical_sim/data/README.md).

```bash
cd numerical_sim
uv sync --locked --extra render --extra gl --group dev
uv run i4h-sim chole_tet_med --data ./data
```

To embed i4h physics in an Isaac Lab scene, see the [component contracts](./numerical_sim/docs/component-contracts.md) and the [Isaac Lab compatibility notes](./numerical_sim/isaaclab/README.md#isaac-lab-compatibility).

### Endoluminal Catheter Solver (deprecated)

**Status:** Deprecated. The solver at [`physics_simulation/endoluminal/catheter-vasculature-solver`](./physics_simulation/endoluminal/catheter-vasculature-solver) will be replaced by the numerical simulator in [`numerical_sim/`](./numerical_sim) (`i4h-endoluminal`) in an upcoming release. Until then it remains available here, and the i4h-workflows catheter scenes still install it.

**What it provides:** A Newton-based endoluminal solver (catheter and related devices) using Cosserat rod / XPBD soft-body physics on NVIDIA Warp and Newton.

- Cosserat / XPBD catheter insertion through static or deformable vessel walls
- Track-guided insertion along a fixed guide axis
- Vessel containment via SDF or mesh-edge collision paths
- Bendable / steerable distal tip with configurable rest curvature
- Deformable vessel walls via a branching centerline Cosserat tree with two-way contact and surface skinning
- Optional Isaac Lab coupling that steps rigid tools and the catheter in one Newton substep

Vessel geometry is authored outside this repository; see the [solver README](./physics_simulation/endoluminal/catheter-vasculature-solver/README.md) for how to feed it a vessel mesh and insertion track.

### Generative Physics Simulation (Cosmos-H-Dreams)

**Status:** Available as a git submodule at [`neural_sim/cosmos_h_dreams`](./neural_sim/cosmos_h_dreams).

Real-time action-conditioned surgical video simulation via WebRTC, built on [FlashDreams](https://github.com/NVIDIA/flashdreams). Given a conditional first frame and a live stream of instrument action vectors, the model rolls forward generated frames and streams them to a browser or Meta Quest headset.

Key features:

- Offline batch inference from a JSON manifest
- Interactive WebRTC control (keyboard browser or Meta Quest / WebXR)
- Multiple runner configs (chunk size, 2- vs 4-step schedule, VAE vs light TAE decoder)

```bash
git submodule update --init --recursive neural_sim/cosmos_h_dreams
cd neural_sim/cosmos_h_dreams

# Build and run — see the submodule README for checkpoints and full flags
docker build -t cosmos-h-dreams:latest docker/
# Offline example:
# uv run flashdreams-run cosmosHDreams-chunk3-vae-vae --input-json ...
```

Full setup, configs, and system requirements: [Cosmos-H-Dreams README](https://github.com/isaac-for-healthcare/Cosmos-H-Dreams/blob/main/README.md).

## Repository Layout

```text
numerical_sim/            # Six Warp / Newton packages and the i4h-sim app
├── physics/  surgical/  endoluminal/  assessment/  isaaclab/  standalone/
├── docs/                 # Architecture, component contracts, XR, performance, examples
├── examples/  scripts/  tests/
└── data/                 # Apache-2.0 production assets
neural_sim/
└── cosmos_h_dreams/      # Git submodule — generative surgical video sim
physics_simulation/
└── endoluminal/
    └── catheter-vasculature-solver/   # Deprecated catheter + vasculature solver (Newton / Warp)
```

## Getting Started

1. Clone this repository (with submodules for generative sim):

   ```bash
   git clone --recurse-submodules https://github.com/isaac-for-healthcare/i4h-physics-simulation.git
   cd i4h-physics-simulation
   ```

   If you already cloned without submodules:

   ```bash
   git submodule update --init --recursive
   ```

2. Pick a component:

   - **Numerical sim (surgical and endoluminal):** follow [numerical_sim/README.md](./numerical_sim/README.md)
   - **Endoluminal catheter solver (deprecated):** follow [physics_simulation/endoluminal/catheter-vasculature-solver/README.md](./physics_simulation/endoluminal/catheter-vasculature-solver/README.md)
   - **Generative sim:** follow the [Cosmos-H-Dreams README](https://github.com/isaac-for-healthcare/Cosmos-H-Dreams/blob/main/README.md)

## Requirements

Shared / typical prerequisites (exact versions depend on the component):

| Requirement | Notes |
| ----------- | ----- |
| OS | Linux (x86_64; `numerical_sim` also supports aarch64) |
| Python | `numerical_sim`: `>=3.12,<3.13` (development pins 3.12.12); Cosmos-H-Dreams: 3.12+; the catheter solver supports 3.10+ |
| GPU | NVIDIA GPU; Cosmos-H-Dreams recommends ≥12 GB VRAM. `numerical_sim` also steps on CPU for its examples and package tests |
| Driver / CUDA | Cosmos-H-Dreams: driver **R580+** (CUDA 13.x). Newton / Warp stacks need a CUDA-capable driver matching the installed toolkit |
| Simulation stack | `numerical_sim` pins Warp 1.17.0 and Newton 1.6.1. Its Isaac Lab tasks target Isaac Lab `release/3.0.0` at `6864dd1` |
| Container | Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for the generative sim image |

Dependencies are declared in [`numerical_sim/pyproject.toml`](./numerical_sim/pyproject.toml) (with its `uv.lock`) and the catheter solver's [`pyproject.toml`](./physics_simulation/endoluminal/catheter-vasculature-solver/pyproject.toml).

## Security

See [SECURITY.md](./SECURITY.md). Do not report security vulnerabilities through public GitHub issues.

## Support

This repository is under active development (experimental). For questions and support, open an issue in the GitHub repository.

## License

The root [LICENSE](./LICENSE) (Apache-2.0) applies except where a part carries its own terms; [NOTICE](./NOTICE) lists each part. The `numerical_sim` packages and original production assets in `numerical_sim/data/` are licensed under Apache-2.0. Cosmos-H-Dreams code is primarily Apache-2.0 with model weights under the NVIDIA Open Model License — see [its LICENSE](https://github.com/isaac-for-healthcare/Cosmos-H-Dreams/blob/main/LICENSE). Third-party notices are in [THIRD-PARTY-NOTICES](./THIRD-PARTY-NOTICES), and contributions follow [CONTRIBUTING.md](./CONTRIBUTING.md).
