# Isaac for Healthcare - Medical Physics Simulation

Isaac Sim–compatible simulation tools for modeling anatomy and healthcare robotics, powered by NVIDIA [Newton](https://github.com/newton-physics/newton), [Warp](https://github.com/NVIDIA/warp), and generative video models.

This repository aggregates physics and generative simulators under [`physics_simulation/`](./physics_simulation). Use them to prototype endoluminal and laparoscopic procedures, train robotics policies, and generate synthetic surgical video.

## Available Components

### Endoluminal Physics Simulator

**Status:** *Partially ported.* [`physics_simulation/endoluminal/catheter-vasculature-solver`](./physics_simulation/endoluminal/catheter-vasculature-solver) holds the catheter and vasculature solver. The wider interventions simulator is still being migrated.

**What it provides:** A Newton-based endoluminal solver (catheter and related devices) using Cosserat rod / XPBD soft-body physics on NVIDIA Warp and Newton.

Available in-tree today:

- Cosserat / XPBD catheter insertion through static or deformable vessel walls
- Track-guided insertion along a fixed guide axis
- Vessel containment via SDF or mesh-edge collision paths
- Bendable / steerable distal tip with configurable rest curvature
- Deformable vessel walls via a branching centerline Cosserat tree with two-way contact and surface skinning
- Optional Isaac Lab coupling that steps rigid tools and the catheter in one Newton substep

Not ported yet:

- Endoscopic camera mode that follows the catheter tip
- Packaged scenes such as aorta and airways (USD / mesh assets), and YAML scene authoring
- Bronchoscope and other non-catheter devices

Vessel geometry is authored outside this repository; see the [solver README](./physics_simulation/endoluminal/catheter-vasculature-solver/README.md) for how to feed it a vessel mesh and insertion track.

### Soft Tissue & Fluid Surgical Simulator

**Status:** *Not yet ported.* The [`physics_simulation/surgical`](./physics_simulation/surgical) folder is a placeholder. Source currently lives in [omnisurg](https://github.com/isaac-for-healthcare) and will be migrated here.

**What it will provide:** A Newton-compatible soft-tissue and fluid simulator for laparoscopic / robotic surgery, including instrument–tissue interaction and optional haptic device output.

Planned capabilities (from the upstream omnisurg codebase):

- Tetrahedral and hex soft-tissue bodies with XPBD deformation, grasp, contact, stretch/breaking, and thermal stages
- Deformable organ surfaces and directed organ–organ contact
- Procedure packages (for example cholecystectomy tet cases) with YAML-authored assets, instruments, and solvers
- Particle-based fluids (PBF) and cloth demos
- Optional haptic output (e.g. MiniMou) with force caps and calibration presets
- Rendering backends including noop (headless) and RTX-oriented paths

Upstream quick reference (until the port lands):

```bash
# From the omnisurg repository
uv run omnisurg --config examples/minimal_case.yaml
uv run omnisurg --config examples/chole_tet_case_med.yaml
```

Haptic presets and hardware checklist: see omnisurg’s `HAPTICS.md`.

### Generative Physics Simulation (Cosmos-H-Dreams)

**Status:** Available as a git submodule at [`physics_simulation/cosmos_h_dreams`](./physics_simulation/cosmos_h_dreams).

Real-time action-conditioned surgical video simulation via WebRTC, built on [FlashDreams](https://github.com/NVIDIA/flashdreams). Given a conditional first frame and a live stream of instrument action vectors, the model rolls forward generated frames and streams them to a browser or Meta Quest headset.

Key features:

- Offline batch inference from a JSON manifest
- Interactive WebRTC control (keyboard browser or Meta Quest / WebXR)
- Multiple runner configs (chunk size, 2- vs 4-step schedule, VAE vs light TAE decoder)

```bash
git submodule update --init --recursive physics_simulation/cosmos_h_dreams
cd physics_simulation/cosmos_h_dreams

# Build and run — see the submodule README for checkpoints and full flags
docker build -t cosmos-h-dreams:latest docker/
# Offline example:
# uv run flashdreams-run cosmosHDreams-chunk3-vae-vae --input-json ...
```

Full setup, configs, and system requirements: [Cosmos-H-Dreams README](https://github.com/isaac-for-healthcare/Cosmos-H-Dreams/blob/main/README.md).

## Repository Layout

```text
physics_simulation/
├── endoluminal/          # Catheter + vasculature solver (Newton / Warp)
├── surgical/             # Placeholder — port from omnisurg (pending)
└── cosmos_h_dreams/      # Git submodule — generative surgical video sim
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

2. Use the component that is available today:

   - **Generative sim:** follow [physics_simulation/cosmos_h_dreams/README.md](https://github.com/isaac-for-healthcare/Cosmos-H-Dreams/blob/main/README.md)
   - **Endoluminal catheter sim:** follow [physics_simulation/endoluminal/catheter-vasculature-solver/README.md](./physics_simulation/endoluminal/catheter-vasculature-solver/README.md)
   - **Surgical Newton sim:** not in-tree yet — use the upstream omnisurg repository until that folder is populated

## Requirements

Shared / typical prerequisites (exact versions depend on the component):

| Requirement | Notes |
| ----------- | ----- |
| OS | Linux (x86_64) |
| Python | 3.12+ (Cosmos-H-Dreams); the endoluminal solver supports 3.10+; omnisurg currently pins `3.12.12` |
| GPU | NVIDIA GPU; Cosmos-H-Dreams recommends ≥12 GB VRAM |
| Driver / CUDA | Cosmos-H-Dreams: driver **R580+** (CUDA 13.x). Newton / Warp stacks need a CUDA-capable driver matching the installed toolkit |
| Container | Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for the generative sim image |

Newton-based endoluminal and surgical stacks additionally depend on NVIDIA Newton and Warp (see the in-tree `physics_simulation/endoluminal/catheter-vasculature-solver/pyproject.toml`, or the upstream omnisurg `pyproject.toml` for the surgical stack).

## Security

See [SECURITY.md](./SECURITY.md). Do not report security vulnerabilities through public GitHub issues.

## Support

This repository is under active development (experimental). For questions and support, open an issue in the GitHub repository.

## License

Licensing varies by component. Cosmos-H-Dreams code is primarily Apache-2.0 with model weights under the NVIDIA Open Model License — see [physics_simulation/cosmos_h_dreams/LICENSE](https://github.com/isaac-for-healthcare/Cosmos-H-Dreams/blob/main/LICENSE). Upstream omnisurg licenses apply until that package is ported and documented here.
