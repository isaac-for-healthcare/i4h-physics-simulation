# Exclusive Stage B measurements, 2026-11-05

Evidence root (relative to `numerical_sim`): `.tmp_tests/stage-b-exclusive-20261105/`.
The user authorized terminating the three identified workloads. The two remaining
catheter processes exited on SIGTERM; the vascular process had already exited.
`process-release.json` records that action. Earlier busy-device attempts remain
preserved and launched no timing workers.

First complete valid campaign: **passed**. 478 process observations; no unrelated compute processes observed. All 20 workers used captured execution, 20 warmup steps, 200 measured steps and zero controls. No measured run was repeated.

## Step timing

| Workload | Worlds | Pre-change Lab 3.0 ms | Replacement Lab 3.0 ms | Lab 3.0 change | Main ms | Main change vs historical host reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tissue_contact | 1 | 0.362472 | 0.362425 | -0.013% | 0.361327 | +0.024% |
| tissue_contact | 64 | 0.529243 | 0.530414 | +0.221% | 0.528337 | -0.001% |
| aorta_static | 1 | 7.789783 | 7.784532 | -0.067% | 7.794817 | -2.113% |
| aorta_static | 64 | 10.560836 | 10.529775 | -0.294% | 10.515898 | +2.392% |

The unchanged bound is at most +5% in each comparison. Main uses its existing historical host-specific reference, not a paired pre-change main measurement. Reference paths and hashes are recorded in `decision.json`.

## Export overhead

| Workload | Worlds | Lab 3.0 exports-on ms | Lab 3.0 overhead | Main exports-on ms | Main overhead |
| --- | ---: | ---: | ---: | ---: | ---: |
| tissue_contact | 1 | 0.362379 | -0.013% | 0.361453 | +0.035% |
| tissue_contact | 64 | 0.530281 | -0.025% | 0.529073 | +0.139% |
| aorta_static | 1 | 7.752744 | -0.408% | 7.757248 | -0.482% |
| aorta_static | 64 | 10.506096 | -0.225% | 10.475392 | -0.385% |

Export overhead is reported separately; negative differences are measured variation, not a claim that exports improve performance.

## Construction and memory

| Version | Workload | Worlds | Construction seconds | Retained Warp bytes | Peak Warp bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| prechange-lab30 | tissue_contact | 1 | 0.761318 | 6,064 | 16,779,500 |
| prechange-lab30 | tissue_contact | 64 | 0.757693 | 310,669 | 16,899,767 |
| prechange-lab30 | aorta_static | 1 | 0.832143 | 5,855,154 | 16,822,188 |
| prechange-lab30 | aorta_static | 64 | 0.852257 | 18,163,527 | 20,486,429 |
| lab30 | tissue_contact | 1 | 0.671185 | 6,192 | 16,779,500 |
| lab30 | tissue_contact | 64 | 0.684498 | 318,861 | 16,899,767 |
| lab30 | aorta_static | 1 | 0.737890 | 5,858,226 | 16,822,188 |
| lab30 | aorta_static | 64 | 0.756993 | 18,360,135 | 20,683,037 |
| main | tissue_contact | 1 | 0.990825 | 6,192 | 16,779,500 |
| main | tissue_contact | 64 | 1.031812 | 318,861 | 16,899,767 |
| main | aorta_static | 1 | 1.063753 | 5,858,226 | 16,822,188 |
| main | aorta_static | 64 | 1.076861 | 18,360,135 | 20,683,037 |
| lab30-exports | tissue_contact | 1 | 0.676480 | 6,192 | 16,779,500 |
| lab30-exports | tissue_contact | 64 | 0.679783 | 318,861 | 16,899,767 |
| lab30-exports | aorta_static | 1 | 0.749044 | 5,858,258 | 16,822,188 |
| lab30-exports | aorta_static | 64 | 0.758841 | 18,362,183 | 20,683,037 |
| main-exports | tissue_contact | 1 | 1.011262 | 6,192 | 16,779,500 |
| main-exports | tissue_contact | 64 | 1.027520 | 318,861 | 16,899,767 |
| main-exports | aorta_static | 1 | 1.068292 | 5,858,258 | 16,822,188 |
| main-exports | aorta_static | 64 | 1.085321 | 18,362,183 | 20,683,037 |

Every measured replay has **zero allocation growth**. The replacement Newton particle grid stays absent; q0 and inverse-mass sidecars remain unchanged. Construction and peak measurements are separate from steady-state timing.

## Cadence

`test_app_loop.py::test_physics_rate_independent_of_render`: **passed**, exit 0, uncontaminated=True. Its single execution followed all timing workers in the same monitored campaign. See `cadence.log` and `cadence.xml`.

Native libraries and runtime source hashes match the qualified stack. No physics/runtime source, dependency pin, timing threshold, or cadence assertion was changed.
