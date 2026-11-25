# Stage B — Warp 1.17.0 captured-grid correction

The [native patch](warp117-hashgrid-capture.patch) fixes the reproduced capture
metadata alias without changing the Warp version, i4h mechanics, case capacities
or dependency pins. It applies to Warp tag `v1.17.0`, commit
`f4c57f26f1e3936a89afd283e39fcabf6d548dc7`.

The stock 1.17.0 binary still has the defect. Version `1.17.0` alone does not
identify the corrected binary: retain the patch and wheel/native-library hashes
when installing or qualifying this stack.

## Running the upstream regression

`tests/runtime/test_hashgrid_capture.py` is skipped by default because it checks
an upstream Warp defect and the stock dependency does not include the local fix.
Run it explicitly when qualifying a patched or newer Warp build:

```bash
I4H_WARP_HASHGRID_TESTS=1 uv run --no-sync pytest tests/runtime/test_hashgrid_capture.py -q
```

The opt-in preserves all nine tests and their assertions. A default skip does not
establish that the installed Warp binary is fixed or qualify captured simulations.

## Implementation

`hash_grid_update_device_impl()` now uses an ordinary local descriptor.
`hash_grid_rebuild_device()` passes that descriptor by value and the destination
grid address to its existing final `compute_cell_offsets` kernel. Thread zero
publishes the descriptor while the other threads compute the same offsets as
before. CUDA graph kernel arguments retain each recorded build's values.

This covers both different grids and different builds of the same grid, including
changed cell widths. Merely using one host staging descriptor per grid would not
cover multiple captured builds of that grid. An empty rebuild launches one lane
so its descriptor is published too.

The shared host-to-device staging copy is removed. For populated grids, there is
no additional kernel, allocation, host readback or synchronization. The ordering
of i4h systems, solves and neighbor queries is unchanged. CPU grid behavior is
unchanged; the CPU-only native stub is updated for the internal function signature.
The public HashGrid API and descriptor layout are unchanged.

## Local artifact and environments

Evidence and artifacts are in `.tmp_tests/stage-b-hashgrid-fix-20261105/`:

- `warp117-original.tar`: exact tag snapshot, before applying the patch.
- `warp-source/`: isolated patched source and build products.
- `patch-manifest.json`, `binaries.json`, `installed-warp-delta.json`: source,
  toolchain, patch and binary provenance.
- `wheels/warp_lang-1.17.0-py3-none-linux_x86_64.whl`: corrected local wheel.
- `core-env`, `lab30-env`, `main-env`: private environments using that native
  binary; other package code is shared read-only with the preserved environments.
  Package metadata is copied so environment tools can enumerate it.

The original `.tmp_tests/newton18/*-env` environments remain unchanged. They are
still available for pre-change measurements and failed reproductions.

The build uses CUDA 12.9.1, release settings without fast math, libmathdx 0.3.1,
and the original 1.17.0 wheel's unchanged `warp-clang.so` for CPU compilation.
All common installed Python/header sources match the original after normalizing
line endings, except the two patched native sources. The native library changes
from `a05f6fa483b43ea4c7d6bba17e75d9ee600b0d22a2a162ae962b51598bd674cb`
to `eb4bda4537f7c7a35ef4640d89fbfba96cc9b2b786fdf1591535ea420ec8b6eb`.

The local build requires glibc 2.38 or newer. Its wheel is tagged `linux_x86_64`;
no manylinux portability claim is made. Warp's setup initially emitted a
`manylinux_2_28` filename; that artifact is retained as build evidence only. The
corrected local wheel has identical native payloads and an appropriate local tag.

For a selected environment on this qualified host, install the local wheel after
normal dependency installation (from `numerical_sim`):

```bash
uv pip install --python /path/to/environment/bin/python --no-deps --reinstall \
  .tmp_tests/stage-b-hashgrid-fix-20261105/wheels/warp_lang-1.17.0-py3-none-linux_x86_64.whl
```

Do not install over the preserved pre-change environments used by timing. A
subsequent dependency reinstall from the stock index can replace this source
patch; check the native hash rather than the version string alone.

The build is reproducible by applying the patch to the exact tag, fetching that
tag's `deps/cuda-toolkit-deps.packman.xml` with tag `cuda-12`, and running:

```bash
python build_lib.py --cuda-path _build/target-deps/cuda --no-standalone -j 4
```

Copy `warp/bin/warp-clang.so` from the preserved 1.17.0 wheel, then use `uv build
--wheel` and `wheel tags --platform-tag linux_x86_64` for this local build.
`build.log`, `cuda-fetch.log`, `wheel-build-uv.log` and `wheel-retag.log` retain the
exact build and packaging output. No dependency source or binary was edited in
the preserved environments.

## Qualification

The final analytic regression tests report **8 failed, 1 passed** with the stock
binary, and **9 passed** with the correction. They cover all three scalar types,
grouped/ungrouped queries, live position/group edits, separate grids, separate
captures and two builds in one capture with different widths, and empty queries.
The three original diagnostic reproductions now pass unchanged. Upstream Warp's
HashGrid suite passes **47 tests**, covering CPU and CUDA; the CPU-only native
stub also compiles successfully.

Early test-harness errors are retained separately: an integer-literal reference
compile error, generic argument typing, and an invalid expectation that
`hash_grid_point_id` bounds-checks the live point count. They are not counted as
captured-grid failures. The final tests use valid public queries and the same
analytic expectations on both binaries.

The original standalone/host test still alternates captured simulations, with
its seed, action sequence, step counts and tolerance unchanged. Its failure path
now saves complete tensors and names the differing products, so another failure
cannot be reduced to a truncated pytest tensor representation.

Exclusive timing and cadence remain separate gates. The timing
runner accepts `--replacement-env-root` so the replacement and cadence use the
corrected environments while pre-change keeps the original binary; it records
the selected roots and each measured worker's native Warp hash.

The prepared exclusive command is:

```bash
python3 scripts/qualify_stage_b_timing.py \
  --replacement-env-root .tmp_tests/stage-b-hashgrid-fix-20261105 \
  --out .tmp_tests/stage-b-hashgrid-fix-20261105/exclusive-final
```

Use a fresh output directory for any attempt. A busy-GPU guard launches no
measurement. Contaminated or failed measured runs remain evidence and are not
retried for a better sample. Chole's reference policy remains awaiting the user's
disposition; this correction does not promote its candidate or begin Stage C.
