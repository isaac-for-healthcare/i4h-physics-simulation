# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create an isolated source-overlay environment without resolving or upgrading dependencies."""
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-env', type=Path, required=True, help='Python 3.12 seed environment with pinned Warp and host dependencies')
    parser.add_argument('--lab', type=Path, required=True, help='external Isaac Lab source checkout')
    parser.add_argument('--newton', type=Path, required=True, help='external Newton source checkout')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / '.venvs/isaaclab-current'
    seed, lab, newton = (p.expanduser().resolve() for p in (args.seed_env, args.lab, args.newton))
    for path in (seed / 'bin/python', lab / 'source/isaaclab', newton / 'newton'):
        if not path.exists():
            parser.error(f'missing explicit source: {path}')
    if target.exists():
        parser.error(f'{target} already exists; left unchanged')
    subprocess.run(['uv', 'venv', '--python', str(seed / 'bin/python'), str(target)], check=True)
    site = next(seed.glob('lib/python3.12/site-packages'))
    dest = target / 'lib/python3.12/site-packages'
    subprocess.run(['cp', '-a', '--reflink=auto', str(site) + '/.', str(dest)], check=True)
    paths = [newton, *(root / name / 'src' for name in
             ('physics', 'surgical', 'endoluminal', 'isaaclab', 'assessment', 'host')),
             *sorted(p for p in (lab / 'source').iterdir() if p.is_dir())]
    # Keep third-party editable hooks, including any locally patched binary package.
    # Explicit source overlays must take precedence over copied seed paths.
    (dest / 'i4h-current.pth').write_text("import sys; sys.path[:0] = " + repr(list(map(str, paths))) + "\n")
    print(target / 'bin/python')


if __name__ == '__main__':
    main()
