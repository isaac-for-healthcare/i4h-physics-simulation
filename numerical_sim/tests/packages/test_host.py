# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys


def test_host_without_domains_assessment_torch_or_hosts():
    code = """
import importlib, pkgutil, sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('i4h_surgical', 'i4h_endoluminal', 'i4h_assessment'):
            raise ModuleNotFoundError(fullname, name=fullname)
        if fullname.split('.')[0] in ('torch', 'isaaclab', 'i4h_isaaclab', 'i4h_standalone', 'i4h_isaaclab_standalone', 'slangpy', 'pyglet'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
import i4h_host
for child in pkgutil.walk_packages(i4h_host.__path__, 'i4h_host.'):
    importlib.import_module(child.name)
from i4h_host.composition import read_case, load_case, build_procedure
case = {'sim': {'device': 'cpu'}, 'assets': {}, 'systems': []}
assert read_case(case)['sim']['device'] == 'cpu'
assert build_procedure(case, None) is None
try:
    read_case({'procedure': {'name': 'test'}})
except ModuleNotFoundError as e:
    assert e.name == 'i4h_assessment' and 'i4h-host[assessment]' in str(e)
else: raise AssertionError('assessment silently omitted')
"""
    subprocess.run([sys.executable, '-c', code], cwd='/tmp', check=True)
