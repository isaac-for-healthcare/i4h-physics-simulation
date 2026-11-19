# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from i4h_isaaclab.versius.qualification import check_task_metadata


def test_matching_metadata_and_actionable_mismatch():
    saved = {'profile': 'block', 'dt': 1 / 60, 'patch_ids': [7, 2, 6, 8]}
    check_task_metadata(dict(saved), saved, 'Checkpoint')
    with pytest.raises(ValueError, match="Checkpoint.*dt:.*profile:"):
        check_task_metadata({**saved, 'profile': 'chole', 'dt': 1 / 120}, saved, 'Checkpoint')
    assert saved['profile'] == 'block'
    assert saved['dt'] == 1 / 60
