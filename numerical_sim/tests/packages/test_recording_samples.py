# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Recording samples keep their values and nested tensor layout across Lab container APIs."""
from types import SimpleNamespace

import torch
from i4h_isaaclab.recording import append_sample, stack_samples


def test_samples_copy_live_views_and_keep_nested_sample_dimensions():
    episode = SimpleNamespace(data={})
    live = torch.tensor([1.0, 2.0])
    append_sample(episode, 'actions', live)
    live.add_(3)
    append_sample(episode, 'actions', live)
    append_sample(episode, 'devices/pad', {'position': live.clone()}, clone=False)
    append_sample(episode, 'initial_episode_length', torch.tensor([7]))
    stack_samples(episode.data)
    torch.testing.assert_close(episode.data['actions'], torch.tensor([[1., 2.], [4., 5.]]))
    assert episode.data['devices']['pad']['position'].shape == (1, 2)
    assert episode.data['initial_episode_length'].tolist() == [[7]]
