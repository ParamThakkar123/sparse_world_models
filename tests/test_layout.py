from __future__ import annotations

import numpy as np

from models import StateLayout, infer_num_objects_from_state_dim, reshape_object_pose, reshape_object_velocity


def test_infer_num_objects_from_state_dim_handles_five_object_layout() -> None:
    layout = StateLayout(num_objects=5)
    assert infer_num_objects_from_state_dim(layout.state_dim) == 5


def test_reshape_helpers_follow_layout() -> None:
    layout = StateLayout(num_objects=5)
    state = np.arange(layout.state_dim, dtype=np.float32)[None, :]

    object_pose = reshape_object_pose(state, num_objects=5)
    object_velocity = reshape_object_velocity(state, num_objects=5)

    assert object_pose.shape == (1, 5, 3)
    assert object_velocity.shape == (1, 5, 6)
    np.testing.assert_array_equal(object_pose.reshape(-1), state[0, layout.object_pose_slice])
    np.testing.assert_array_equal(object_velocity.reshape(-1), state[0, layout.object_velocity_slice])
