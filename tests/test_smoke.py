"""Smoke test for the Wuji hand adaptor node.

Exercises the :class:`unienv_wuji.WujiHandActor` node **directly**
(constructed with ``connect=False``) without going through ``WorldEnv`` or
``RealWorld``. This is required because the ``unienv`` package published on
PyPI predates a recent local core fix (``RealWorld.reset`` kwargs), so the
node is driven through its lifecycle hooks manually.
"""

import numpy as np
import pytest

import unienv_wuji
from unienv_wuji import WujiHandActor


def test_wuji_hand_node_smoke():
    node = WujiHandActor(connect=False)
    try:
        # Static metadata.
        assert WujiHandActor.n_joints == 20
        assert node.n_joints == 20

        # Action space.
        assert node.action_space is not None
        assert node.action_space.shape == (20,)

        # Observation space keys.
        obs_space = node.observation_space
        expected_keys = {"joint_positions", "joint_errors"}
        assert expected_keys.issubset(set(obs_space.spaces.keys()))
        assert obs_space.spaces["joint_positions"].shape == (20,)
        assert obs_space.spaces["joint_errors"].shape == (20,)

        # Before reset, observation is None.
        assert node.get_observation() is None

        # after_reset must populate the observation.
        node.after_reset()
        obs = node.get_observation()
        assert obs is not None
        for key in expected_keys:
            assert key in obs
        assert obs["joint_positions"].shape == (20,)
        assert obs["joint_errors"].shape == (20,)

        # set_next_action accepts a correctly-shaped float32 action.
        action = np.zeros((20,), dtype=np.float32)
        node.set_next_action(action)

        # set_next_action rejects a wrong-shape action.
        with pytest.raises(AssertionError):
            node.set_next_action(np.zeros((3,), dtype=np.float32))

        # Lifecycle hooks run without error.
        node.pre_environment_step(0.04)
        node.post_environment_step(0.04)
        node.after_reset()

        # get_observation still returns a populated dict after stepping.
        obs2 = node.get_observation()
        assert obs2 is not None
        assert obs2["joint_positions"].shape == (20,)
    finally:
        # close() must run cleanly even when never connected.
        node.close()


def test_public_api_exposed():
    assert hasattr(unienv_wuji, "WujiHandActor")
