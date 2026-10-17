"""Smoke and hardware-safety tests for the Wuji hand adaptor."""

import inspect
import sys
import types
import warnings
from unittest import mock

import numpy as np
import pytest

import unienv_wuji
from unienv_wuji import (
    WujiHandActor,
    WujiHandHardwareError,
    WujiHandThermalError,
)


EXPECTED_OBS_KEYS = {
    "joint_positions",
    "joint_efforts",
    "joint_temperatures",
    "joint_bus_voltages",
}


def _install_fake_sdk(
    monkeypatch,
    *,
    temperatures=None,
    bus_voltages=None,
):
    temperatures = np.asarray(
        np.full((20,), 25.0) if temperatures is None else temperatures,
        dtype=np.float32,
    )
    bus_voltages = np.asarray(
        np.full((20,), 12.0) if bus_voltages is None else bus_voltages,
        dtype=np.float32,
    )

    class FakeController:
        def __init__(self):
            self.commands = []

        def get_joint_actual_position(self):
            return np.zeros((5, 4), dtype=np.float64)

        def get_joint_actual_effort(self):
            return np.arange(20, dtype=np.float64).reshape(5, 4) / 10.0

        def set_joint_target_position(self, positions):
            self.commands.append(np.asarray(positions).copy())

    class FakeControllerContext:
        def __init__(self):
            self.controller = FakeController()
            self.exited = False

        def __enter__(self):
            return self.controller

        def __exit__(self, exc_type, exc, tb):
            self.exited = True

    class FakeHand:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.temperatures = temperatures.copy()
            self.bus_voltages = bus_voltages.copy()
            self.temperature_reads = 0
            self.voltage_reads = 0
            self.temperature_timeouts = []
            self.voltage_timeouts = []
            self.enabled_writes = []
            self.context = FakeControllerContext()
            FakeHand.instances.append(self)

        def read_joint_lower_limit(self):
            return np.full((5, 4), -1.0, dtype=np.float64)

        def read_joint_upper_limit(self):
            return np.full((5, 4), 1.0, dtype=np.float64)

        def read_joint_error_code(self):
            return np.zeros((5, 4), dtype=np.uint32)

        def write_joint_reset_error(self, value):
            pass

        def read_joint_temperature(self, timeout=0.5):
            self.temperature_reads += 1
            self.temperature_timeouts.append(timeout)
            return self.temperatures.copy().reshape(5, 4)

        def read_joint_bus_voltage(self, timeout=0.5):
            self.voltage_reads += 1
            self.voltage_timeouts.append(timeout)
            return self.bus_voltages.copy().reshape(5, 4)

        def write_joint_enabled(self, enabled):
            self.enabled_writes.append(bool(enabled))

        def realtime_controller(self, *, enable_upstream, filter):
            return self.context

    class LowPass:
        def __init__(self, cutoff_freq):
            self.cutoff_freq = cutoff_freq

    fake_sdk = types.SimpleNamespace(
        Hand=FakeHand,
        filter=types.SimpleNamespace(LowPass=LowPass),
    )
    monkeypatch.setitem(sys.modules, "wujihandpy", fake_sdk)
    return FakeHand


def _make_manual_connected_node(*, error_codes=None) -> WujiHandActor:
    node = WujiHandActor(connect=False)
    node.connect = True
    node._hand = mock.Mock()
    node._controller = mock.Mock()
    node.read_joint_positions = lambda: np.zeros((20,), dtype=np.float32)
    node.read_joint_efforts = lambda: np.zeros((20,), dtype=np.float32)
    node.read_joint_error_codes = lambda: np.asarray(
        np.zeros((20,)) if error_codes is None else error_codes,
        dtype=np.float32,
    )
    node._last_diagnostics_time = float("inf")
    return node


def test_constructor_defaults_and_no_diagnostics_flag():
    signature = inspect.signature(WujiHandActor.__init__)
    assert signature.parameters["max_temperature_c"].default == 60.0
    assert signature.parameters["warning_temperature_c"].default == 55.0
    assert signature.parameters["diagnostics_period_s"].default == 2.0
    assert signature.parameters["diagnostics_read_timeout_s"].default == 0.1
    assert "monitor_period_s" not in signature.parameters
    assert "read_diagnostics" not in signature.parameters
    assert "disable_thread_safe_check" not in inspect.getsource(WujiHandActor)


def test_wuji_hand_node_smoke_offline():
    node = WujiHandActor(connect=False)
    try:
        assert node.n_joints == 20
        assert node.max_temperature_c == 60.0
        assert node.warning_temperature_c == 55.0
        assert node.diagnostics_period_s == 2.0
        assert node.diagnostics_read_timeout_s == 0.1
        assert set(node.observation_space.spaces) == EXPECTED_OBS_KEYS

        for space in node.observation_space.spaces.values():
            assert space.shape == (20,)
            assert space.dtype == np.float32

        node.after_reset()
        obs = node.get_observation()
        assert set(obs) == EXPECTED_OBS_KEYS
        for value in obs.values():
            assert value.shape == (20,)
            assert value.dtype == np.float32
            np.testing.assert_array_equal(value, np.zeros((20,), dtype=np.float32))

        node.read_joint_error_codes = lambda: np.ones((20,), dtype=np.float32)
        node.post_environment_step(0.04)  # Offline never raises.
        node.set_next_action(np.zeros((20,), dtype=np.float32))
        node.pre_environment_step(0.04)

        # Offline remains inert even if its local cache is manually made hot.
        node._update_temperature_cache(np.full((20,), 100.0, dtype=np.float32))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            node.send_joint_command(np.zeros((20,), dtype=np.float32))
        assert caught == []
    finally:
        node.close()
        node.close()  # Idempotent.


def test_public_api_exposed():
    assert unienv_wuji.__all__ == [
        "WujiHandActor",
        "WujiHandHardwareError",
        "WujiHandThermalError",
    ]


def test_init_refuses_hot_hand_and_cleans_up(monkeypatch, capsys):
    temperatures = np.full((20,), 30.0, dtype=np.float32)
    temperatures[6] = 61.5
    fake_hand_type = _install_fake_sdk(monkeypatch, temperatures=temperatures)

    with pytest.warns(RuntimeWarning, match="too hot to initialize"):
        with pytest.raises(WujiHandThermalError) as excinfo:
            WujiHandActor(connect=True, max_temperature_c=60.0)

    error = excinfo.value
    assert error.joint_indices == (6,)
    assert error.joint_temperatures[6] == pytest.approx(61.5)
    assert "finger2_joint3" in str(error)
    assert "61.5" in str(error)
    assert "CRITICAL THERMAL SAFETY WARNING" in capsys.readouterr().err

    hand = fake_hand_type.instances[-1]
    assert hand.enabled_writes == [False]


def test_init_warning_band_constructs_and_warns(monkeypatch):
    temperatures = np.full((20,), 30.0, dtype=np.float32)
    temperatures[9] = 57.0
    fake_hand_type = _install_fake_sdk(monkeypatch, temperatures=temperatures)

    with pytest.warns(RuntimeWarning, match="warning threshold is 55.0"):
        node = WujiHandActor(connect=True)
    try:
        assert node._controller is not None
        assert fake_hand_type.instances[-1].enabled_writes == [True]
    finally:
        node.close()


def test_warning_temperature_must_not_exceed_maximum():
    with pytest.raises(ValueError, match="less than or equal"):
        WujiHandActor(
            connect=False,
            warning_temperature_c=61.0,
            max_temperature_c=60.0,
        )


def test_action_above_max_raises_thermal_error(capsys):
    node = _make_manual_connected_node()
    try:
        action = np.linspace(-0.5, 0.5, 20, dtype=np.float32)
        hot = np.full((20,), 25.0, dtype=np.float32)
        hot[2] = 62.5
        node._update_temperature_cache(hot)

        with pytest.raises(WujiHandThermalError) as excinfo:
            node.send_joint_command(action)
        assert excinfo.value.joint_indices == (2,)
        assert excinfo.value.joint_temperatures[2] == pytest.approx(62.5)
        assert excinfo.value.max_temperature_c == 60.0
        assert "finger1_joint3" in str(excinfo.value)
        assert "62.5" in str(excinfo.value)
        assert "critical threshold is 60.0" in str(excinfo.value)
        assert "CRITICAL THERMAL SAFETY WARNING" in capsys.readouterr().err
        node._controller.set_joint_target_position.assert_not_called()
    finally:
        node.close()


def test_warning_band_sends_and_warnings_are_throttled(monkeypatch):
    node = _make_manual_connected_node()
    try:
        temperatures = np.full((20,), 25.0, dtype=np.float32)
        temperatures[4] = 57.0
        node._update_temperature_cache(temperatures)
        monkeypatch.setattr("unienv_wuji.wuji_hand.time.monotonic", lambda: 100.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(20):
                node.send_joint_command(np.zeros((20,), dtype=np.float32))
        assert len(caught) == 1
        assert "finger2_joint1" in str(caught[0].message)
        assert "57.0" in str(caught[0].message)
        assert "warning threshold is 55.0" in str(caught[0].message)
        assert node._controller.set_joint_target_position.call_count == 20
    finally:
        node.close()


def test_below_warning_sends_without_warning():
    node = _make_manual_connected_node()
    try:
        node._update_temperature_cache(np.full((20,), 54.9, dtype=np.float32))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            node.send_joint_command(np.zeros((20,), dtype=np.float32))
        assert caught == []
        node._controller.set_joint_target_position.assert_called_once()
    finally:
        node.close()


def test_diagnostics_inline_refresh_is_throttled(monkeypatch):
    temperatures = np.linspace(20.0, 39.0, 20, dtype=np.float32)
    voltages = np.linspace(11.0, 12.9, 20, dtype=np.float32)
    fake_hand_type = _install_fake_sdk(
        monkeypatch,
        temperatures=temperatures,
        bus_voltages=voltages,
    )
    now = [100.0]
    monkeypatch.setattr("unienv_wuji.wuji_hand.time.monotonic", lambda: now[0])
    node = WujiHandActor(
        connect=True,
        diagnostics_period_s=2.0,
        diagnostics_read_timeout_s=0.1,
    )
    hand = fake_hand_type.instances[-1]
    try:
        # Construction performs one mandatory temperature and one initial bus
        # voltage read, both with the startup 0.5 s timeout.
        assert hand.temperature_reads == 1
        assert hand.voltage_reads == 1
        assert hand.temperature_timeouts == [0.5]
        assert hand.voltage_timeouts == [0.5]

        hand.temperatures[:] = 35.0
        hand.bus_voltages[:] = 13.0

        # First observation refresh always performs inline diagnostics.
        node.after_reset()
        obs = node.get_observation()
        assert set(obs) == EXPECTED_OBS_KEYS
        np.testing.assert_array_equal(
            obs["joint_temperatures"], np.full((20,), 35.0, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            obs["joint_bus_voltages"], np.full((20,), 13.0, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            obs["joint_efforts"], np.arange(20, dtype=np.float32) / 10.0
        )
        assert hand.temperature_reads == 2
        assert hand.voltage_reads == 2
        assert hand.temperature_timeouts[-1] == pytest.approx(0.1)
        assert hand.voltage_timeouts[-1] == pytest.approx(0.1)

        # Refreshes inside the two-second period use the cache only.
        hand.temperatures[:] = 40.0
        hand.bus_voltages[:] = 14.0
        now[0] = 101.9
        node.post_environment_step(0.04)
        assert hand.temperature_reads == 2
        assert hand.voltage_reads == 2
        np.testing.assert_array_equal(
            node.get_observation()["joint_temperatures"],
            np.full((20,), 35.0, dtype=np.float32),
        )

        # Once the period elapses, both bulk reads run again.
        now[0] = 102.0
        node.post_environment_step(0.04)
        assert hand.temperature_reads == 3
        assert hand.voltage_reads == 3
        np.testing.assert_array_equal(
            node.get_observation()["joint_temperatures"],
            np.full((20,), 40.0, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            node.get_observation()["joint_bus_voltages"],
            np.full((20,), 14.0, dtype=np.float32),
        )
    finally:
        node.close()


def test_diagnostics_failures_keep_cache_and_warn_after_three(monkeypatch):
    initial_temperatures = np.full((20,), 32.0, dtype=np.float32)
    initial_voltages = np.full((20,), 12.5, dtype=np.float32)
    fake_hand_type = _install_fake_sdk(
        monkeypatch,
        temperatures=initial_temperatures,
        bus_voltages=initial_voltages,
    )
    now = [10.0]
    monkeypatch.setattr("unienv_wuji.wuji_hand.time.monotonic", lambda: now[0])
    node = WujiHandActor(connect=True, diagnostics_period_s=2.0)
    hand = fake_hand_type.instances[-1]
    hand.read_joint_temperature = mock.Mock(side_effect=TimeoutError("timed out"))

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for value in (10.0, 12.0, 14.0):
                now[0] = value
                node.post_environment_step(0.04)

        assert len(caught) == 1
        assert "3 consecutive" in str(caught[0].message)
        assert node._diagnostics_consecutive_failures == 3
        assert hand.read_joint_temperature.call_count == 3
        assert hand.read_joint_temperature.call_args.kwargs["timeout"] == pytest.approx(0.1)
        temperatures, voltages = node._diagnostics_snapshot()
        np.testing.assert_array_equal(temperatures, initial_temperatures)
        np.testing.assert_array_equal(voltages, initial_voltages)

        # A later successful full refresh resets the failure count.
        hand.read_joint_temperature = mock.Mock(
            return_value=np.full((5, 4), 34.0, dtype=np.float32)
        )
        hand.bus_voltages[:] = 13.0
        now[0] = 16.0
        node.post_environment_step(0.04)
        assert node._diagnostics_consecutive_failures == 0
        np.testing.assert_array_equal(
            node.get_observation()["joint_temperatures"],
            np.full((20,), 34.0, dtype=np.float32),
        )
    finally:
        node.close()


def test_offline_diagnostics_never_read():
    node = WujiHandActor(connect=False)
    node._read_joint_temperatures_from_hardware = mock.Mock(
        side_effect=AssertionError("offline temperature read")
    )
    node._read_joint_bus_voltages_from_hardware = mock.Mock(
        side_effect=AssertionError("offline voltage read")
    )
    try:
        node.after_reset()
        node.post_environment_step(0.04)
        node._read_joint_temperatures_from_hardware.assert_not_called()
        node._read_joint_bus_voltages_from_hardware.assert_not_called()
    finally:
        node.close()


def test_hardware_error_policy_is_preserved():
    error_codes = np.zeros((20,), dtype=np.float32)
    error_codes[3] = 7.0
    node = _make_manual_connected_node(error_codes=error_codes)
    try:
        with pytest.raises(WujiHandHardwareError) as excinfo:
            node.post_environment_step(0.04)
        np.testing.assert_array_equal(excinfo.value.error_codes, error_codes)
        assert "clear_errors" in str(excinfo.value)
    finally:
        node.close()


def test_all_zero_hardware_errors_do_not_raise():
    node = _make_manual_connected_node(error_codes=np.zeros((20,), dtype=np.float32))
    try:
        node.post_environment_step(0.04)
        assert set(node.get_observation()) == EXPECTED_OBS_KEYS
    finally:
        node.close()


def test_spaces_carry_metadata_offline():
    from unienv_interface.space.space_utils.metadata_utils import validate_metadata

    node = WujiHandActor(connect=False, name="left_hand", control_timestep=0.04)
    validate_metadata(node.observation_space)
    validate_metadata(node.action_space)

    obs = node.observation_space.spaces
    expected = {
        "joint_positions": ("rad", 25.0),
        "joint_efforts": ("A", 25.0),
        "joint_temperatures": ("degC", 0.5),  # throttled diagnostics (2 s)
        "joint_bus_voltages": ("V", 0.5),
    }
    for key, (units, fps) in expected.items():
        md = obs[key].metadata
        assert md["kind"] == "proprio"
        assert md["units"] == units
        assert md["fps"] == fps
        assert md["source"] == "left_hand"
        assert tuple(md["names"]) == tuple(WujiHandActor.joint_names)
        assert "serial_number" not in md

    act = node.action_space.metadata
    assert act["kind"] == "action"
    assert act["control"] == "joint_position"
    assert act["mode"] == "absolute"
    assert act["units"] == "rad"
    assert tuple(act["names"]) == tuple(WujiHandActor.joint_names)


def test_spaces_carry_metadata_connected(monkeypatch):
    from unienv_interface.space.space_utils.metadata_utils import validate_metadata

    _install_fake_sdk(monkeypatch)
    node = WujiHandActor(connect=True, serial_number="WJ123", diagnostics_period_s=4.0)
    try:
        validate_metadata(node.observation_space)
        validate_metadata(node.action_space)
        assert node.observation_space["joint_positions"].metadata["serial_number"] == "WJ123"
        assert node.observation_space["joint_temperatures"].metadata["fps"] == 0.25
        assert node.action_space.metadata["units"] == "rad"
    finally:
        node.close()
