import sys
import time
import warnings
from typing import Dict, List, Optional

import numpy as np

from unienv_interface.world import WorldNode, RealWorld, World
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.numpy import NumpyComputeBackend, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
from unienv_interface.space import DictSpace, BoxSpace


class WujiHandHardwareError(RuntimeError):
    """Raised when the Wuji hand reports nonzero per-joint error codes.

    Hardware error codes are not policy observations — a nonzero code indicates
    a fault that must be handled (e.g. inspect the hand, then call
    :meth:`WujiHandActor.clear_errors`) rather than silently propagated through
    the observation stream. The offending joint error codes are attached on
    ``error_codes``.

    Attributes
    ----------
    error_codes: np.ndarray
        Flat ``(n_joints,)`` array of per-joint error codes reported by the hand.
    """

    def __init__(self, message: str, error_codes: Optional[np.ndarray] = None):
        super().__init__(message)
        self.error_codes = error_codes


class WujiHandThermalError(WujiHandHardwareError):
    """Raised when the hand reaches the configured critical temperature.

    Attributes
    ----------
    joint_indices: tuple[int, ...]
        Flat indices of joints at or above the configured threshold.
    joint_temperatures: np.ndarray
        Flat ``(n_joints,)`` temperature array in degrees Celsius.
    max_temperature_c: float
        Configured critical threshold in degrees Celsius.
    """

    def __init__(
        self,
        message: str,
        *,
        joint_indices: tuple[int, ...],
        joint_temperatures: np.ndarray,
        max_temperature_c: float,
    ):
        super().__init__(message)
        self.joint_indices = joint_indices
        self.joint_temperatures = joint_temperatures
        self.max_temperature_c = max_temperature_c


class WujiHandActor(WorldNode[
    None, Dict[str, NumpyArrayType], NumpyArrayType,
    NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
]):
    # The first WorldEnv.reset uses the reload flow, so we must also register
    # after_reload to refresh the initial observation (mirrors the RohandActor
    # bug-fix: without this the initial observation would be None).
    after_reset_priorities = {0}
    after_reload_priorities = {0}
    pre_environment_step_priorities = {0}
    post_environment_step_priorities = {0}

    # Joint layout: 5 fingers (thumb, index, middle, ring, pinky) x
    # 4 joints (proximal, abduction/spread, pip, dip), flat row-major.
    joint_names: List[str] = [
        f"finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5)
    ]
    n_joints: int = len(joint_names)  # 20

    # Fallback joint limits (radians) used when not connecting to hardware.
    _fallback_joint_limit_low = np.full((n_joints,), -3.0, dtype=np.float32)
    _fallback_joint_limit_high = np.full((n_joints,), 3.0, dtype=np.float32)

    # Upper bound for joint error codes (unsigned 32-bit range).
    _joint_error_high = float(2 ** 31)

    # Generous-but-sane bounds for the extra proprioception channels. These are
    # not device-reported; they only bound the observation BoxSpace so downstream
    # consumers (sims, normalisers) have finite ranges.
    _joint_effort_low = np.full((n_joints,), -5.0, dtype=np.float32)
    _joint_effort_high = np.full((n_joints,), 5.0, dtype=np.float32)
    _joint_temperature_low = np.full((n_joints,), -20.0, dtype=np.float32)
    _joint_temperature_high = np.full((n_joints,), 100.0, dtype=np.float32)
    _joint_bus_voltage_low = np.full((n_joints,), 0.0, dtype=np.float32)
    _joint_bus_voltage_high = np.full((n_joints,), 50.0, dtype=np.float32)
    _overheat_warning_interval_s = 1.0
    _diagnostics_failure_warning_threshold = 3

    def __init__(
        self,
        world: Optional[RealWorld] = None,
        name: str = "wuji_hand",
        serial_number: Optional[str] = None,
        *,
        filter_cutoff_hz: float = 5.0,
        connect: bool = True,
        control_timestep: Optional[float] = 0.04,  # 25Hz
        update_timestep: Optional[float] = 0.04,  # background read/send frequency
        max_temperature_c: float = 60.0,
        warning_temperature_c: float = 55.0,
        diagnostics_period_s: float = 2.0,
        diagnostics_read_timeout_s: float = 0.1,
    ):
        """Create a Wuji hand actor.

        Parameters
        ----------
        max_temperature_c:
            Per-joint critical temperature at which initialization or action
            sending raises :class:`WujiHandThermalError`. The conservative
            default is 60 °C: official Wuji Hand v1 documentation only begins
            firmware effort derating at 85 °C and 93 °C.
        warning_temperature_c:
            Per-joint early-warning temperature. At or above this value but
            below ``max_temperature_c``, commands are still sent while a
            throttled warning is emitted. Defaults to 55 °C.
        diagnostics_period_s:
            Minimum period between inline bulk temperature / bus-voltage reads.
            The default 2 seconds is fast relative to the reported ~1-minute
            damage timescale while avoiding unnecessary SDO traffic.
        diagnostics_read_timeout_s:
            Timeout for each normal inline bulk diagnostics read. Healthy reads
            normally complete in milliseconds; the 0.1 second default bounds a
            slow/failing read to one 5 Hz control period.
        """
        # Set WorldNode-related attributes first so backend/device properties work.
        self.name = name
        self.serial_number = serial_number
        self.filter_cutoff_hz = float(filter_cutoff_hz)
        self.connect = bool(connect)
        self.max_temperature_c = float(max_temperature_c)
        self.warning_temperature_c = float(warning_temperature_c)
        self.diagnostics_period_s = float(diagnostics_period_s)
        self.diagnostics_read_timeout_s = float(diagnostics_read_timeout_s)
        if not np.isfinite(self.max_temperature_c) or self.max_temperature_c <= 0:
            raise ValueError("max_temperature_c must be a positive finite value.")
        if not np.isfinite(self.warning_temperature_c):
            raise ValueError("warning_temperature_c must be finite.")
        if self.warning_temperature_c > self.max_temperature_c:
            raise ValueError(
                "warning_temperature_c must be less than or equal to max_temperature_c."
            )
        if not np.isfinite(self.diagnostics_period_s) or self.diagnostics_period_s <= 0:
            raise ValueError("diagnostics_period_s must be a positive finite value.")
        if (
            not np.isfinite(self.diagnostics_read_timeout_s)
            or self.diagnostics_read_timeout_s <= 0
        ):
            raise ValueError("diagnostics_read_timeout_s must be a positive finite value.")

        if isinstance(world, World):
            assert world.backend == NumpyComputeBackend, "World backend must be NumpyComputeBackend."
            assert world.is_control_timestep_compatible(control_timestep), "Control timestep must be a multiple of world timestep."
        self.world = world

        # Hardware handles — only populated when connect=True.
        self._hand = None
        self._controller = None
        self._controller_ctx = None
        self._joint_temperature_cache = np.zeros((self.n_joints,), dtype=np.float32)
        self._joint_bus_voltage_cache = np.zeros((self.n_joints,), dtype=np.float32)
        self._last_diagnostics_time = float("-inf")
        self._diagnostics_consecutive_failures = 0
        self._last_temperature_warning_time = float("-inf")

        # Joint limits (radians), shape (20,). Defaults to the fallback values;
        # overwritten with device-reported limits when connecting to hardware.
        self.joint_limit_low = self._fallback_joint_limit_low.copy()
        self.joint_limit_high = self._fallback_joint_limit_high.copy()

        if self.connect:
            try:
                self._connect_hardware()
            except Exception:
                self.close()
                raise

        # Build observation / action spaces using the (possibly device-read) limits.
        # The key set is always stable so downstream consumers can rely on it.
        # Slow-changing diagnostics are refreshed inline at a throttled rate.
        self.observation_space = DictSpace(
            NumpyComputeBackend,
            {
                "joint_positions": BoxSpace(  # radians, realtime bulk-read
                    NumpyComputeBackend,
                    low=self.joint_limit_low,
                    high=self.joint_limit_high,
                    dtype=np.float32,
                    shape=(self.n_joints,),
                ),
                "joint_efforts": BoxSpace(  # A, realtime bulk-read
                    NumpyComputeBackend,
                    low=self._joint_effort_low,
                    high=self._joint_effort_high,
                    dtype=np.float32,
                    shape=(self.n_joints,),
                ),
                "joint_temperatures": BoxSpace(  # °C, cached SDO (diagnostic)
                    NumpyComputeBackend,
                    low=self._joint_temperature_low,
                    high=self._joint_temperature_high,
                    dtype=np.float32,
                    shape=(self.n_joints,),
                ),
                "joint_bus_voltages": BoxSpace(  # V, cached SDO (diagnostic)
                    NumpyComputeBackend,
                    low=self._joint_bus_voltage_low,
                    high=self._joint_bus_voltage_high,
                    dtype=np.float32,
                    shape=(self.n_joints,),
                ),
            },
        )
        self.action_space = BoxSpace(
            NumpyComputeBackend,
            low=self.joint_limit_low,
            high=self.joint_limit_high,
            dtype=np.float32,
            shape=(self.n_joints,),
        )

        self.control_timestep = control_timestep  # Control timestep in seconds
        self.update_timestep = update_timestep
        self._current_observation: Optional[Dict[str, NumpyArrayType]] = None
        self._next_action: Optional[NumpyArrayType] = None

    # ========== Hardware connection ==========

    def _connect_hardware(self) -> None:
        """Connect, verify thermal safety, enable, and start diagnostics."""
        try:
            import wujihandpy  # noqa: PLC0415 — lazy import so the package imports without the SDK.
        except ImportError as e:  # pragma: no cover — requires hardware/SDK
            raise ImportError(
                "The 'wujihandpy' SDK is required to connect to a Wuji hand. "
                "Install it with: pip install wujihandpy"
            ) from e

        hand_kwargs = {}
        if self.serial_number is not None:
            hand_kwargs["serial_number"] = self.serial_number
        self._hand = wujihandpy.Hand(**hand_kwargs)

        # Read per-joint limits from the device: (5, 4) -> (20,) row-major.
        lower = np.asarray(self._hand.read_joint_lower_limit(), dtype=np.float32).reshape(-1)
        upper = np.asarray(self._hand.read_joint_upper_limit(), dtype=np.float32).reshape(-1)
        if lower.shape == (self.n_joints,) and upper.shape == (self.n_joints,):
            self.joint_limit_low = lower
            self.joint_limit_high = upper

        # Read and clear any startup error codes, then (re-)enable the hand.
        try:
            errs = np.asarray(self._hand.read_joint_error_code()).reshape(-1)
            if np.any(errs):
                self._hand.write_joint_reset_error(1)
                time.sleep(0.1)
        except Exception:
            # Best-effort: don't fail construction on a transient error read.
            pass

        # A fresh blocking temperature read is mandatory before enabling motors.
        # If it fails, construction fails closed because safe startup cannot be
        # established. Its 0.5 s timeout is intentionally retained for startup.
        # Bus voltage is diagnostic-only, so a failed initial read is left for
        # the normal inline refresh path to retry.
        try:
            initial_temperatures = self._read_joint_temperatures_from_hardware(
                timeout_s=0.5
            )
        except Exception as exc:
            raise WujiHandHardwareError(
                "Unable to verify Wuji hand joint temperatures during initialization."
            ) from exc
        self._update_temperature_cache(initial_temperatures)
        self._check_initial_temperature_policy(initial_temperatures)

        try:
            self._update_bus_voltage_cache(
                self._read_joint_bus_voltages_from_hardware(timeout_s=0.5)
            )
        except Exception:
            pass

        self._hand.write_joint_enabled(True)

        # Enter the realtime controller context (upstream enabled + LowPass filter).
        self._controller_ctx = self._hand.realtime_controller(
            enable_upstream=True,
            filter=wujihandpy.filter.LowPass(cutoff_freq=self.filter_cutoff_hz),
        )
        self._controller = self._controller_ctx.__enter__()

    def _check_initial_temperature_policy(self, temperatures: np.ndarray) -> None:
        hot_indices = np.flatnonzero(temperatures >= self.max_temperature_c)
        if hot_indices.size:
            self._raise_thermal_error(
                temperatures,
                hot_indices,
                context="initialize",
                recovery="Allow the hand to cool before reconnecting.",
                emit_warning=True,
            )

        warning_indices = np.flatnonzero(
            (temperatures >= self.warning_temperature_c)
            & (temperatures < self.max_temperature_c)
        )
        if warning_indices.size:
            self._emit_temperature_warning(
                temperatures,
                warning_indices,
                context="during initialization",
                throttled=False,
            )

    def _raise_thermal_error(
        self,
        temperatures: np.ndarray,
        indices: np.ndarray,
        *,
        context: str,
        recovery: str,
        emit_warning: bool = False,
    ) -> None:
        details = self._format_hot_joints(temperatures, indices)
        message = (
            f"Wuji hand is too hot to {context}: {details}; critical threshold is "
            f"{self.max_temperature_c:.1f} °C. {recovery}"
        )
        if emit_warning:
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        print(f"CRITICAL THERMAL SAFETY WARNING: {message}", file=sys.stderr, flush=True)
        raise WujiHandThermalError(
            message,
            joint_indices=tuple(int(i) for i in indices),
            joint_temperatures=temperatures.copy(),
            max_temperature_c=self.max_temperature_c,
        )

    def _emit_temperature_warning(
        self,
        temperatures: np.ndarray,
        indices: np.ndarray,
        *,
        context: str,
        throttled: bool,
    ) -> None:
        if throttled:
            now = time.monotonic()
            if (
                now - self._last_temperature_warning_time
                < self._overheat_warning_interval_s
            ):
                return
            self._last_temperature_warning_time = now

        warnings.warn(
            f"Wuji hand {self.name!r} temperature warning {context}: "
            f"{self._format_hot_joints(temperatures, indices)}; warning threshold is "
            f"{self.warning_temperature_c:.1f} °C and critical threshold is "
            f"{self.max_temperature_c:.1f} °C.",
            RuntimeWarning,
            stacklevel=3,
        )

    def _format_hot_joints(self, temperatures: np.ndarray, indices: np.ndarray) -> str:
        return ", ".join(
            f"{self.joint_names[int(i)]} (index {int(i)}): {temperatures[int(i)]:.1f} °C"
            for i in indices
        )

    def _refresh_diagnostics_if_due(self) -> None:
        if not self.connect or self._hand is None:
            return

        now = time.monotonic()
        if now - self._last_diagnostics_time < self.diagnostics_period_s:
            return
        # Throttle retries as well as successful reads so a failing device cannot
        # stall every control step.
        self._last_diagnostics_time = now

        try:
            temperatures = self._read_joint_temperatures_from_hardware(
                timeout_s=self.diagnostics_read_timeout_s
            )
            bus_voltages = self._read_joint_bus_voltages_from_hardware(
                timeout_s=self.diagnostics_read_timeout_s
            )
        except Exception as exc:
            self._diagnostics_consecutive_failures += 1
            if (
                self._diagnostics_consecutive_failures
                == self._diagnostics_failure_warning_threshold
            ):
                warnings.warn(
                    f"Wuji hand {self.name!r} diagnostics refresh failed "
                    f"{self._diagnostics_consecutive_failures} consecutive times "
                    f"({exc}); keeping the last good cache.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return

        self._update_temperature_cache(temperatures)
        self._update_bus_voltage_cache(bus_voltages)
        self._diagnostics_consecutive_failures = 0

    # ========== Actor Implementation ==========

    @property
    def backend(self) -> ComputeBackend[NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType]:
        return NumpyComputeBackend

    @property
    def device(self) -> None:
        return None

    def pre_environment_step(self, dt: float, *, priority: int = 0) -> None:
        if self._next_action is not None:
            self.send_joint_command(self._next_action)

    def post_environment_step(self, dt: float, *, priority: int = 0) -> None:
        self._refresh_observation()

    def after_reset(self, *, priority: int = 0, mask=None) -> None:
        self._refresh_observation()

    # after_reload defaults to calling after_reset; we register its priority so
    # the reload flow (used by the first WorldEnv.reset) refreshes observation.
    # No override needed — the base implementation delegates to after_reset.

    def _refresh_observation(self) -> None:
        """Read all proprioception channels and (when connected) check for hardware errors.

        Offline (``connect=False``) never raises and returns zeros. When
        connected, any nonzero per-joint error code is treated as a hardware
        fault and raises :class:`WujiHandHardwareError` — error codes are not
        exposed as a policy observation. Inspect the hand and call
        :meth:`clear_errors` to recover.

        ``joint_positions`` and ``joint_efforts`` are read from the realtime
        controller. Temperature and bus-voltage bulk reads are performed inline
        only when ``diagnostics_period_s`` has elapsed, with each read bounded by
        ``diagnostics_read_timeout_s``. Between refreshes, observations use the
        last good cache.
        """
        joint_positions = self.read_joint_positions()
        joint_efforts = self.read_joint_efforts()
        self._refresh_diagnostics_if_due()
        joint_temperatures, joint_bus_voltages = self._diagnostics_snapshot()

        if self.connect and self._hand is not None:
            error_codes = self.read_joint_error_codes()
            if np.any(error_codes):
                raise WujiHandHardwareError(
                    "Wuji hand reported nonzero joint error codes "
                    f"(max={int(np.max(error_codes))}); inspect the hand and call "
                    "clear_errors() to recover.",
                    error_codes=error_codes,
                )
        self._current_observation = {
            "joint_positions": joint_positions,
            "joint_efforts": joint_efforts,
            "joint_temperatures": joint_temperatures,
            "joint_bus_voltages": joint_bus_voltages,
        }

    def get_observation(self):
        return self._current_observation

    def set_next_action(self, action):
        assert isinstance(action, NumpyArrayType), "Action must be a numpy array."
        assert action.shape == (self.n_joints,), f"Action shape must be ({self.n_joints},), got {action.shape}"
        self._next_action = action

    def close(self):
        # Best-effort teardown: exit the realtime controller context and disable the hand.
        controller_ctx = getattr(self, "_controller_ctx", None)
        if controller_ctx is not None:
            try:
                controller_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._controller_ctx = None
            self._controller = None
        hand = getattr(self, "_hand", None)
        if hand is not None:
            try:
                hand.write_joint_enabled(False)
            except Exception:
                pass
            self._hand = None

    # ========== Helper Methods ==========

    def read_joint_positions(self) -> np.ndarray:
        """
        Read the actual joint positions of the hand in radians, flat (20,) row-major.
        Returns zeros when not connected to hardware.
        """
        if not self.connect or self._hand is None:
            return np.zeros((self.n_joints,), dtype=np.float32)

        actual = None
        if self._controller is not None:
            getter = getattr(self._controller, "get_joint_actual_position", None)
            if getter is not None:
                try:
                    actual = np.asarray(getter(), dtype=np.float32).reshape(-1)
                except Exception:
                    actual = None
        if actual is None:
            try:
                actual = np.asarray(
                    self._hand.read_joint_actual_position(), dtype=np.float32
                ).reshape(-1)
            except Exception:
                actual = None
        if actual is None:
            return np.zeros((self.n_joints,), dtype=np.float32)
        return actual.astype(np.float32, copy=False)

    def read_joint_efforts(self) -> np.ndarray:
        """
        Read the actual signed motor effort/current of the hand,
        flat (20,) row-major, from the realtime controller bulk-read.

        Units: amperes (A), matching the wujihandpy realtime example and effort
        limit API. This is motor-current-derived and only proportional to torque;
        it is not an independently measured joint torque.

        Returns zeros when not connected to hardware or when the realtime
        controller does not expose the accessor.
        """
        if not self.connect or self._controller is None:
            return np.zeros((self.n_joints,), dtype=np.float32)

        effort = None
        getter = getattr(self._controller, "get_joint_actual_effort", None)
        if getter is not None:
            try:
                effort = np.asarray(getter(), dtype=np.float32).reshape(-1)
            except Exception:
                effort = None
        if effort is None or effort.shape != (self.n_joints,):
            return np.zeros((self.n_joints,), dtype=np.float32)
        return effort.astype(np.float32, copy=False)

    def _diagnostics_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._joint_temperature_cache.copy(),
            self._joint_bus_voltage_cache.copy(),
        )

    def _update_temperature_cache(self, temperatures: np.ndarray) -> None:
        self._joint_temperature_cache = temperatures.astype(np.float32, copy=True)

    def _update_bus_voltage_cache(self, voltages: np.ndarray) -> None:
        self._joint_bus_voltage_cache = voltages.astype(np.float32, copy=True)

    def _read_joint_temperatures_from_hardware(self, *, timeout_s: float) -> np.ndarray:
        if not self.connect or self._hand is None:
            return np.zeros((self.n_joints,), dtype=np.float32)
        temperatures = np.asarray(
            self._hand.read_joint_temperature(timeout=timeout_s), dtype=np.float32
        ).reshape(-1)
        if temperatures.shape != (self.n_joints,):
            raise ValueError(
                f"Expected {self.n_joints} joint temperatures, got {temperatures.shape}."
            )
        if not np.all(np.isfinite(temperatures)):
            raise ValueError("Joint temperature read contained non-finite values.")
        return temperatures

    def _read_joint_bus_voltages_from_hardware(self, *, timeout_s: float) -> np.ndarray:
        if not self.connect or self._hand is None:
            return np.zeros((self.n_joints,), dtype=np.float32)
        voltages = np.asarray(
            self._hand.read_joint_bus_voltage(timeout=timeout_s), dtype=np.float32
        ).reshape(-1)
        if voltages.shape != (self.n_joints,):
            raise ValueError(
                f"Expected {self.n_joints} joint bus voltages, got {voltages.shape}."
            )
        if not np.all(np.isfinite(voltages)):
            raise ValueError("Joint bus voltage read contained non-finite values.")
        return voltages

    def read_joint_temperatures(self) -> np.ndarray:
        """
        Issue a synchronous bulk joint-temperature SDO read (°C, flat (20,))
        and update the actor's diagnostics cache.

        .. note::

            This explicit helper uses a 0.5 s timeout. Normal observation
            refreshes instead use ``diagnostics_read_timeout_s`` and are
            throttled by ``diagnostics_period_s``. Returns the last good cached
            value if a connected read fails, or zeros offline.
        """
        if not self.connect or self._hand is None:
            return np.zeros((self.n_joints,), dtype=np.float32)
        try:
            temperatures = self._read_joint_temperatures_from_hardware(timeout_s=0.5)
        except Exception:
            return self._diagnostics_snapshot()[0]
        self._update_temperature_cache(temperatures)
        return temperatures

    def read_joint_bus_voltages(self) -> np.ndarray:
        """
        Issue a synchronous bulk joint bus-voltage SDO read (V, flat (20,))
        and update the actor's diagnostics cache.

        .. note::

            This explicit helper uses a 0.5 s timeout. Normal observation
            refreshes instead use ``diagnostics_read_timeout_s`` and are
            throttled by ``diagnostics_period_s``. Returns the last good cached
            value if a connected read fails, or zeros offline.
        """
        if not self.connect or self._hand is None:
            return np.zeros((self.n_joints,), dtype=np.float32)
        try:
            voltages = self._read_joint_bus_voltages_from_hardware(timeout_s=0.5)
        except Exception:
            return self._diagnostics_snapshot()[1]
        self._update_bus_voltage_cache(voltages)
        return voltages

    def send_joint_command(self, positions_rad: np.ndarray) -> None:
        """
        Send target joint positions (radians, flat (20,)) to the hand.
        Validates shape, clips to device limits, reshapes to (5, 4) and dispatches
        via the realtime controller. No-op when not connected.

        If the latest diagnostics cache contains any joint at or above
        ``max_temperature_c``, raises :class:`WujiHandThermalError` before
        sending. Temperatures from ``warning_temperature_c`` up to the critical
        threshold emit a warning at most once per second while still sending.
        """
        if positions_rad.shape != (self.n_joints,):
            raise ValueError(
                f"Expected positions shape to be ({self.n_joints},), got {positions_rad.shape}"
            )

        if not self.connect or self._controller is None:
            return

        temperatures = self._diagnostics_snapshot()[0]
        hot_indices = np.flatnonzero(temperatures >= self.max_temperature_c)
        if hot_indices.size:
            self._raise_thermal_error(
                temperatures,
                hot_indices,
                context="send an action",
                recovery="Abort or reset the episode and allow the hand to cool.",
            )

        warning_indices = np.flatnonzero(
            temperatures >= self.warning_temperature_c
        )
        if warning_indices.size:
            self._emit_temperature_warning(
                temperatures,
                warning_indices,
                context="while sending an action",
                throttled=True,
            )

        q = np.asarray(positions_rad, dtype=np.float32)
        q = np.clip(q, self.joint_limit_low, self.joint_limit_high)
        q_5x4 = q.reshape(5, 4)
        self._controller.set_joint_target_position(q_5x4)

    def read_joint_error_codes(self) -> np.ndarray:
        """
        Read per-joint error codes, flat (20,). Returns zeros when not connected.
        """
        if not self.connect or self._hand is None:
            return np.zeros((self.n_joints,), dtype=np.float32)
        try:
            errs = np.asarray(
                self._hand.read_joint_error_code(), dtype=np.float32
            ).reshape(-1)
        except Exception:
            return np.zeros((self.n_joints,), dtype=np.float32)
        return errs.astype(np.float32, copy=False)

    def clear_errors(self) -> None:
        """Clear any latched joint error codes on the hand. No-op when not connected."""
        if not self.connect or self._hand is None:
            return
        try:
            self._hand.write_joint_reset_error(1)
        except Exception:
            pass

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the hand's torque. No-op when not connected."""
        if not self.connect or self._hand is None:
            return
        try:
            self._hand.write_joint_enabled(bool(enabled))
        except Exception:
            pass
