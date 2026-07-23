import time
from typing import Dict, List, Optional

import numpy as np

from unienv_interface.world import WorldNode, RealWorld, World
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.numpy import NumpyComputeBackend, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
from unienv_interface.space import DictSpace, BoxSpace


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
    ):
        # Set WorldNode-related attributes first so backend/device properties work.
        self.name = name
        self.serial_number = serial_number
        self.filter_cutoff_hz = float(filter_cutoff_hz)
        self.connect = bool(connect)

        if isinstance(world, World):
            assert world.backend == NumpyComputeBackend, "World backend must be NumpyComputeBackend."
            assert world.is_control_timestep_compatible(control_timestep), "Control timestep must be a multiple of world timestep."
        self.world = world

        # Hardware handles — only populated when connect=True.
        self._hand = None
        self._controller = None
        self._controller_ctx = None

        # Joint limits (radians), shape (20,). Defaults to the fallback values;
        # overwritten with device-reported limits when connecting to hardware.
        self.joint_limit_low = self._fallback_joint_limit_low.copy()
        self.joint_limit_high = self._fallback_joint_limit_high.copy()

        if self.connect:
            self._connect_hardware()

        # Build observation / action spaces using the (possibly device-read) limits.
        self.observation_space = DictSpace(
            NumpyComputeBackend,
            {
                "joint_positions": BoxSpace(  # Joint positions in radians
                    NumpyComputeBackend,
                    low=self.joint_limit_low,
                    high=self.joint_limit_high,
                    dtype=np.float32,
                    shape=(self.n_joints,),
                ),
                "joint_errors": BoxSpace(  # Joint error codes (per-joint)
                    NumpyComputeBackend,
                    low=0.0,
                    high=self._joint_error_high,
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
        """Open the Wuji hand, read limits, clear errors, enable, enter realtime controller."""
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

        self._hand.write_joint_enabled(True)

        # Enter the realtime controller context (upstream enabled + LowPass filter).
        self._controller_ctx = self._hand.realtime_controller(
            enable_upstream=True,
            filter=wujihandpy.filter.LowPass(cutoff_freq=self.filter_cutoff_hz),
        )
        self._controller = self._controller_ctx.__enter__()

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
        self._current_observation = {
            "joint_positions": self.read_joint_positions(),
            "joint_errors": self.read_joint_error_codes(),
        }

    def after_reset(self, *, priority: int = 0, mask=None) -> None:
        self.post_environment_step(0.0, priority=priority)

    # after_reload defaults to calling after_reset; we register its priority so
    # the reload flow (used by the first WorldEnv.reset) refreshes observation.
    # No override needed — the base implementation delegates to after_reset.

    def get_observation(self):
        return self._current_observation

    def set_next_action(self, action):
        assert isinstance(action, NumpyArrayType), "Action must be a numpy array."
        assert action.shape == (self.n_joints,), f"Action shape must be ({self.n_joints},), got {action.shape}"
        self._next_action = action

    def close(self):
        # Best-effort teardown: exit the realtime controller context and disable the hand.
        if self._controller_ctx is not None:
            try:
                self._controller_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._controller_ctx = None
            self._controller = None
        if self._hand is not None:
            try:
                self._hand.write_joint_enabled(False)
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

    def send_joint_command(self, positions_rad: np.ndarray) -> None:
        """
        Send target joint positions (radians, flat (20,)) to the hand.
        Validates shape, clips to device limits, reshapes to (5, 4) and dispatches
        via the realtime controller. No-op when not connected.
        """
        if positions_rad.shape != (self.n_joints,):
            raise ValueError(
                f"Expected positions shape to be ({self.n_joints},), got {positions_rad.shape}"
            )

        if not self.connect or self._controller is None:
            return

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
