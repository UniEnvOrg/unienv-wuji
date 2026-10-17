# Wuji Hand Adaptor

Wraps the Wuji dexterous hand (SDK: [`wujihandpy`](https://pypi.org/project/wujihandpy/)) in the UniEnv node system.

## Observation & Hardware Errors

`WujiHandActor` exposes the full proprioception surface of the wujihandpy SDK as
observation keys (all `(20,)`, `float32`, flat row-major in the
5 fingers × 4 joints layout):

| Key | Units | Source | Notes |
| --- | --- | --- | --- |
| `joint_positions` | rad | realtime controller bulk-read (`get_joint_actual_position`) | Always populated when connected. |
| `joint_efforts` | A | realtime controller bulk-read (`get_joint_actual_effort`) | Always populated when connected. Motor-current-derived effort, proportional to torque but not a separate torque measurement. |
| `joint_temperatures` | °C | throttled inline bulk read (`read_joint_temperature`) | Always populated from the latest successful read. |
| `joint_bus_voltages` | V | throttled inline bulk read (`read_joint_bus_voltage`) | Always populated from the latest successful read. |

The SDK does **not** expose joint velocity, motor current, or a separately
measured joint torque; those channels are therefore not available from this
adaptor.

### Space metadata

Every observation and action space carries UniEnv space metadata
(`unienv>=0.0.1b16`): observation leaves are `kind="proprio"` with the `units`
of the table above (`"rad"`, `"A"`, `"degC"`, `"V"`), the joint `names`
(`finger{1..5}_joint{1..4}`), `source` (the node `name`), `serial_number` (when
one is given) and `fps` — `1 / control_timestep` (25 by default) for the
realtime channels and `1 / diagnostics_period_s` (0.5 by default) for the
throttled diagnostics. The action space is `kind="action"`,
`control="joint_position"`, `mode="absolute"`, `units="rad"` with the same
joint `names`.

### Always-on diagnostics

`read_joint_temperature()` and `read_joint_bus_voltage()` are bulk operations:
each call returns all 20 joint values. Their SDK `timeout` argument is a maximum
wait, not their normal cost; healthy reads complete in sub-millisecond to a few
milliseconds. The actor therefore performs them inline from the normal
observation-refresh path, throttled by `diagnostics_period_s` (default `2.0` s).

Normal refreshes use `diagnostics_read_timeout_s=0.1`, bounding each bulk read's
worst-case stall while allowing slow-but-healthy reads. The initial mandatory
temperature safety read retains its more permissive 0.5 s timeout. On a normal
refresh failure, the actor keeps the last good temperature/voltage cache,
throttles retries to the same period, and emits a warning after three
consecutive failures. A successful refresh resets the failure counter. Offline
actors perform no diagnostic reads and expose zero-filled channels.

### Thermal safety

The adaptor uses two configurable per-joint temperature tiers:

- `warning_temperature_c=55.0`: early warning. This is already far above the
  official 15–30 °C recommended ambient range, so it flags abnormal heating
  early while still allowing the current command to be sent.
- `max_temperature_c=60.0`: critical, episode-killing threshold. This remains
  well below the firmware's own 85 °C and 93 °C effort-derating points and is
  intentionally conservative to protect motors from sustained overheating.

`warning_temperature_c` must be less than or equal to `max_temperature_c`.
The official [Wuji Hand v1 user
notice](https://docs.wuji.tech/docs/en/wuji-hand/v1/user-notice/) documents the
15–30 °C environment recommendation and firmware thresholds. Tune both adaptor
thresholds only for validated hardware and workloads.

- **Initialization refusal:** construction performs a fresh blocking read of
  all joint temperatures before enabling the motors. If any joint is at or
  above `max_temperature_c`, it emits a loud warning and raises
  `WujiHandThermalError`, listing the offending joints and temperatures. Partial
  hardware state is closed cleanly. Temperatures in the warning tier allow
  construction but emit a warning.
- **Mid-episode warning:** if any joint is at or above
  `warning_temperature_c` but below `max_temperature_c`, the command is still
  sent and a warning is emitted at most once per second per actor.
- **Mid-episode critical temperature:** if any joint is at or above
  `max_temperature_c`, the command is not sent and `WujiHandThermalError` is
  raised after writing a loud stderr line. This intentional episode crash
  protects the motors; catch the exception at the episode level to abort/reset
  the run and allow the hand to cool.
- **Diagnostics tuning:** `diagnostics_period_s=2.0` is short relative to the
  observed roughly one-minute damage timescale. `diagnostics_read_timeout_s=0.1`
  bounds a slow/failing bulk read to one control period at 5 Hz.

### Hardware error codes

Hardware error codes are **not** exposed as a policy observation. After every
feedback refresh (`post_environment_step`, `after_reset`, and the `after_reload`
flow used by the first `WorldEnv.reset`), the actor reads the per-joint error
codes via `read_joint_error_codes()`. If the actor is connected to hardware
(`connect=True`) and **any** joint error code is nonzero, it raises
[`WujiHandHardwareError`](./unienv_wuji/wuji_hand.py) carrying the offending
codes on `error_codes`. Inspect the hand and call `clear_errors()` to recover.

When `connect=False` (offline), the actor never raises and `read_joint_error_codes()`
returns zeros; `read_joint_error_codes()` and `clear_errors()` remain available
as helper methods.

## Installation

```bash
pip install unienv-wuji
```

## Hardware Setup

The Wuji hand presents as a USB device with VID:PID `0483:2000`.

On Linux, ensure your user has permission to access the device. You can list
connected hands and their USB serial numbers with:

```bash
lsusb -v -d 0483:2000 | grep iSerial
```

If `serial_number=None` is passed to `WujiHandActor` (the default), the SDK
connects to the only/first available hand. When multiple hands are attached,
pass the specific serial number returned by the `lsusb` command above.

If you get permission errors, add a udev rule granting access to the device,
e.g. as root create `/etc/udev/rules.d/99-wuji-hand.rules` with:

```
SUBSYSTEM=="usb", ATTR{idVendor}=="0483", ATTR{idProduct}=="2000", MODE="0666"
```

Then reload udev:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Usage

```python
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.world import RealWorld
from unienv_wuji import WujiHandActor
import numpy as np
import time

world = RealWorld(
    NumpyComputeBackend,
    world_timestep=0.04,  # Usually set this to exactly the control timestep
    batch_size=None,  # None means single instance
)

actor = WujiHandActor(
    world,
    serial_number=None,  # or the USB serial from `lsusb -v -d 0483:2000 | grep iSerial`
    filter_cutoff_hz=5.0,
    connect=True,
    control_timestep=0.04,
    update_timestep=0.04,
    warning_temperature_c=55.0,
    max_temperature_c=60.0,
    diagnostics_period_s=2.0,
    diagnostics_read_timeout_s=0.1,
)

rng = np.random.default_rng(42)
actor.reset()

time_s = time.monotonic()

while True:
    dt = time.monotonic() - time_s
    obs = actor.get_observation()
    print(obs)
    rng, action = actor.action_space.sample(rng)
    actor.set_next_action(action)
    actor.pre_environment_step(dt)
    actor.post_environment_step(dt)
    time_s += dt
```
