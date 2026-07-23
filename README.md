# Wuji Hand Adaptor

Wraps the Wuji dexterous hand (SDK: [`wujihandpy`](https://pypi.org/project/wujihandpy/)) in the UniEnv node system.

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
