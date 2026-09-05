import os
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

from pyrpl import Pyrpl


# ============================================================
# SETTINGS
# ============================================================

HOSTNAME = "rp-f0c970.local"

# Desired half-ramp:
#
#   -0.5 V  -------------------->  +0.5 V
#
# Total voltage span = 1.0 V
#
CENTER_V = 0.0
SWEEP_RANGE_V = 1.0

SWEEP_TIME_S = 2.0

# Scope
SCOPE_DECIMATION = 16384


# ============================================================
# VOLTAGE CALCULATION
# ============================================================

AMPLITUDE_V = SWEEP_RANGE_V / 2.0
OFFSET_V = CENTER_V

START_V = CENTER_V - AMPLITUDE_V
END_V = CENTER_V + AMPLITUDE_V

FREQUENCY_HZ = 1.0 / SWEEP_TIME_S


# ============================================================
# CONNECT
# ============================================================

print("Connecting to Red Pitaya...")

current_dir = os.path.dirname(os.path.abspath(__file__))
local_user_dir = os.path.join(current_dir, "pyrpl_data")

p = Pyrpl(
    config="my_config",
    hostname=HOSTNAME,
    user_dir=local_user_dir,
    reloadfpga=True,
    gui=False,
)

rp = p.rp

asg = rp.asg0
scope = rp.scope

print("Connected.")


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print()
print("============================================================")
print("PYRPL HALF-RAMP TEST")
print("============================================================")

print()
print("ASG:")
print(f"  Waveform:       halframp")
print(f"  Start voltage:  {START_V:+.3f} V")
print(f"  End voltage:    {END_V:+.3f} V")
print(f"  Amplitude:      {AMPLITUDE_V:.3f} V")
print(f"  Offset:         {OFFSET_V:+.3f} V")
print(f"  Sweep time:     {SWEEP_TIME_S:.3f} s")
print(f"  Frequency:      {FREQUENCY_HZ:.6f} Hz")

print()
print("Scope:")
print(f"  Input:          IN1")
print(f"  Decimation:     {SCOPE_DECIMATION}")


# ============================================================
# MAKE SURE ASG IS OFF
# ============================================================

asg.output_direct = "off"


# ============================================================
# CONFIGURE SCOPE
# ============================================================

print()
print("Configuring scope...")

scope.setup(
    input1="in1",
    input2="off",
    ch1_active=True,
    ch2_active=False,
    average=False,
    trace_average=1,
    run_continuous=False,
    rolling_mode=False,
    decimation=SCOPE_DECIMATION,
    trigger_source="asg0",
    trigger_delay=0,
)

print(f"Scope duration: {scope.duration:.6f} s")


# ============================================================
# CONFIGURE ASG
# ============================================================

print()
print("Configuring ASG0...")

asg.output_direct = "out1"

asg.setup(
    waveform="halframp",
    frequency=FREQUENCY_HZ,
    amplitude=AMPLITUDE_V,
    offset=OFFSET_V,
    trigger_source="immediately",
)

print("ASG configured.")


# ============================================================
# ARM SCOPE
# ============================================================

print()
print("Arming scope...")

scope_future = scope.single_async()

sleep(0.05)

print("Scope armed.")


# ============================================================
# START HALF-RAMP
# ============================================================

print()
print("Starting half-ramp...")

# Re-run the ASG setup to generate the trigger event.
#
# This is intentionally the documented PyRPL mechanism.
asg.setup(
    waveform="halframp",
    frequency=FREQUENCY_HZ,
    amplitude=AMPLITUDE_V,
    offset=OFFSET_V,
    trigger_source="immediately",
)

print("Half-ramp triggered.")


# ============================================================
# WAIT FOR ACQUISITION
# ============================================================

print()
print("Waiting for acquisition...")

raw_data = scope_future.result()

print("Acquisition complete.")


# ============================================================
# EXTRACT CHANNEL 1
# ============================================================

raw_data = np.asarray(raw_data, dtype=float)

print()
print(f"Raw data shape: {raw_data.shape}")

if raw_data.ndim == 2:

    if raw_data.shape[0] == 2:
        voltage_data = raw_data[0]

    elif raw_data.shape[1] == 2:
        voltage_data = raw_data[:, 0]

    else:
        raise RuntimeError(
            f"Unexpected scope data shape: {raw_data.shape}"
        )

elif raw_data.ndim == 1:

    voltage_data = raw_data

else:

    raise RuntimeError(
        f"Unexpected scope data shape: {raw_data.shape}"
    )

voltage_data = np.asarray(
    voltage_data,
    dtype=float,
).reshape(-1)


# ============================================================
# TIME AXIS
# ============================================================

time_data = np.asarray(
    scope.times,
    dtype=float,
).reshape(-1)

time_data -= time_data[0]


# ============================================================
# RESULTS
# ============================================================

print()
print("============================================================")
print("MEASURED RESULTS")
print("============================================================")

print(f"Samples:          {len(voltage_data)}")
print(f"Time:             {time_data[0]:.6f} -> {time_data[-1]:.6f} s")

print()
print(f"Measured minimum: {np.min(voltage_data):+.6f} V")
print(f"Measured maximum: {np.max(voltage_data):+.6f} V")

print()
print("First 10 samples:")
print(voltage_data[:10])

print()
print("Last 10 samples:")
print(voltage_data[-10:])


# ============================================================
# TURN OUTPUT OFF
# ============================================================

asg.output_direct = "off"

print()
print("OUT1 disabled.")


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    time_data,
    voltage_data,
    linewidth=1.2,
)

plt.axhline(
    START_V,
    linestyle="--",
    linewidth=1,
    label=f"Start = {START_V:+.2f} V",
)

plt.axhline(
    END_V,
    linestyle="--",
    linewidth=1,
    label=f"End = {END_V:+.2f} V",
)

plt.xlabel("Time (s)")
plt.ylabel("IN1 voltage (V)")

plt.title(
    "Red Pitaya OUT1 → IN1\n"
    "1 V Half-Ramp: −0.5 V → +0.5 V"
)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()