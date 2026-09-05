import os
import time
import numpy as np
import matplotlib.pyplot as plt

from pyrpl import Pyrpl
from pyrpl.async_utils import wait


# ============================================================
# USER SETTINGS
# ============================================================

HOSTNAME = "rp-f0c970.local"

# Physical connection:
#
#     ASG0 -> OUT1 -> wire -> IN1 -> scope
#

MIN_V = 0.0
MAX_V = 0.5
FINAL_V = 0.25

RAMP_UP_TIME = 2.0
TOP_HOLD_TIME = 0.1
RAMP_DOWN_TIME = 1.0
FINAL_HOLD_TIME = 0.1

TOTAL_TIME = (
    RAMP_UP_TIME
    + TOP_HOLD_TIME
    + RAMP_DOWN_TIME
    + FINAL_HOLD_TIME
)

# 32768 gives approximately 4.295 s of scope data.
# Our waveform lasts exactly 4.0 s.
SCOPE_DECIMATION = 32768

SCOPE_TIMEOUT_EXTRA = 3.0


# ============================================================
# CONNECT
# ============================================================

print("Connecting to Red Pitaya with PyRPL...")

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
# ASG INFORMATION
# ============================================================

N = asg.data_length

print()
print("ASG:")
print(f"  Waveform memory: {N} points")
print(f"  Output:          OUT1")


# ============================================================
# BUILD WAVEFORM
# ============================================================
#
# The ASG table uses normalized values from -1 to +1.
#
# With:
#
#     amplitude = 0.5 V
#     offset    = 0.5 V
#
# we get:
#
#     -1 -> 0.0 V
#      0 -> 0.5 V
#     +1 -> 1.0 V
#
# Desired trajectory:
#
#     0 V
#       / 1 second
#      /
#     1 V
#     | 1 second
#     |
#     1 V
#       \
#        \ 1 second
#         \
#          0.5 V
#          | 1 second
#          |
#          0.5 V
#
# ============================================================

amplitude = (MAX_V - MIN_V) / 2.0
offset = (MAX_V + MIN_V) / 2.0

frequency = 1.0 / TOTAL_TIME


# Number of waveform points assigned to each section.
n_up = int(round(N * RAMP_UP_TIME / TOTAL_TIME))
n_top = int(round(N * TOP_HOLD_TIME / TOTAL_TIME))
n_down = int(round(N * RAMP_DOWN_TIME / TOTAL_TIME))

# Everything left goes into the final hold.
n_final = N - n_up - n_top - n_down

if n_final < 1:
    raise RuntimeError("Not enough waveform points for final hold.")


# Convert desired voltages into normalized ASG values.
def voltage_to_normalized(voltage):
    return (voltage - offset) / amplitude


normalized_min = voltage_to_normalized(MIN_V)
normalized_max = voltage_to_normalized(MAX_V)
normalized_final = voltage_to_normalized(FINAL_V)


# ------------------------------------------------------------
# UP-RAMP
# ------------------------------------------------------------

up = np.linspace(
    normalized_min,
    normalized_max,
    n_up,
    endpoint=False,
)


# ------------------------------------------------------------
# TOP HOLD
# ------------------------------------------------------------

top = np.full(
    n_top,
    normalized_max,
)


# ------------------------------------------------------------
# DOWN-RAMP
# ------------------------------------------------------------

down = np.linspace(
    normalized_max,
    normalized_final,
    n_down,
    endpoint=False,
)


# ------------------------------------------------------------
# FINAL HOLD
# ------------------------------------------------------------

final = np.full(
    n_final,
    normalized_final,
)


waveform = np.concatenate(
    [
        up,
        top,
        down,
        final,
    ]
)


if len(waveform) != N:
    raise RuntimeError(
        f"Waveform length is {len(waveform)}, expected {N}"
    )


# Force exact first and final values.
waveform[0] = normalized_min
waveform[-1] = normalized_final


# ============================================================
# PRINT WAVEFORM CONFIGURATION
# ============================================================

print()
print("============================================================")
print("ONE-SHOT 0 V -> 1 V -> 0.5 V TEST")
print("============================================================")

print()
print("Voltage:")
print(f"  Start:          {MIN_V:.3f} V")
print(f"  Maximum:        {MAX_V:.3f} V")
print(f"  Final:          {FINAL_V:.3f} V")
print(f"  Amplitude:      {amplitude:.3f} V")
print(f"  Offset:         {offset:.3f} V")

print()
print("Timing:")
print(f"  Ramp up:        {RAMP_UP_TIME:.3f} s")
print(f"  Hold at 1 V:    {TOP_HOLD_TIME:.3f} s")
print(f"  Ramp down:      {RAMP_DOWN_TIME:.3f} s")
print(f"  Hold at 0.5 V:  {FINAL_HOLD_TIME:.3f} s")
print(f"  Total:          {TOTAL_TIME:.3f} s")

print()
print("Waveform points:")
print(f"  Up-ramp:        {n_up}")
print(f"  Top hold:       {n_top}")
print(f"  Down-ramp:      {n_down}")
print(f"  Final hold:     {n_final}")
print(f"  Total:          {len(waveform)}")

print()
print("ASG:")
print(f"  Frequency:      {frequency:.6f} Hz")
print(f"  Normalized min: {normalized_min:.6f}")
print(f"  Normalized max: {normalized_max:.6f}")
print(f"  Normalized final: {normalized_final:.6f}")


# ============================================================
# CONFIGURE ASG
# ============================================================
#
# IMPORTANT:
#
# We configure the numerical ASG parameters first.
#
# Then we upload our custom table.
#
# We NEVER call asg.setup() after uploading the table because
# PyRPL's _setup() calls:
#
#     self.waveform = self.waveform
#
# which regenerates the waveform table.
#
# This is exactly why some of the previous attempts lost the
# custom waveform.
# ============================================================

print()
print("Configuring ASG0...")

# Make absolutely sure output is initially disconnected.
asg.output_direct = "off"

# Configure the ASG using an ordinary waveform first.
asg.setup(
    waveform="dc",
    frequency=frequency,
    amplitude=amplitude,
    offset=offset,
    trigger_source="off",
)

# Make sure ASG is stopped while we load the waveform.
asg.on = False

# Reset state machine.
asg.sm_reset = True

# Important:
# False = after one complete waveform, stop at final value.
asg.periodic = False

# Don't wrap back to data[0].
asg._sm_wrappointer = False

# One table traversal.
asg.cycles_per_burst = 1

# Upload custom waveform.
print("Uploading custom waveform...")

asg.data = waveform


# ============================================================
# VERIFY LOCAL WAVEFORM
# ============================================================

loaded = np.asarray(
    asg.data,
    dtype=float,
)

print("Custom waveform loaded.")

print()
print("Waveform verification:")
print(f"  First value: {loaded[0]:.6f}")
print(f"  Maximum:     {np.max(loaded):.6f}")
print(f"  Final value: {loaded[-1]:.6f}")

expected_final = normalized_final

if not np.isclose(
    loaded[-1],
    expected_final,
    atol=2.0 / 2**13,
):
    raise RuntimeError(
        "Waveform verification failed."
    )


# Finish resetting the ASG.
asg.sm_reset = False


# ============================================================
# CONFIGURE SCOPE
# ============================================================

print()
print("Configuring scope...")

scope.input1 = "in1"
scope.input2 = "off"

scope.ch1_active = True
scope.ch2_active = False

scope.average = False
scope.trace_average = 1

scope.run_continuous = False
scope.rolling_mode = False

scope.decimation = SCOPE_DECIMATION

# Immediate scope acquisition.
#
# The scope starts recording first.
# We then trigger the ASG.
#
# Therefore the beginning of the waveform is captured without
# needing to synchronize the scope trigger to the ASG.
scope.trigger_source = "immediately"
scope.trigger_delay = 0

print()
print("Scope:")
print(f"  Input:           {scope.input1}")
print(f"  Decimation:      {scope.decimation}")
print(f"  Samples:         {scope.data_length}")
print(f"  Duration:        {scope.duration:.6f} s")


# ============================================================
# ARM SCOPE
# ============================================================

print()
print("Arming scope...")

scope_future = scope.single_async()

# Give the scope a small amount of time to arm.
time.sleep(0.05)


# ============================================================
# CONNECT OUT1
# ============================================================

print("Connecting ASG0 -> OUT1...")

asg.output_direct = "out1"


# ============================================================
# START ASG
# ============================================================
#
# We deliberately do NOT do:
#
#     asg.setup(... trigger_source="immediately")
#
# because that would regenerate the waveform table.
#
# Instead we directly write the ASG trigger register.
#
# From PyRPL's ASG source:
#
#     off         = 0
#     immediately = 1
#
# and ASG0 uses bits 0-2.
# ============================================================

print("Starting waveform...")

# Make sure the ASG starts from table index 0.
asg.sm_reset = True
time.sleep(0.005)
asg.sm_reset = False

# Turn ASG on.
asg.on = True

# Read control register.
control = asg._read(0x0)

# Clear trigger-source bits 0-2.
control &= ~0x7

# Set trigger source to "immediately".
control |= 0x1

# Write it back.
asg._write(0x0, control)

print("Waveform started.")


# ============================================================
# WAIT FOR ACQUISITION
# ============================================================

print()
print("Waiting for scope acquisition...")

try:

    raw_data = wait(
        scope_future,
        timeout=scope.duration + SCOPE_TIMEOUT_EXTRA,
    )

except Exception:

    print()
    print("Acquisition failed.")

    # Stop output safely.
    asg.on = False
    asg.output_direct = "off"

    raise

print("Scope acquisition complete.")


# ============================================================
# EXTRACT IN1
# ============================================================

raw_data = np.asarray(
    raw_data,
    dtype=float,
)

print()
print(f"Raw scope data shape: {raw_data.shape}")


if raw_data.ndim == 2:

    # PyRPL normally returns:
    #
    #     [channel1, channel2]
    #
    if raw_data.shape[0] == 2:

        voltage = raw_data[0]

    elif raw_data.shape[1] == 2:

        voltage = raw_data[:, 0]

    else:

        asg.on = False
        asg.output_direct = "off"

        raise RuntimeError(
            f"Unexpected scope data shape: {raw_data.shape}"
        )

elif raw_data.ndim == 1:

    voltage = raw_data

else:

    asg.on = False
    asg.output_direct = "off"

    raise RuntimeError(
        f"Unexpected scope data shape: {raw_data.shape}"
    )


voltage = np.asarray(
    voltage,
    dtype=float,
).reshape(-1)


time_data = np.asarray(
    scope.times,
    dtype=float,
).reshape(-1)


if len(voltage) != len(time_data):

    asg.on = False
    asg.output_direct = "off"

    raise RuntimeError(
        f"Time/data mismatch: "
        f"{len(time_data)} vs {len(voltage)}"
    )


# ============================================================
# RESULTS
# ============================================================

print()
print("============================================================")
print("MEASURED RESULTS")
print("============================================================")

print(f"Samples:          {len(voltage)}")
print(
    f"Time:             "
    f"{time_data[0]:.6f} -> {time_data[-1]:.6f} s"
)

print()
print(f"Measured minimum: {np.min(voltage):.6f} V")
print(f"Measured maximum: {np.max(voltage):.6f} V")

print()
print("First 10 samples:")
print(voltage[:10])

print()
print("Last 10 samples:")
print(voltage[-10:])


# ============================================================
# STOP OUTPUT
# ============================================================
#
# The important part:
#
# The waveform itself should have stopped at 0.5 V because
# periodic=False.
#
# We now disconnect OUT1 after acquisition.
# ============================================================

print()
print("Acquisition finished.")

print("Disabling OUT1...")

asg.on = False
asg.output_direct = "off"

print("OUT1 disabled.")


# ============================================================
# PLOT
# ============================================================

t = time_data - time_data[0]

plt.figure(figsize=(12, 7))

plt.plot(
    t,
    voltage,
    linewidth=1.2,
    label="IN1 measured",
)

plt.axhline(
    0.0,
    linestyle="--",
    linewidth=1,
    label="0 V",
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1,
    label="1 V",
)

plt.axhline(
    0.5,
    linestyle="--",
    linewidth=1,
    label="Final 0.5 V",
)

# Expected transition times.
plt.axvline(
    RAMP_UP_TIME,
    linestyle=":",
    linewidth=1,
)

plt.axvline(
    RAMP_UP_TIME + TOP_HOLD_TIME,
    linestyle=":",
    linewidth=1,
)

plt.axvline(
    RAMP_UP_TIME
    + TOP_HOLD_TIME
    + RAMP_DOWN_TIME,
    linestyle=":",
    linewidth=1,
)

plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")

plt.title(
    "PyRPL ASG0 → OUT1 → IN1\n"
    "0 V → 1 V → 0.5 V One-Shot Waveform"
)

plt.grid(
    True,
    linestyle="--",
    alpha=0.5,
)

plt.legend()

plt.tight_layout()

plt.show()


print()
print("============================================================")
print("TEST COMPLETE")
print("============================================================")