"""Launch the PyRPL GUI with a forced PyRPL FPGA reload.

Do not run this at the same time as the lock notebook (..._Master.ipynb).

This script is not very important unless you wish to test the IQ/PID modules. 

"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
USER_DIR = ROOT / "pyrpl_data"
os.makedirs(USER_DIR, exist_ok=True)
os.environ["PYRPL_USER_DIR"] = str(USER_DIR)

from pyrpl import Pyrpl

print("Launching PyRPL GUI and flashing the PyRPL FPGA...")
p = Pyrpl(
    config="my_config",
    hostname="rp-f0c970.local",
    user_dir=str(USER_DIR),
    reloadfpga=True,
    gui=True,
)
print(f"PyRPL GUI is running. Config/data: {USER_DIR}")
input("Press Enter to quit...")
