import os

# 1. Fixing some issue with the pyrpl bullshit code
# Get the directory where THIS script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
local_user_dir = os.path.join(current_dir, "pyrpl_data")

# Set the environment variable BEFORE importing pyrpl
os.environ['PYRPL_USER_DIR'] = local_user_dir

# 3. NOW IMPORT PYRPL
from pyrpl import Pyrpl

# 4. LAUNCH
# We still pass user_dir just to be safe, but the environment variable 
p = Pyrpl(config='my_config', 
          hostname='rp-f0ac1b.local', 
          user_dir=local_user_dir)

print(f"Success! PyRPL is contained. Data is at: {local_user_dir}")
input("Press Enter to quit...")


from pyrpl import Pyrpl

# Get the directory where THIS script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Define a local folder for all the PyRPL clutter
local_user_dir = os.path.join(current_dir, "pyrpl_data")

# Launch PyRPL and force it to use the local folder
p = Pyrpl(config='my_config', 
          hostname='rp-f0ac1b.local', #f0c970
          user_dir=local_user_dir)

print(f"PyRPL is running. Data is being saved to: {local_user_dir}")
input("Press Enter to quit...")