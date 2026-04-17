import rpyc
from redpid.client import RedpidClient
import time

# Replace with your Red Pitaya's actual IP address
RP_IP = "rp-f0c970.local" 

try:
    # 1. Initialize the client
    client = RedpidClient(RP_IP)
    print(f"Connected to Red Pitaya at {RP_IP}")

    # 2. Setup PID 0 (typically used for the fast feedback loop)
    pid = client.pids[0]
    
    # Configure parameters
    pid.setpoint = 0.05    # Volts
    pid.kp = 0.1           # Proportional gain
    pid.ki = 1000          # Integral gain (rad/s)
    pid.input_channel = 0  # IN1
    pid.output_channel = 0 # OUT1
    
    # 3. Engage the lock
    pid.enabled = True
    print("PID lock engaged.")

    # 4. Monitor (Optional)
    # You can loop here to print the current error or status
    for _ in range(10):
        # Depending on your redpid version, you can read the current monitor value
        print(f"Current Setpoint: {pid.setpoint} V")
        time.sleep(1)

except Exception as e:
    print(f"Error: {e}")
    print("Make sure the redpid server is running on the Red Pitaya.")