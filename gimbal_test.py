from dynamixel_sdk import *
from pynput import keyboard
import time
import sys

# Settings - adjust these to match your setup
DEVICENAME = '/dev/ttyUSB0'
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0
DXL_IDS = [1, 2]

# XC330 Control Table Addresses (Protocol 2.0)
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

# Zero position offsets (calibrated home position)
ZERO_POSITIONS = {
    1: 1617,  # ID 1 zero point (yaw - left/right control)
    2: 1757   # ID 2 zero point (pitch - up/down control)
}

# Movement increment (in ticks)
STEP_SIZE = 20

# Current goal positions (start at zero)
goal_positions = {
    1: ZERO_POSITIONS[1],
    2: ZERO_POSITIONS[2]
}

# Initialize PortHandler and PacketHandler
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if not portHandler.openPort():
    print("Failed to open the port")
    quit()

# Set baudrate
if not portHandler.setBaudRate(BAUDRATE):
    print("Failed to change the baudrate")
    quit()

print("Port opened successfully")

# Enable torque on both servos
for dxl_id in DXL_IDS:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
        portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE
    )
    if dxl_comm_result != COMM_SUCCESS:
        print(f"ID {dxl_id}: {packetHandler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"ID {dxl_id}: {packetHandler.getRxPacketError(dxl_error)}")
    else:
        print(f"Torque enabled on ID {dxl_id}")

# Move to zero position initially
for dxl_id in DXL_IDS:
    packetHandler.write4ByteTxRx(
        portHandler, dxl_id, ADDR_GOAL_POSITION, ZERO_POSITIONS[dxl_id]
    )
print("Servos moved to zero position\n")

def update_goal_position(dxl_id, position):
    """Send goal position to servo"""
    # Clamp to valid range
    position = max(0, min(4095, position))
    goal_positions[dxl_id] = position
    
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(
        portHandler, dxl_id, ADDR_GOAL_POSITION, position
    )
    return position

def on_press(key):
    """Handle key press events"""
    try:
        if key == keyboard.Key.up:
            # Up arrow - increase ID 2 (yaw up)
            new_pos = update_goal_position(2, goal_positions[2] + STEP_SIZE)
            print(f"↑ ID 2: {new_pos - ZERO_POSITIONS[2]:+5d} ticks")
        elif key == keyboard.Key.down:
            # Down arrow - decrease ID 2 (yaw down)
            new_pos = update_goal_position(2, goal_positions[2] - STEP_SIZE)
            print(f"↓ ID 2: {new_pos - ZERO_POSITIONS[2]:+5d} ticks")
        elif key == keyboard.Key.left:
            # Left arrow - decrease ID 1 (pitch left)
            new_pos = update_goal_position(1, goal_positions[1] - STEP_SIZE)
            print(f"← ID 1: {new_pos - ZERO_POSITIONS[1]:+5d} ticks")
        elif key == keyboard.Key.right:
            # Right arrow - increase ID 1 (pitch right)
            new_pos = update_goal_position(1, goal_positions[1] + STEP_SIZE)
            print(f"→ ID 1: {new_pos - ZERO_POSITIONS[1]:+5d} ticks")
        elif hasattr(key, 'char') and key.char == 'z':
            # Z key - return to zero position
            for dxl_id in DXL_IDS:
                update_goal_position(dxl_id, ZERO_POSITIONS[dxl_id])
            print(f"⌂ Returned to zero position")
    except Exception as e:
        print(f"Error: {e}")

def on_release(key):
    """Handle key release - exit on ESC"""
    if key == keyboard.Key.esc:
        print("\nESC pressed - stopping...")
        return False

# Start keyboard listener
print("Arrow key control active:")
print("  ↑/↓ : Control ID 2 (yaw)")
print("  ←/→ : Control ID 1 (pitch)")
print("  Z   : Return to zero position")
print("  ESC : Exit\n")

running = True
try:
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
except KeyboardInterrupt:
    print("\nCtrl+C pressed - stopping...")

# Disable torque and close port
print("\nDisabling torque...")
try:
    for dxl_id in DXL_IDS:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        print(f"Torque disabled on ID {dxl_id}")
except Exception as e:
    print(f"Warning: Error during cleanup: {e}")

try:
    portHandler.closePort()
    print("Port closed")
except Exception as e:
    print(f"Warning: Error closing port: {e}")
