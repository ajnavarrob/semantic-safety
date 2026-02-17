#!/usr/bin/env python3
"""
Read current servo positions and use them as zero point reference.
Prints the current tick values for both servos.
"""

from dynamixel_sdk import *
import sys

# Settings - adjust these to match your setup
DEVICENAME = '/dev/ttyUSB0'
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0
DXL_IDS = [1, 2]  # Servo IDs: 1=pitch, 2=yaw

# XC330 Control Table Addresses (Protocol 2.0)
ADDR_PRESENT_POSITION = 132

def main():
    # Initialize PortHandler and PacketHandler
    portHandler = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    # Open port
    if not portHandler.openPort():
        print("Failed to open the port")
        print(f"Make sure {DEVICENAME} is available and you have permissions")
        sys.exit(1)

    # Set baudrate
    if not portHandler.setBaudRate(BAUDRATE):
        print("Failed to change the baudrate")
        portHandler.closePort()
        sys.exit(1)

    print("Port opened successfully\n")
    print("=" * 50)
    print("Reading current servo positions as ZERO POINT")
    print("=" * 50)

    zero_positions = {}
    
    # Read current position from each servo
    for dxl_id in DXL_IDS:
        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
            portHandler, dxl_id, ADDR_PRESENT_POSITION
        )
        
        if dxl_comm_result != COMM_SUCCESS:
            print(f"ID {dxl_id}: Communication error - {packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"ID {dxl_id}: Servo error - {packetHandler.getRxPacketError(dxl_error)}")
        else:
            zero_positions[dxl_id] = dxl_present_position
            servo_name = "pitch (left/right)" if dxl_id == 1 else "yaw (up/down)"
            print(f"\nID {dxl_id} ({servo_name}):")
            print(f"  Current position: {dxl_present_position} ticks")

    # Print summary for easy copy-paste
    if len(zero_positions) == len(DXL_IDS):
        print("\n" + "=" * 50)
        print("COPY THESE VALUES TO YOUR CODE:")
        print("=" * 50)
        print("ZERO_POSITIONS = {")
        for dxl_id in sorted(zero_positions.keys()):
            servo_name = "yaw - left/right control" if dxl_id == 1 else "pitch - up/down control"
            print(f"    {dxl_id}: {zero_positions[dxl_id]},  # ID {dxl_id} zero point ({servo_name})")
        print("}")
        print("=" * 50)
    else:
        print("\n[WARNING] Could not read all servo positions")

    # Close port
    portHandler.closePort()
    print("\nPort closed")

if __name__ == "__main__":
    main()
