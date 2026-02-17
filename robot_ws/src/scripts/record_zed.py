#!/usr/bin/env python3

import sys
import select
import pyzed.sl as sl
import cv2
import numpy as np
from datetime import datetime
import signal

# Handle Ctrl+C gracefully
def signal_handler(sig, frame):
    print("\nExiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.camera_fps = 30                          # Set fps at 30

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera: {err}")
        return

    # Create an image object to store the image
    image_zed = sl.Mat()
    
    # Runtime parameters
    runtime_parameters = sl.RuntimeParameters()

    # Video writer setup
    video_writer = None
    is_recording = False
    
    print("\nZED Camera Recording Script")
    print("---------------------------")
    print(" Controls:")
    print("  'r' : Start recording (Press 'r' + Enter in terminal, or 'r' in window)")
    print("  's' : Stop and save recording (Press 's' + Enter, or 's' in window)")
    print("  'q' : Quit")
    print("---------------------------")

    try:
        while True:
            # Grab an image
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Retrieve the left image
                zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                
                # Get the image data as a numpy array
                # ZED images are BGRA, OpenCV uses BGR (or BGRA)
                frame_bgra = image_zed.get_data()
                
                # Convert to BGR for VideoWriter (MJPG typically expects 3 channels)
                frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
                
                # Update live view
                cv2.imshow("ZED Live View", frame_bgr)
                
                # Handle recording
                if is_recording:
                    if video_writer is None:
                        # Initialize video writer
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"recording_{timestamp}.mp4"
                        height, width = frame_bgr.shape[:2]
                        # mp4v is a safe codec for MP4 container in OpenCV
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
                        print(f"Started recording to {filename}")
                    
                    video_writer.write(frame_bgr)
                    
                    # Visual indicator for recording
                    cv2.circle(frame_bgr, (30, 30), 10, (0, 0, 255), -1) # Red dot
                    cv2.imshow("ZED Live View", frame_bgr)

                # Key handling
                key = cv2.waitKey(1) & 0xFF
                
                # Check for terminal input (non-blocking)
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline().strip()
                    if line:
                        key = ord(line[0].lower())

                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if not is_recording:
                        is_recording = True
                        print("Recording started...")
                elif key == ord('s'):
                    if is_recording:
                        is_recording = False
                        if video_writer:
                            video_writer.release()
                            video_writer = None
                        print("Recording saved.")

    finally:
        # Cleanup
        if video_writer:
            video_writer.release()
        
        image_zed.free(sl.MEM.CPU)
        zed.close()
        cv2.destroyAllWindows()
        print("Camera closed.")

if __name__ == "__main__":
    main()
