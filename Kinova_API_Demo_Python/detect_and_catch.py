"""
This file combines the functionality of creating a connection, getting video from the robot and getting it to catch a ball
"""
from connect import RobotConnect
from vision import BallDetector
from arm_mover import ArmMover
import cv2
import time


# Parameters
ip = "192.168.2.9"
port = 10000 # Default TCP port
credentials = ("admin", "admin") # TODO: Change to your own username-password

# Start connection
robot_connection = RobotConnect(ip, port, credentials)
robot_connection.create_connection()

# Attach the vision and movement objects to the same session
vision_process = BallDetector(robot_connection)
mover = ArmMover(robot_connection)

# Move the arm to its object tracking position
track_status = mover.object_tracking_position() # Blocking



# Automatically detect and catch the green ball
print("Starting automatic ball detection and catching...")
print("Press 'q' to quit the program")

ball_detected = False
attempts = 0
max_attempts = 100  # Prevent infinite loops

while attempts < max_attempts and not ball_detected:
    # Detect the ball
    vision_process.find_object()
    
    # Check if ball detection is stable enough for catching
    if vision_process.is_ball_detection_stable(min_confidence=5, min_duration=1.5):
        print(f"Stable ball detection at position: {vision_process.global_point}")
        print("Starting automatic catching sequence...")
        
        try:
            # Catch the detected ball
            mover.catch_target(vision_process.global_point)
            print("Ball catching sequence completed!")
            ball_detected = True
        except Exception as e:
            print(f"Error during catching sequence: {e}")
            # Reset detection and try again
            vision_process.global_point = None
            vision_process.detection_confidence = 0
    
    # Show detection status
    if attempts % 10 == 0:  # Print status every 10 attempts
        print(f"Detection attempt {attempts}/{max_attempts} - Ball detected: {vision_process.global_point is not None}")
    
    attempts += 1
    
    # Add small delay between detection attempts
    time.sleep(0.1)
    
    # Check for quit command
    if cv2.waitKey(33) == ord('q'):
        print("User requested to quit")
        break

if not ball_detected:
    print("No ball detected or catching failed after maximum attempts")

# End the program
vision_process.end_vision()
robot_connection.close_connection()
