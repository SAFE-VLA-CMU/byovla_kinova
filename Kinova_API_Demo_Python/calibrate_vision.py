"""
Vision System Calibration Script
Use this to calibrate the coordinate mapping for better ball grasping accuracy
"""
from connect import RobotConnect
from vision import BallDetector
import cv2
import numpy as np

def main():
    # Connection parameters
    ip = "192.168.2.9"
    port = 10000
    credentials = ("admin", "admin")
    
    print("=== Vision System Calibration ===")
    print("This script helps calibrate the coordinate mapping for accurate ball grasping")
    print()
    
    # Connect to robot
    print("Connecting to robot...")
    robot_connection = RobotConnect(ip, port, credentials)
    robot_connection.create_connection()
    
    # Initialize vision system
    vision_process = BallDetector(robot_connection)
    
    print("\n=== CALIBRATION INSTRUCTIONS ===")
    print("1. Place a green ball at a KNOWN position on the table")
    print("2. Note the ball's position relative to the robot base")
    print("3. Use the calibration commands below to adjust the mapping")
    print("4. Test until the calculated position matches the known position")
    print()
    
    # Show current calibration
    vision_process.test_calibration()
    
    print("=== AVAILABLE COMMANDS ===")
    print("Commands you can type:")
    print("  'test' - Test current calibration")
    print("  'detect' - Detect ball and show coordinates")
    print("  'cal x_scale y_scale x_offset y_offset' - Update calibration")
    print("  'reset' - Reset to default calibration")
    print("  'quit' - Exit calibration")
    print()
    
    while True:
        try:
            command = input("Enter command: ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'test':
                vision_process.test_calibration()
            elif command == 'detect':
                print("Detecting ball...")
                vision_process.find_object()
                if vision_process.global_point is not None:
                    print(f"Ball detected at: {vision_process.global_point}")
                else:
                    print("No ball detected")
            elif command.startswith('cal '):
                try:
                    parts = command.split()
                    if len(parts) == 5:
                        x_scale = float(parts[1])
                        y_scale = float(parts[2])
                        x_offset = float(parts[3])
                        y_offset = float(parts[4])
                        vision_process.calibrate_coordinates(x_scale, y_scale, x_offset, y_offset)
                    else:
                        print("Usage: cal x_scale y_scale x_offset y_offset")
                        print("Example: cal 0.12 0.12 0.05 -0.02")
                except ValueError:
                    print("Invalid numbers. Use format: cal x_scale y_scale x_offset y_offset")
            elif command == 'reset':
                vision_process.calibrate_coordinates(0.15, 0.15, 0.0, 0.0)
                print("Reset to default calibration")
            else:
                print("Unknown command. Type 'test' for available commands.")
                
        except KeyboardInterrupt:
            print("\nCalibration interrupted")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    # Cleanup
    print("Closing connection...")
    vision_process.end_vision()
    robot_connection.close_connection()
    print("Calibration complete!")

if __name__ == "__main__":
    main() 