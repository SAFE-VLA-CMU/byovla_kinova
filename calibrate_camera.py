#!/usr/bin/env python3
"""
Camera-to-Robot Calibration Utility for BYOVLA with Kinova
This script helps calibrate the transformation between camera and robot coordinates.
"""

import numpy as np
import cv2
import os
import yaml
import argparse
from pathlib import Path

# Import the calibration classes from byovla_kinova
try:
    from byovla_kinova import CameraCalibration, KinovaController
except ImportError:
    print("❌ Could not import from byovla_kinova.py")
    print("   Make sure you're running this from the same directory")
    exit(1)

def create_calibration_target():
    """Create a calibration target pattern"""
    # Create a simple checkerboard pattern
    pattern_size = (7, 6)  # 7x6 internal corners
    square_size = 0.025  # 25mm squares
    
    # Generate 3D points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    return objp, pattern_size, square_size

def detect_calibration_pattern(image):
    """Detect calibration pattern in image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    pattern_size = (7, 6)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if ret:
        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners
    else:
        return False, None

def interactive_calibration(camera_calibration, robot_controller, num_points=6):
    """
    Interactive calibration using visual feedback
    """
    print(f"\n🎯 Interactive Camera-to-Robot Calibration")
    print("=" * 50)
    print("This will calibrate the transformation between camera and robot coordinates.")
    print(f"Number of calibration points: {num_points}")
    print("\nInstructions:")
    print("1. Place a calibration object (e.g., gripper tip) at known robot positions")
    print("2. Move robot to each position")
    print("3. Capture image and click on the calibration object")
    print("4. Repeat for all calibration points")
    
    try:
        import pyrealsense2 as rs
        
        # Initialize camera
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)
        
        # Warm up camera
        for i in range(5):
            pipeline.wait_for_frames()
        
        calibration_data = []
        
        for i in range(num_points):
            print(f"\n📍 Calibration Point {i+1}/{num_points}")
            print("-" * 30)
            
            # Get robot position
            print("🤖 Getting current robot position...")
            robot_pos = robot_controller.get_current_pose()
            if robot_pos is None:
                print("❌ Could not get robot position - skipping point")
                continue
            
            robot_xyz = robot_pos[:3]
            print(f"🤖 Robot position: X={robot_xyz[0]:.3f}m, Y={robot_xyz[1]:.3f}m, Z={robot_xyz[2]:.3f}m")
            
            # Capture image
            print("📸 Capturing image...")
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                print("❌ Failed to capture frames")
                continue
            
            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            
            # Rotate images if needed (matching your setup)
            color_img = cv2.rotate(color_img, cv2.ROTATE_90_CLOCKWISE)
            color_img = cv2.rotate(color_img, cv2.ROTATE_90_CLOCKWISE)
            depth_img = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)
            depth_img = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)
            
            # Get calibration point from user
            print("🖱️  Click on the calibration object center in the image...")
            print("   (Press any key after clicking)")
            
            point = get_click_point(color_img)
            if point is None:
                print("❌ No point selected - skipping")
                continue
            
            u, v = point
            
            # Get depth at that point
            depth = depth_img[v, u] * 0.001  # Convert to meters
            
            if depth < 0.1 or depth > 10.0:
                print(f"⚠️  Invalid depth: {depth:.3f}m - skipping")
                continue
            
            print(f"📍 Selected point: pixel=({u}, {v}), depth={depth:.3f}m")
            
            # Store calibration data
            calibration_data.append({
                'robot_pos': robot_xyz,
                'pixel_pos': point,
                'depth': depth
            })
            
            print(f"✅ Point {i+1} captured successfully")
            
            # Show next steps
            if i < num_points - 1:
                print("\nNext steps:")
                print("1. Move robot to a different position")
                print("2. Place calibration object at new position")
                print("3. Press Enter when ready...")
                input()
        
        pipeline.stop()
        
        if len(calibration_data) < 3:
            print("❌ Need at least 3 calibration points")
            return False
        
        # Calculate transformation
        print(f"\n🔢 Calculating transformation matrix from {len(calibration_data)} points...")
        success = calculate_camera_transform(camera_calibration, calibration_data)
        
        if success:
            print("✅ Calibration completed successfully!")
            return True
        else:
            print("❌ Calibration calculation failed")
            return False
            
    except Exception as e:
        print(f"❌ Interactive calibration failed: {e}")
        return False

def get_click_point(image):
    """Get a point by clicking on the image"""
    point = [None]
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point[0] = (x, y)
    
    cv2.namedWindow('Calibration Image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Calibration Image', mouse_callback)
    
    # Resize image for better viewing
    height, width = image.shape[:2]
    scale = min(800 / width, 600 / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized_img = cv2.resize(image, (new_width, new_height))
    
    cv2.imshow('Calibration Image', resized_img)
    cv2.resizeWindow('Calibration Image', new_width, new_height)
    
    print("Click on the calibration object and press any key...")
    
    while point[0] is None:
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            break
    
    cv2.destroyAllWindows()
    
    if point[0] is not None:
        # Convert back to original image coordinates
        x = int(point[0][0] / scale)
        y = int(point[0][1] / scale)
        return (x, y)
    
    return None

def calculate_camera_transform(camera_calibration, calibration_data):
    """
    Calculate camera-to-robot transformation from calibration data
    """
    try:
        print("Processing calibration data...")
        
        # Convert calibration data to numpy arrays
        robot_positions = np.array([data['robot_pos'] for data in calibration_data])
        pixel_positions = np.array([data['pixel_pos'] for data in calibration_data])
        depths = np.array([data['depth'] for data in calibration_data])
        
        print(f"Robot positions shape: {robot_positions.shape}")
        print(f"Pixel positions shape: {pixel_positions.shape}")
        print(f"Depths shape: {depths.shape}")
        
        # Convert pixels to camera coordinates using depth and calibrated intrinsics
        camera_positions = []
        for i, (u, v) in enumerate(pixel_positions):
            # Undistort pixel coordinates
            u_undist, v_undist = camera_calibration.undistort_point(u, v)
            
            # Convert to camera coordinates
            camera_coords = camera_calibration.pixel_to_camera_coords(u_undist, v_undist, depths[i])
            camera_positions.append(camera_coords)
            
            print(f"Point {i+1}: pixel=({u}, {v}) -> undistorted=({u_undist:.1f}, {v_undist:.1f}) -> camera=({camera_coords[0]:.3f}, {camera_coords[1]:.3f}, {camera_coords[2]:.3f})")
        
        camera_positions = np.array(camera_positions)
        
        print(f"Camera positions shape: {camera_positions.shape}")
        
        # Use SVD to find transformation matrix
        # This is a simplified approach - more robust methods exist
        camera_homog = np.column_stack([camera_positions, np.ones(len(camera_positions))])
        robot_homog = np.column_stack([robot_positions, np.ones(len(robot_positions))])
        
        # Calculate transformation matrix using pseudo-inverse
        transform = robot_homog.T @ np.linalg.pinv(camera_homog.T)
        
        # Update calibration
        camera_calibration.camera_to_robot_transform = transform
        
        print(f"\n✅ Transformation matrix calculated:")
        print(transform)
        
        # Validate transformation
        print("\n🔍 Validating transformation...")
        validation_errors = []
        
        for i, data in enumerate(calibration_data):
            # Transform camera coordinates to robot coordinates
            camera_coords = camera_positions[i]
            predicted_robot = camera_calibration.camera_to_robot_coords(camera_coords)
            actual_robot = data['robot_pos']
            
            error = np.linalg.norm(predicted_robot - actual_robot)
            validation_errors.append(error)
            
            print(f"Point {i+1}: predicted=({predicted_robot[0]:.3f}, {predicted_robot[1]:.3f}, {predicted_robot[2]:.3f}), "
                  f"actual=({actual_robot[0]:.3f}, {actual_robot[1]:.3f}, {actual_robot[2]:.3f}), "
                  f"error={error:.3f}m")
        
        mean_error = np.mean(validation_errors)
        max_error = np.max(validation_errors)
        
        print(f"\n📊 Calibration Quality:")
        print(f"   Mean error: {mean_error:.3f}m")
        print(f"   Max error: {max_error:.3f}m")
        
        if mean_error < 0.01:  # 1cm
            print("   ✅ Excellent calibration!")
        elif mean_error < 0.02:  # 2cm
            print("   ✅ Good calibration!")
        elif mean_error < 0.05:  # 5cm
            print("   ⚠️  Acceptable calibration")
        else:
            print("   ❌ Poor calibration - consider recalibrating")
        
        return True
        
    except Exception as e:
        print(f"❌ Transform calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_calibration_results(camera_calibration, output_file="camera_calibration.yaml"):
    """Save calibration results"""
    try:
        success = camera_calibration.save_calibration(output_file)
        if success:
            print(f"💾 Calibration saved to {output_file}")
            return True
        else:
            print(f"❌ Failed to save calibration to {output_file}")
            return False
    except Exception as e:
        print(f"❌ Error saving calibration: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Camera-to-Robot Calibration for BYOVLA")
    parser.add_argument("--robot-ip", default="192.168.2.9", help="Robot IP address")
    parser.add_argument("--robot-port", type=int, default=10000, help="Robot port")
    parser.add_argument("--points", type=int, default=6, help="Number of calibration points")
    parser.add_argument("--output", default="camera_calibration.yaml", help="Output calibration file")
    parser.add_argument("--load-existing", action="store_true", help="Load existing calibration file")
    
    args = parser.parse_args()
    
    print("🎯 Camera-to-Robot Calibration Utility for BYOVLA")
    print("=" * 50)
    
    # Initialize camera calibration
    print("📷 Initializing camera calibration...")
    camera_calibration = CameraCalibration()
    
    # Check if we should load existing calibration
    if args.load_existing and os.path.exists(args.output):
        print(f"📁 Loading existing calibration from {args.output}")
        if camera_calibration.load_calibration(args.output):
            print("✅ Existing calibration loaded")
            print("Current camera-to-robot transform:")
            print(camera_calibration.camera_to_robot_transform)
            
            if input("\nRecalibrate? (y/n): ").lower() != 'y':
                print("Using existing calibration")
                return
        else:
            print("❌ Failed to load existing calibration")
    
    # Initialize robot controller
    print("🤖 Initializing robot controller...")
    try:
        robot_controller = KinovaController(args.robot_ip, args.robot_port)
        
        if not robot_controller.connect():
            print("❌ Failed to connect to robot")
            print("Running in simulation mode...")
            robot_controller = None
        else:
            print("✅ Robot connected successfully")
            
    except Exception as e:
        print(f"⚠️  Robot controller initialization failed: {e}")
        print("Running in simulation mode...")
        robot_controller = None
    
    # Run calibration
    print(f"\n🚀 Starting calibration with {args.points} points...")
    
    if robot_controller is not None:
        success = interactive_calibration(camera_calibration, robot_controller, args.points)
    else:
        print("⚠️  Robot not available - cannot run interactive calibration")
        print("You can still use the camera intrinsics for coordinate transformations")
        success = True
    
    if success:
        # Save results
        save_calibration_results(camera_calibration, args.output)
        
        print("\n🎉 Calibration completed successfully!")
        print(f"📁 Results saved to: {args.output}")
        print("\nNext steps:")
        print("1. Use this calibration file in your BYOVLA system")
        print("2. Test the coordinate transformations")
        print("3. Run the full VLA pipeline")
        
    else:
        print("\n❌ Calibration failed")
        print("Check the error messages above and try again")
    
    # Cleanup
    if robot_controller is not None:
        robot_controller.disconnect()

if __name__ == "__main__":
    main() 