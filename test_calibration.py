#!/usr/bin/env python3
"""
Test script for camera calibration system
"""

import numpy as np
import cv2
from byovla_kinova import CameraCalibration, DepthProcessor

def test_camera_calibration():
    """Test the camera calibration system"""
    print("🧪 Testing Camera Calibration System")
    print("=" * 40)
    
    # Initialize calibration
    calibration = CameraCalibration()
    
    print("\n📷 Camera Parameters:")
    print(f"   Focal length X: {calibration.intrinsics['fx']:.2f}")
    print(f"   Focal length Y: {calibration.intrinsics['fy']:.2f}")
    print(f"   Principal point: ({calibration.intrinsics['cx']:.2f}, {calibration.intrinsics['cy']:.2f})")
    print(f"   Resolution: {calibration.intrinsics['width']}x{calibration.intrinsics['height']}")
    
    # Test coordinate transformations
    print("\n🔀 Testing Coordinate Transformations:")
    
    # Test pixel to camera coordinates
    test_pixels = [
        (320, 240),  # Center
        (0, 0),      # Top-left
        (639, 479),  # Bottom-right
        (100, 100),  # Random point
    ]
    
    test_depth = 0.5  # 50cm
    
    for u, v in test_pixels:
        camera_coords = calibration.pixel_to_camera_coords(u, v, test_depth)
        print(f"   Pixel ({u}, {v}) at depth {test_depth}m -> Camera ({camera_coords[0]:.3f}, {camera_coords[1]:.3f}, {camera_coords[2]:.3f})")
    
    # Test camera matrix
    print("\n📐 Camera Matrix K:")
    K = calibration.get_camera_matrix()
    print(K)
    
    # Test undistortion (should be identity for this camera)
    print("\n🔧 Testing Undistortion:")
    test_point = (100, 100)
    undistorted = calibration.undistort_point(*test_point)
    print(f"   Original: {test_point}")
    print(f"   Undistorted: ({undistorted[0]:.2f}, {undistorted[1]:.2f})")
    
    # Test depth processor
    print("\n📏 Testing Depth Processor:")
    depth_processor = DepthProcessor(calibration)
    
    # Create a test depth frame (simulated)
    test_depth_frame = np.ones((480, 640), dtype=np.uint16) * 500  # 50cm everywhere
    
    # Create a test mask
    test_mask = np.zeros((480, 640), dtype=np.uint8)
    test_mask[200:300, 250:350] = 1  # Center region
    
    # Test object depth
    object_depth = depth_processor.get_object_depth(test_depth_frame, test_mask)
    print(f"   Object depth: {object_depth:.3f}m")
    
    # Test 3D position calculation
    test_pixel = (300, 250)
    object_3d = depth_processor.get_object_3d_position(test_depth_frame, test_mask, test_pixel)
    if object_3d is not None:
        print(f"   Object 3D position: ({object_3d[0]:.3f}, {object_3d[1]:.3f}, {object_3d[2]:.3f})")
    
    print("\n✅ All tests completed successfully!")

def test_coordinate_mapping():
    """Test the VLA coordinate mapping system"""
    print("\n🗺️  Testing VLA Coordinate Mapping")
    print("=" * 40)
    
    from byovla_kinova import VLACoordinateMapper
    
    # Initialize components
    calibration = CameraCalibration()
    depth_processor = DepthProcessor(calibration)
    vla_mapper = VLACoordinateMapper(calibration, depth_processor)
    
    # Create a test detection
    class TestDetection:
        def __init__(self, mask):
            self.mask = [mask]
    
    # Create test mask
    test_mask = np.zeros((480, 640), dtype=np.uint8)
    test_mask[200:300, 250:350] = 1
    
    # Create test detection
    test_detection = TestDetection(test_mask)
    
    # Create test depth frame
    test_depth_frame = np.ones((480, 640), dtype=np.uint16) * 500
    
    # Test detection mapping
    print("Testing detection to action mapping...")
    result = vla_mapper.map_detection_to_action(test_detection, test_depth_frame)
    
    if result is not None:
        print(f"   Detection mapped to: {result}")
        
        # Test grasp pose calculation
        grasp_pose = vla_mapper.calculate_grasp_pose(result)
        if grasp_pose is not None:
            print(f"   Grasp pose: {grasp_pose}")
    else:
        print("   ❌ Detection mapping failed")
    
    print("✅ Coordinate mapping tests completed!")

if __name__ == "__main__":
    try:
        test_camera_calibration()
        test_coordinate_mapping()
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 