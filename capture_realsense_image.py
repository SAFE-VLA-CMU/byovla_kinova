#!/usr/bin/env python3
"""
RealSense Single Image Capture
Captures one image from the RealSense camera and saves it
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

def capture_single_image():
    """Capture a single image from RealSense camera"""
    
    # Create pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure streams
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    
    print("Starting RealSense camera...")
    
    try:
        # Start pipeline
        profile = pipeline.start(config)
        
        # Get device info
        device = profile.get_device()
        print(f"Device: {device.get_info(rs.camera_info.name)}")
        print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
        print(f"Firmware: {device.get_info(rs.camera_info.firmware_version)}")
        
        # Create alignment object
        align = rs.align(rs.stream.color)
        
        print("Warming up camera...")
        time.sleep(2)  # Give camera time to warm up
        
        print("Capturing frame...")
        
        # Wait for frames
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        
        # Align frames
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            print("Failed to get frames!")
            return None, None
        
        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        print(f"Captured color image: {color_image.shape}")
        print(f"Captured depth image: {depth_image.shape}")
        
        return color_image, depth_image
        
    except Exception as e:
        print(f"Error capturing image: {e}")
        return None, None
    
    finally:
        pipeline.stop()
        print("Camera stopped")

def save_images(color_image, depth_image, base_filename="realsense_capture"):
    """Save the captured images"""
    
    if color_image is None or depth_image is None:
        print("No images to save!")
        return
    
    # Create timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save color image
    color_filename = f"{base_filename}_{timestamp}_color.jpg"
    cv2.imwrite(color_filename, color_image)
    print(f"Saved color image: {color_filename}")
    
    # Save depth image (normalized for visualization)
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.05), 
        cv2.COLORMAP_JET
    )
    depth_filename = f"{base_filename}_{timestamp}_depth.jpg"
    cv2.imwrite(depth_filename, depth_colormap)
    print(f"Saved depth image: {depth_filename}")
    
    # Save raw depth data as numpy array
    depth_raw_filename = f"{base_filename}_{timestamp}_depth_raw.npy"
    np.save(depth_raw_filename, depth_image)
    print(f"Saved raw depth data: {depth_raw_filename}")
    
    # Create combined image
    # Resize images to match for display
    target_height = min(color_image.shape[0], depth_colormap.shape[0], 480)
    target_width = int(target_height * color_image.shape[1] / color_image.shape[0])
    
    color_resized = cv2.resize(color_image, (target_width, target_height))
    depth_resized = cv2.resize(depth_colormap, (target_width, target_height))
    
    # Stack horizontally
    combined_image = np.hstack((color_resized, depth_resized))
    
    # Add labels
    cv2.putText(combined_image, "Color", (10, combined_image.shape[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(combined_image, "Depth", (target_width + 10, combined_image.shape[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    combined_filename = f"{base_filename}_{timestamp}_combined.jpg"
    cv2.imwrite(combined_filename, combined_image)
    print(f"Saved combined image: {combined_filename}")
    
    return {
        'color': color_filename,
        'depth': depth_filename,
        'depth_raw': depth_raw_filename,
        'combined': combined_filename
    }

def main():
    print("RealSense Single Image Capture")
    print("=" * 40)
    
    # Check if RealSense is available
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("No RealSense devices found!")
        return
    
    print(f"Found {len(devices)} RealSense device(s)")
    
    # Capture image
    color_image, depth_image = capture_single_image()
    
    if color_image is not None and depth_image is not None:
        # Save images
        saved_files = save_images(color_image, depth_image)
        
        print("\nCapture completed successfully!")
        print("Saved files:")
        for file_type, filename in saved_files.items():
            print(f"  {file_type}: {filename}")
        
        # Print some statistics
        print(f"\nImage statistics:")
        print(f"  Color image shape: {color_image.shape}")
        print(f"  Depth image shape: {depth_image.shape}")
        print(f"  Depth range: {np.min(depth_image)} - {np.max(depth_image)} mm")
        print(f"  Color image size: {color_image.nbytes / 1024:.1f} KB")
        print(f"  Depth image size: {depth_image.nbytes / 1024:.1f} KB")
        
    else:
        print("Failed to capture image!")

if __name__ == "__main__":
    main()
