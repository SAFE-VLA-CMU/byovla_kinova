#!/usr/bin/env python3
"""
RealSense Camera Test Script (Adaptive)
Automatically uses available stream profiles from the connected camera
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import time

def get_best_stream_config(device):
    """Find the best available stream configuration for the device"""
    depth_profiles = []
    color_profiles = []
    
    for sensor in device.query_sensors():
        sensor_name = sensor.get_info(rs.camera_info.name)
        
        try:
            stream_profiles = sensor.get_stream_profiles()
            for profile in stream_profiles:
                video_profile = profile.as_video_stream_profile()
                
                if profile.stream_type() == rs.stream.depth:
                    if profile.format() == rs.format.z16:
                        depth_profiles.append({
                            'width': video_profile.width(),
                            'height': video_profile.height(),
                            'fps': video_profile.fps(),
                            'format': profile.format()
                        })
                
                elif profile.stream_type() == rs.stream.color:
                    if profile.format() in [rs.format.bgr8, rs.format.rgb8]:
                        color_profiles.append({
                            'width': video_profile.width(),
                            'height': video_profile.height(),
                            'fps': video_profile.fps(),
                            'format': profile.format()
                        })
        
        except Exception as e:
            print(f"Error getting profiles for {sensor_name}: {e}")
    
    # Sort by preference (resolution and fps)
    depth_profiles.sort(key=lambda x: (x['width'] * x['height'], x['fps']), reverse=True)
    color_profiles.sort(key=lambda x: (x['width'] * x['height'], x['fps']), reverse=True)
    
    print(f"\nFound {len(depth_profiles)} depth profiles and {len(color_profiles)} color profiles")
    
    # Show available profiles
    print("\nDepth profiles (best first):")
    for i, p in enumerate(depth_profiles[:5]):
        print(f"  {i+1}: {p['width']}x{p['height']} @ {p['fps']}fps")
    
    print("\nColor profiles (best first):")
    for i, p in enumerate(color_profiles[:5]):
        print(f"  {i+1}: {p['width']}x{p['height']} @ {p['fps']}fps")
    
    return depth_profiles, color_profiles

def try_stream_combination(pipeline, depth_profile, color_profile):
    """Try a specific combination of depth and color streams"""
    config = rs.config()
    
    try:
        config.enable_stream(
            rs.stream.depth, 
            depth_profile['width'], 
            depth_profile['height'], 
            depth_profile['format'], 
            depth_profile['fps']
        )
        
        # Convert RGB to BGR if needed
        color_format = rs.format.bgr8 if color_profile['format'] == rs.format.rgb8 else color_profile['format']
        
        config.enable_stream(
            rs.stream.color, 
            color_profile['width'], 
            color_profile['height'], 
            color_format, 
            color_profile['fps']
        )
        
        pipeline.start(config)
        return True, config
        
    except Exception as e:
        print(f"    Failed: {e}")
        try:
            pipeline.stop()
        except:
            pass
        return False, None

def main():
    # List devices
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("No RealSense devices found!")
        return
    
    device = devices[0]
    print(f"Using device: {device.get_info(rs.camera_info.name)}")
    print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
    print(f"Firmware: {device.get_info(rs.camera_info.firmware_version)}")
    
    # Get available stream configurations
    depth_profiles, color_profiles = get_best_stream_config(device)
    
    if not depth_profiles or not color_profiles:
        print("No compatible stream profiles found!")
        return
    
    # Try different combinations
    pipeline = rs.pipeline()
    config = None
    
    print(f"\nTrying stream combinations...")
    
    combinations_to_try = [
        # Try matching resolutions first
        (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),  # Same index
        # Then try different combinations
        (0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)
    ]
    
    success = False
    used_depth = None
    used_color = None
    
    for depth_idx, color_idx in combinations_to_try:
        if depth_idx >= len(depth_profiles) or color_idx >= len(color_profiles):
            continue
        
        depth_profile = depth_profiles[depth_idx]
        color_profile = color_profiles[color_idx]
        
        print(f"  Trying depth {depth_profile['width']}x{depth_profile['height']}@{depth_profile['fps']} "
              f"+ color {color_profile['width']}x{color_profile['height']}@{color_profile['fps']}")
        
        success, config = try_stream_combination(pipeline, depth_profile, color_profile)
        if success:
            used_depth = depth_profile
            used_color = color_profile
            print(f"    ✓ Success!")
            break
    
    if not success:
        print("Could not find a working stream combination!")
        return
    
    print(f"\nUsing configuration:")
    print(f"  Depth: {used_depth['width']}x{used_depth['height']} @ {used_depth['fps']}fps")
    print(f"  Color: {used_color['width']}x{used_color['height']} @ {used_color['fps']}fps")
    
    try:
        print("\nCamera started successfully!")
        print("Warming up... (this may take a few seconds)")
        
        # Warm up with more patience
        warmup_frames = 0
        max_warmup_attempts = 100
        
        for attempt in range(max_warmup_attempts):
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)  # Longer timeout for warmup
                if frames.size() >= 2:  # Both depth and color
                    warmup_frames += 1
                    if warmup_frames >= 5:  # Got 5 good frames
                        break
                print(f"Warmup progress: {warmup_frames}/5 frames", end='\r')
            except Exception as e:
                if attempt % 10 == 0:
                    print(f"Warmup attempt {attempt+1}/{max_warmup_attempts}...")
                continue
        
        if warmup_frames < 5:
            print(f"Warning: Only got {warmup_frames} frames during warmup")
        else:
            print("Warmup complete!                    ")
        
        print("\nControls:")
        print("  'q' = quit")
        print("  's' = save frame")
        print("  'i' = show info")
        print("  'r' = reset/restart")
        
        # Create alignment object
        align = rs.align(rs.stream.color)
        
        frame_count = 0
        last_fps_time = time.time()
        fps = 0
        consecutive_failures = 0
        max_failures = 10
        
        while True:
            try:
                # Wait for frames
                frames = pipeline.wait_for_frames(timeout_ms=2000)
                
                # Reset failure counter on success
                consecutive_failures = 0
                
                # Align frames
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Convert to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Handle different color formats
                if len(color_image.shape) == 3 and color_image.shape[2] == 3:
                    # Already BGR or RGB
                    if used_color['format'] == rs.format.rgb8:
                        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                
                # Apply colormap to depth
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.05), 
                    cv2.COLORMAP_JET
                )
                
                # Resize images to match for display
                target_height = min(color_image.shape[0], depth_colormap.shape[0], 480)
                target_width = int(target_height * color_image.shape[1] / color_image.shape[0])
                
                color_resized = cv2.resize(color_image, (target_width, target_height))
                depth_resized = cv2.resize(depth_colormap, (target_width, target_height))
                
                # Stack horizontally
                display_image = np.hstack((color_resized, depth_resized))
                
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = frame_count / (current_time - last_fps_time)
                    frame_count = 0
                    last_fps_time = current_time
                
                # Add overlays
                cv2.putText(display_image, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_image, f"Depth: {used_depth['width']}x{used_depth['height']}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_image, f"Color: {used_color['width']}x{used_color['height']}", 
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.putText(display_image, "Color", (10, display_image.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_image, "Depth", (target_width + 10, display_image.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display
                cv2.namedWindow('RealSense D435 Test', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense D435 Test', display_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"realsense_d435_{timestamp}.png"
                    cv2.imwrite(filename, display_image)
                    print(f"Saved: {filename}")
                elif key == ord('i'):
                    print(f"\nCurrent stream info:")
                    print(f"  Display FPS: {fps:.1f}")
                    print(f"  Color shape: {color_image.shape}")
                    print(f"  Depth shape: {depth_image.shape}")
                    print(f"  Depth range: {np.min(depth_image)} - {np.max(depth_image)}")
                elif key == ord('r'):
                    print("Restarting streams...")
                    pipeline.stop()
                    time.sleep(1)
                    pipeline.start(config)
                    print("Restarted!")
            
            except Exception as e:
                consecutive_failures += 1
                print(f"Frame error ({consecutive_failures}/{max_failures}): {e}")
                
                if consecutive_failures >= max_failures:
                    print("Too many consecutive failures, exiting...")
                    break
                
                # Brief pause before retry
                time.sleep(0.1)
                continue
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            pipeline.stop()
        except:
            pass
        cv2.destroyAllWindows()
        print("Camera stopped and cleanup complete")

if __name__ == "__main__":
    print("RealSense D435 Adaptive Test Script")
    print("=" * 40)
    main()