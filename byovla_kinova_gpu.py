#!/usr/bin/env python3
"""
BYOVLA Kinova - GPU Version
GPU-enabled version with careful memory management to avoid segmentation faults
"""

import sys
import os
import time
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import gc

# Set environment variables to control GPU memory growth
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

print("🚀 BYOVLA Kinova - GPU Version")
print("=" * 40)

# GPU setup and verification
def setup_gpu():
    """Setup and verify GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU available: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
            print(f"   Device: {device}")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            return device
        else:
            print("⚠️  CUDA not available - falling back to CPU")
            return torch.device('cpu')
    except ImportError:
        print("⚠️  PyTorch not available - falling back to CPU")
        return None
    except Exception as e:
        print(f"⚠️  GPU setup failed: {e}")
        return None

# Initialize GPU
DEVICE = setup_gpu()

# Kinova API imports (replacing WidowX)
try:
    from kortex_api.TCPTransport import TCPTransport
    from kortex_api.RouterClient import RouterClient
    from kortex_api.SessionManager import SessionManager
    from kortex_api.autogen.messages import Base_pb2, DeviceConfig_pb2, Session_pb2
    from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    KINOVA_AVAILABLE = True
    print("✅ Kinova API imported successfully")
except AttributeError as e:
    if "MutableMapping" in str(e):
        print("⚠️  Kinova API has Python 3.10 compatibility issue")
        print("   This is a known issue with older Kinova API versions")
        print("   Basic functionality will still work")
        KINOVA_AVAILABLE = False
    else:
        raise e
except ImportError as e:
    print(f"⚠️  Kinova API not available: {e}")
    KINOVA_AVAILABLE = False

# Camera support
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
    print("✅ RealSense camera support available")
except ImportError as e:
    print(f"⚠️  RealSense camera not available: {e}")
    REALSENSE_AVAILABLE = False

# VLA (Vision-Language-Action) Processing
def vla_object_detection(image, language_instruction, device=None):
    """
    VLA Object Detection using GPU-accelerated processing
    This simulates the VLA pipeline with object detection and segmentation
    """
    if device is None or device.type == 'cpu':
        return vla_object_detection_cpu(image, language_instruction)
    
    try:
        import torch
        import torch.nn.functional as F
        
        print(f"🔍 Processing VLA instruction: '{language_instruction}'")
        
        # Convert image to tensor and move to GPU
        img_tensor = torch.from_numpy(image).float().to(device) / 255.0
        
        # Simulate object detection based on language instruction
        # This is a simplified version - in a real VLA system, you'd use a proper vision-language model
        
        # Extract color information for object detection
        red_channel = img_tensor[:, :, 2]  # Red channel
        green_channel = img_tensor[:, :, 1]  # Green channel
        blue_channel = img_tensor[:, :, 0]  # Blue channel
        
        # Create masks based on language instruction
        masks = {}
        
        if "red" in language_instruction.lower():
            # Detect red objects
            red_mask = (red_channel > 0.6) & (red_channel > green_channel * 1.2) & (red_channel > blue_channel * 1.2)
            masks["red_object"] = red_mask.cpu().numpy().astype(np.uint8) * 255
            
        if "blue" in language_instruction.lower():
            # Detect blue objects
            blue_mask = (blue_channel > 0.6) & (blue_channel > red_channel * 1.2) & (blue_channel > green_channel * 1.2)
            masks["blue_object"] = blue_mask.cpu().numpy().astype(np.uint8) * 255
            
        if "green" in language_instruction.lower():
            # Detect green objects
            green_mask = (green_channel > 0.6) & (green_channel > red_channel * 1.2) & (green_channel > blue_channel * 1.2)
            masks["green_object"] = green_mask.cpu().numpy().astype(np.uint8) * 255
            
        # If no specific color mentioned, detect bright objects
        if not masks:
            brightness = (red_channel + green_channel + blue_channel) / 3
            bright_mask = brightness > 0.7
            masks["bright_object"] = bright_mask.cpu().numpy().astype(np.uint8) * 255
        
        # Apply morphological operations to clean up masks
        for key in masks:
            kernel = np.ones((5, 5), np.uint8)
            masks[key] = cv2.morphologyEx(masks[key], cv2.MORPH_CLOSE, kernel)
            masks[key] = cv2.morphologyEx(masks[key], cv2.MORPH_OPEN, kernel)
        
        # Clear GPU memory
        del img_tensor, red_channel, green_channel, blue_channel
        torch.cuda.empty_cache()
        
        return masks
        
    except Exception as e:
        print(f"❌ GPU VLA processing failed: {e}")
        return vla_object_detection_cpu(image, language_instruction)

def vla_object_detection_cpu(image, language_instruction):
    """CPU fallback for VLA object detection"""
    print(f"🔍 Processing VLA instruction (CPU): '{language_instruction}'")
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    masks = {}
    
    if "red" in language_instruction.lower():
        # Detect red objects (red wraps around in HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        masks["red_object"] = red_mask
        
    if "blue" in language_instruction.lower():
        # Detect blue objects
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        
        masks["blue_object"] = blue_mask
        
    if "green" in language_instruction.lower():
        # Detect green objects
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        masks["green_object"] = green_mask
    
    # If no specific color mentioned, detect bright objects
    if not masks:
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        
        masks["bright_object"] = bright_mask
    
    return masks

def apply_vla_processing(image, masks, device=None):
    """Apply VLA processing effects to the image based on detected objects"""
    if device is None or device.type == 'cpu':
        return apply_vla_processing_cpu(image, masks)
    
    try:
        import torch
        
        # Convert to tensor and move to GPU
        img_tensor = torch.from_numpy(image).float().to(device) / 255.0
        
        # Create processed image
        processed_img = img_tensor.clone()
        
        # Apply effects to detected objects
        for object_name, mask in masks.items():
            mask_tensor = torch.from_numpy(mask).float().to(device) / 255.0
            mask_tensor = mask_tensor.unsqueeze(-1).expand_as(img_tensor)
            
            if "red" in object_name:
                # Highlight red objects
                processed_img = torch.where(mask_tensor > 0.5, 
                                          processed_img * 1.2, 
                                          processed_img)
            elif "blue" in object_name:
                # Highlight blue objects
                processed_img = torch.where(mask_tensor > 0.5, 
                                          processed_img * 1.1, 
                                          processed_img)
            elif "green" in object_name:
                # Highlight green objects
                processed_img = torch.where(mask_tensor > 0.5, 
                                          processed_img * 1.15, 
                                          processed_img)
            else:
                # General object highlighting
                processed_img = torch.where(mask_tensor > 0.5, 
                                          processed_img * 1.1, 
                                          processed_img)
        
        # Convert back to CPU and numpy
        result = (processed_img.cpu().numpy() * 255).astype(np.uint8)
        
        # Clear GPU memory
        del img_tensor, processed_img, mask_tensor
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"❌ GPU VLA processing failed: {e}")
        return apply_vla_processing_cpu(image, masks)

def apply_vla_processing_cpu(image, masks):
    """CPU fallback for VLA processing effects"""
    processed_img = image.copy()
    
    # Apply effects to detected objects
    for object_name, mask in masks.items():
        mask_normalized = mask.astype(np.float32) / 255.0
        mask_3d = np.stack([mask_normalized] * 3, axis=2)
        
        if "red" in object_name:
            # Highlight red objects
            processed_img = processed_img * (1 + 0.2 * mask_3d)
        elif "blue" in object_name:
            # Highlight blue objects
            processed_img = processed_img * (1 + 0.1 * mask_3d)
        elif "green" in object_name:
            # Highlight green objects
            processed_img = processed_img * (1 + 0.15 * mask_3d)
        else:
            # General object highlighting
            processed_img = processed_img * (1 + 0.1 * mask_3d)
    
    return np.clip(processed_img, 0, 255).astype(np.uint8)

def generate_vla_actions(masks, language_instruction, current_pose):
    """
    Generate robot actions based on VLA analysis
    This creates a sequence of actions to interact with detected objects
    """
    actions = []
    
    print(f"🤖 Generating actions for instruction: '{language_instruction}'")
    print(f"   📊 Number of masks: {len(masks)}")
    print(f"   🤖 Current robot pose: {current_pose}")
    
    # Analyze the language instruction to determine action type
    instruction_lower = language_instruction.lower()
    print(f"   📝 Instruction keywords: {[word for word in instruction_lower.split() if word in ['pick', 'grab', 'move', 'push', 'point', 'show']]}")
    
    for object_name, mask in masks.items():
        print(f"\n   🔍 Analyzing mask: {object_name}")
        print(f"      Mask shape: {mask.shape}")
        print(f"      Mask dtype: {mask.dtype}")
        print(f"      Mask min/max: {mask.min()}/{mask.max()}")
        
        # Calculate object center and properties
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"      Number of contours found: {len(contours)}")
        
        if contours:
            # Find the largest contour (main object)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            print(f"      Largest contour area: {area:.0f} pixels")
            
            if area > 100:  # Only act on objects with sufficient size
                print(f"      ✅ Object area > 100 pixels - proceeding with action generation")
                
                # Calculate object center
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Convert pixel coordinates to robot coordinates
                    # This is a simplified conversion - in practice you'd need camera calibration
                    # Swapped y and z values for robot arm base to camera transformation
                    robot_x = current_pose['x'] + (cx - 320) * 0.001  # Approximate conversion
                    robot_z = current_pose['y'] + (cy - 240) * 0.001  # Swapped: was robot_y
                    robot_y = current_pose['z'] - 0.05  # Swapped: was robot_z, move down slightly
                    
                    print(f"      📍 Object center: pixel ({cx}, {cy}) -> robot ({robot_x:.3f}, {robot_y:.3f}, {robot_z:.3f})")
                    
                    # Generate action sequence based on instruction
                    if "pick" in instruction_lower or "grab" in instruction_lower:
                        print(f"      🎯 Generating PICK action sequence")
                        # Pick up action sequence
                        actions.extend([
                            {
                                'type': 'move_to_object',
                                'description': f'Move to {object_name}',
                                'target': (robot_x, robot_y, robot_z, 0, 0, 0),
                                'duration': 2.0
                            },
                            {
                                'type': 'grasp',
                                'description': f'Grasp {object_name}',
                                'target': (robot_x, robot_y, robot_z - 0.02, 0, 0, 0),
                                'duration': 1.0
                            },
                            {
                                'type': 'lift',
                                'description': f'Lift {object_name}',
                                'target': (robot_x, robot_y, robot_z + 0.05, 0, 0, 0),
                                'duration': 2.0
                            }
                        ])
                        
                    elif "move" in instruction_lower or "push" in instruction_lower:
                        print(f"      🎯 Generating MOVE/PUSH action sequence")
                        # Move/push action sequence
                        actions.extend([
                            {
                                'type': 'approach_object',
                                'description': f'Approach {object_name}',
                                'target': (robot_x, robot_y, robot_z + 0.02, 0, 0, 0),
                                'duration': 2.0
                            },
                            {
                                'type': 'push_object',
                                'description': f'Push {object_name}',
                                'target': (robot_x + 0.05, robot_y, robot_z, 0, 0, 0),
                                'duration': 1.5
                            }
                        ])
                        
                    elif "point" in instruction_lower or "show" in instruction_lower:
                        print(f"      🎯 Generating POINT action")
                        # Point to object action
                        actions.append({
                            'type': 'point_to_object',
                            'description': f'Point to {object_name}',
                            'target': (robot_x, robot_y, robot_z + 0.03, 0, 0, 0),
                            'duration': 1.0
                        })
                        
                    else:
                        print(f"      🎯 Generating DEFAULT action (move towards)")
                        # Default: move towards object
                        actions.append({
                            'type': 'move_towards',
                            'description': f'Move towards {object_name}',
                            'target': (robot_x, robot_y, robot_z, 0, 0, 0),
                            'duration': 2.0
                        })
                    
                    print(f"      ✅ Generated {len([a for a in actions if object_name in a['description']])} actions for {object_name}")
                else:
                    print(f"      ❌ Could not calculate object center (M['m00'] = 0)")
            else:
                print(f"      ❌ Object area ({area:.0f}) too small (< 100 pixels) - skipping")
        else:
            print(f"      ❌ No contours found in mask")
    
    print(f"\n   📋 Total actions generated: {len(actions)}")
    for i, action in enumerate(actions):
        print(f"      {i+1}. {action['description']} ({action['type']})")
    
    return actions

def execute_vla_actions(bot, actions):
    """
    Execute the generated VLA actions on the robot
    """
    if not actions:
        print("⚠️  No actions to execute")
        return False
    
    print(f"🤖 Executing {len(actions)} VLA actions...")
    
    for i, action in enumerate(actions):
        print(f"\n🔄 Action {i+1}/{len(actions)}: {action['description']}")
        print(f"   Type: {action['type']}")
        print(f"   Target: {action['target']}")
        
        try:
            # Calculate relative movement from current pose
            current_pose = bot.get_current_pose()
            if current_pose:
                dx = action['target'][0] - current_pose['x']
                dy = action['target'][1] - current_pose['y']
                dz = action['target'][2] - current_pose['z']
                dtheta_x = action['target'][3] - current_pose['theta_x']
                dtheta_y = action['target'][4] - current_pose['theta_y']
                dtheta_z = action['target'][5] - current_pose['theta_z']
                
                # Execute the movement
                success = bot.execute_action(dx, dy, dz, dtheta_x, dtheta_y, dtheta_z, action['duration'])
                
                if success:
                    print(f"   ✅ {action['description']} completed successfully")
                else:
                    print(f"   ❌ {action['description']} failed")
                    return False
                    
                # Wait a bit between actions
                time.sleep(0.5)
                
            else:
                print(f"   ❌ Could not get current pose for {action['description']}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error executing {action['description']}: {e}")
            return False
    
    print(f"\n🎉 All {len(actions)} VLA actions completed successfully!")
    return True

# GPU-accelerated image processing
def gpu_warm_filter(image, device=None):
    """Apply a warm filter using GPU acceleration"""
    if device is None or device.type == 'cpu':
        # Fallback to CPU version
        return cpu_warm_filter(image)
    
    try:
        import torch
        
        # Convert to tensor and move to GPU
        img_tensor = torch.from_numpy(image).float().to(device) / 255.0
        
        # Apply warm filter on GPU
        img_warm = img_tensor.clone()
        img_warm[:, :, 0] = torch.clamp(img_warm[:, :, 0] * 1.1, 0, 1)  # Increase red
        img_warm[:, :, 2] = torch.clamp(img_warm[:, :, 2] * 0.9, 0, 1)  # Decrease blue
        
        # Convert back to CPU and numpy
        result = (img_warm.cpu().numpy() * 255).astype(np.uint8)
        
        # Clear GPU memory
        del img_tensor, img_warm
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"❌ GPU warm filter failed: {e}")
        return cpu_warm_filter(image)

def cpu_warm_filter(image):
    """Apply a warm filter using CPU"""
    try:
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Apply warm filter (increase red channel, decrease blue channel)
        img_warm = img_float.copy()
        img_warm[:, :, 0] = np.clip(img_warm[:, :, 0] * 1.1, 0, 1)  # Increase red
        img_warm[:, :, 2] = np.clip(img_warm[:, :, 2] * 0.9, 0, 1)  # Decrease blue
        
        # Convert back to uint8
        return (img_warm * 255).astype(np.uint8)
        
    except Exception as e:
        print(f"❌ Failed to apply warm filter: {e}")
        return image

def gpu_blur_filter(image, kernel_size=15, device=None):
    """Apply Gaussian blur using GPU acceleration"""
    if device is None or device.type == 'cpu':
        # Fallback to CPU version
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    try:
        import torch
        import torch.nn.functional as F
        
        # Convert to tensor and move to GPU
        img_tensor = torch.from_numpy(image).float().to(device).permute(2, 0, 1).unsqueeze(0)
        
        # Create Gaussian kernel
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(device) / (kernel_size * kernel_size)
        
        # Apply blur to each channel
        blurred = torch.zeros_like(img_tensor)
        for i in range(3):
            blurred[:, i:i+1] = F.conv2d(img_tensor[:, i:i+1], kernel, padding=kernel_size//2)
        
        # Convert back to CPU and numpy
        result = blurred.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        # Clear GPU memory
        del img_tensor, blurred, kernel
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"❌ GPU blur filter failed: {e}")
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def gpu_edge_detection(image, device=None):
    """Apply edge detection using GPU acceleration"""
    if device is None or device.type == 'cpu':
        # Fallback to CPU version
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 50, 150)
    
    try:
        import torch
        import torch.nn.functional as F
        
        # Convert to grayscale tensor and move to GPU
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_tensor = torch.from_numpy(gray).float().to(device).unsqueeze(0).unsqueeze(0)
        
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
        
        # Apply Sobel filters
        grad_x = F.conv2d(img_tensor, sobel_x, padding=1)
        grad_y = F.conv2d(img_tensor, sobel_y, padding=1)
        
        # Compute gradient magnitude
        magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Convert back to CPU and numpy
        result = magnitude.squeeze().cpu().numpy().astype(np.uint8)
        
        # Clear GPU memory
        del img_tensor, grad_x, grad_y, magnitude, sobel_x, sobel_y
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"❌ GPU edge detection failed: {e}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 50, 150)

# Kinova Robot Controller Class
class KinovaController:
    """Kinova robot controller replacing WidowX functionality"""
    
    def __init__(self, ip="192.168.2.9", port=10000, credentials=("admin", "admin")):
        self.ip = ip
        self.port = port
        self.credentials = credentials
        
        # Connection objects
        self.transport = None
        self.router = None
        self.session_manager = None
        self.base = None
        
        # Movement parameters
        self.TIMEOUT_DURATION = 20
        
        print(f"🤖 Kinova Controller for BYOVLA")
        print(f"   IP: {self.ip}")
        print(f"   Port: {self.port}")
        print(f"   Credentials: {self.credentials}")
        print("=" * 40)
    
    def connect(self):
        """Connect to Kinova arm"""
        try:
            print("🔌 Connecting to Kinova robot...")
            
            # Set up API
            self.transport = TCPTransport()
            self.transport.connect(self.ip, self.port)
            self.router = RouterClient(self.transport)
            
            # Create session
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 60000
            session_info.connection_inactivity_timeout = 2000
            
            print("📝 Creating session...")
            self.session_manager = SessionManager(self.router)
            session_handle = self.session_manager.CreateSession(session_info)
            
            # Create base client
            self.base = BaseClient(self.router)
            
            print("✅ Connected to Kinova robot successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Kinova robot: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Kinova arm"""
        try:
            if self.session_manager:
                self.session_manager.CloseSession()
            if self.transport:
                self.transport.disconnect()
            print("🔌 Disconnected from Kinova robot")
        except Exception as e:
            print(f"⚠️  Error during disconnect: {e}")
    
    def get_current_pose(self):
        """Get current pose of the robot"""
        try:
            pose = self.base.GetMeasuredCartesianPose()
            return {
                'x': pose.x,
                'y': pose.y,
                'z': pose.z,
                'theta_x': pose.theta_x,
                'theta_y': pose.theta_y,
                'theta_z': pose.theta_z
            }
        except Exception as e:
            print(f"❌ Failed to get current pose: {e}")
            return None
    
    def execute_action(self, dx, dy, dz, dtheta_x, dtheta_y, dtheta_z, duration=1.0):
        """Execute a relative movement action"""
        try:
            # Get current pose
            current_pose = self.base.GetMeasuredCartesianPose()
            
            # Calculate target pose using the correct protobuf structure
            target_pose = Base_pb2.Pose()
            target_pose.x = current_pose.x + dx
            target_pose.y = current_pose.y + dy
            target_pose.z = current_pose.z + dz
            target_pose.theta_x = current_pose.theta_x + dtheta_x
            target_pose.theta_y = current_pose.theta_y + dtheta_y
            target_pose.theta_z = current_pose.theta_z + dtheta_z
            
            # Execute movement using the correct method
            self.base.SendSelectedToolForConstrainedMotion(target_pose)
            
            # Wait for movement to complete
            time.sleep(duration)
            
            return True
            
        except AttributeError:
            # Fallback method if SendSelectedToolForConstrainedMotion doesn't exist
            try:
                # Try alternative movement method
                action = Base_pb2.Action()
                action.reach_pose.target_pose.x = current_pose.x + dx
                action.reach_pose.target_pose.y = current_pose.y + dy
                action.reach_pose.target_pose.z = current_pose.z + dz
                action.reach_pose.target_pose.theta_x = current_pose.theta_x + dtheta_x
                action.reach_pose.target_pose.theta_y = current_pose.theta_y + dtheta_y
                action.reach_pose.target_pose.theta_z = current_pose.theta_z + dtheta_z
                
                self.base.ExecuteAction(action)
                time.sleep(duration)
                return True
                
            except Exception as e2:
                print(f"❌ Failed to execute action (fallback): {e2}")
                return False
        except Exception as e:
            print(f"❌ Failed to execute action: {e}")
            return False
    
    def reset_to_home_position(self):
        """Reset the arm to a safe home position"""
        try:
            print("🏠 Resetting arm to home position...")
            
            # Define safe home position (adjust these values based on your setup)
            home_x = 0.4  # Forward position
            home_y = 0.0  # Center position
            home_z = 0.3  # Elevated position
            home_theta_x = -180.0  # Gripper pointing down
            home_theta_y = 0.0     # No pitch
            home_theta_z = 90.0    # Gripper oriented forward
            
            # Get current pose
            current_pose = self.base.GetMeasuredCartesianPose()
            
            # Calculate movement to home position
            dx = home_x - current_pose.x
            dy = home_y - current_pose.y
            dz = home_z - current_pose.z
            dtheta_x = home_theta_x - current_pose.theta_x
            dtheta_y = home_theta_y - current_pose.theta_y
            dtheta_z = home_theta_z - current_pose.theta_z
            
            print(f"   📍 Current: ({current_pose.x:.3f}, {current_pose.y:.3f}, {current_pose.z:.3f})")
            print(f"   🏠 Target:  ({home_x:.3f}, {home_y:.3f}, {home_z:.3f})")
            print(f"   📏 Movement: ({dx:.3f}, {dy:.3f}, {dz:.3f})")
            
            # Execute movement to home position
            success = self.execute_action(dx, dy, dz, dtheta_x, dtheta_y, dtheta_z, 3.0)
            
            if success:
                print("✅ Arm reset to home position successfully")
                return True
            else:
                print("❌ Failed to reset arm to home position")
                return False
                
        except Exception as e:
            print(f"❌ Error during arm reset: {e}")
            return False
    
    def move_to_absolute_position(self, x, y, z, theta_x, theta_y, theta_z, duration=3.0):
        """Move to an absolute position"""
        try:
            # Get current pose
            current_pose = self.base.GetMeasuredCartesianPose()
            
            # Calculate relative movement
            dx = x - current_pose.x
            dy = y - current_pose.y
            dz = z - current_pose.z
            dtheta_x = theta_x - current_pose.theta_x
            dtheta_y = theta_y - current_pose.theta_y
            dtheta_z = theta_z - current_pose.theta_z
            
            # Execute movement
            return self.execute_action(dx, dy, dz, dtheta_x, dtheta_y, dtheta_z, duration)
            
        except Exception as e:
            print(f"❌ Failed to move to absolute position: {e}")
            return False

def init_camera():
    """Initialize RealSense camera"""
    if not REALSENSE_AVAILABLE:
        return None
    
    try:
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        pipeline.start(config)
        print("📷 RealSense camera initialized")
        return pipeline
        
    except Exception as e:
        print(f"❌ Failed to initialize camera: {e}")
        return None

def take_picture(pipeline):
    """Take a picture using RealSense camera"""
    if pipeline is None:
        return None
    
    try:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            print("❌ No color frame received")
            return None
        
        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        return color_image
        
    except Exception as e:
        print(f"❌ Failed to take picture: {e}")
        return None

def display_vla_results(original_img, vla_processed_img, masks, title="BYOVLA VLA Results"):
    """Display the VLA results using matplotlib"""
    try:
        n_masks = len(masks)
        n_images = 2 + n_masks  # original + processed + masks
        
        # Create a more compact layout
        cols = min(3, n_images)  # Max 3 columns
        rows = (n_images + cols - 1) // cols  # Calculate needed rows
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # VLA processed image
        if cols > 1:
            axes[0, 1].imshow(cv2.cvtColor(vla_processed_img, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title("VLA Processed")
            axes[0, 1].axis('off')
        else:
            axes[1, 0].imshow(cv2.cvtColor(vla_processed_img, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title("VLA Processed")
            axes[1, 0].axis('off')
        
        # Display masks
        mask_idx = 0
        for row in range(rows):
            for col in range(cols):
                if (row == 0 and col < 2) or mask_idx >= len(masks):
                    continue
                
                object_name = list(masks.keys())[mask_idx]
                mask = masks[object_name]
                axes[row, col].imshow(mask, cmap='gray')
                axes[row, col].set_title(f"Mask: {object_name}")
                axes[row, col].axis('off')
                mask_idx += 1
        
        # Hide unused subplots
        for row in range(rows):
            for col in range(cols):
                if (row == 0 and col < 2) or mask_idx < len(masks):
                    continue
                axes[row, col].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
        # Also print a summary of detected objects
        print(f"\n📋 VLA Detection Summary:")
        print(f"   📸 Original image size: {original_img.shape}")
        print(f"   🎯 Objects detected: {len(masks)}")
        for object_name, mask in masks.items():
            # Calculate object properties
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                print(f"      • {object_name}: {area:.0f} pixels")
            else:
                print(f"      • {object_name}: no contours found")
        
    except Exception as e:
        print(f"❌ Failed to display VLA results: {e}")
        import traceback
        traceback.print_exc()

def save_vla_results(original_img, vla_processed_img, masks, results_dir="./vla_results"):
    """Save the VLA results to files"""
    try:
        # Create results directory
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Save original and processed images
        cv2.imwrite(f"{results_dir}/original.jpg", original_img)
        cv2.imwrite(f"{results_dir}/vla_processed.jpg", vla_processed_img)
        
        # Save masks
        for object_name, mask in masks.items():
            filename = f"{results_dir}/mask_{object_name}.jpg"
            cv2.imwrite(filename, mask)
        
        print(f"💾 VLA results saved to {results_dir}")
        
    except Exception as e:
        print(f"❌ Failed to save VLA results: {e}")

def display_results(original_img, processed_imgs, titles, title="BYOVLA GPU Results"):
    """Display the results using matplotlib"""
    try:
        n_images = len(processed_imgs) + 1  # +1 for original
        fig, axes = plt.subplots(1, n_images, figsize=(4*n_images, 4))
        
        if n_images == 1:
            axes = [axes]
        
        # Original image
        axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Processed images
        for i, (img, title) in enumerate(zip(processed_imgs, titles)):
            if len(img.shape) == 2:  # Grayscale
                axes[i+1].imshow(img, cmap='gray')
            else:  # Color
                axes[i+1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i+1].set_title(title)
            axes[i+1].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Failed to display results: {e}")

def save_results(original_img, processed_imgs, titles, results_dir="./gpu_results"):
    """Save the results to files"""
    try:
        # Create results directory
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Save original image
        cv2.imwrite(f"{results_dir}/original.jpg", original_img)
        
        # Save processed images
        for img, title in zip(processed_imgs, titles):
            filename = f"{results_dir}/{title.lower().replace(' ', '_')}.jpg"
            cv2.imwrite(filename, img)
        
        print(f"💾 Results saved to {results_dir}")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

def main():
    """Main function"""
    print("🚀 BYOVLA with Kinova Robot - GPU Version")
    print("=" * 50)
    
    # Initialize Kinova robot
    bot = None
    if KINOVA_AVAILABLE:
        try:
            bot = KinovaController()
            if not bot.connect():
                print("❌ Failed to connect to Kinova robot - running in simulation mode")
                bot = None
            else:
                # Reset arm to home position every time we run
                print("\n🏠 Initializing arm position...")
                bot.reset_to_home_position()
        except Exception as e:
            print(f"⚠️  Kinova controller creation failed: {e}")
            print("   Running in simulation mode")
            bot = None
    
    try:
        # Camera Setup
        pipeline = None
        if REALSENSE_AVAILABLE:
            pipeline = init_camera()
        
        # Take picture
        if pipeline:
            original_img = take_picture(pipeline)
            if original_img is not None:
                print("📸 Image captured successfully")
            else:
                print("⚠️  Failed to capture image - using placeholder")
                original_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        else:
            print("⚠️  No camera available - using placeholder image")
            original_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # VLA (Vision-Language-Action) Processing
        print("\n🔍 Starting VLA (Vision-Language-Action) processing...")
        language_instruction = "Pick up the red object"  # Example instruction
        
        # Object detection using VLA
        masks = vla_object_detection(original_img, language_instruction, DEVICE)
        print(f"✅ Detected {len(masks)} objects: {list(masks.keys())}")
        
        # Apply VLA processing
        vla_processed_img = apply_vla_processing(original_img, masks, DEVICE)
        print("✅ VLA processing completed")
        
        # Save VLA results
        save_vla_results(original_img, vla_processed_img, masks)
        
        # Display VLA results
        print("📊 Displaying VLA results...")
        display_vla_results(original_img, vla_processed_img, masks)
        
        # Generate and execute VLA actions
        if bot is not None and masks:
            current_pose = bot.get_current_pose()
            if current_pose:
                # Generate actions based on VLA analysis
                actions = generate_vla_actions(masks, language_instruction, current_pose)
                
                if actions:
                    # Execute the generated actions
                    execute_vla_actions(bot, actions)
                else:
                    print("⚠️  No actions generated from VLA analysis")
            else:
                print("❌ Could not get current pose for VLA action generation")
        else:
            print("⚠️  Skipping VLA action execution (robot not available or no objects detected)")
        
        # GPU-accelerated image processing
        print("\n🖼️  Processing image with GPU acceleration...")
        
        processed_imgs = []
        titles = []
        
        # Warm filter
        warm_img = gpu_warm_filter(original_img, DEVICE)
        processed_imgs.append(warm_img)
        titles.append("Warm Filter")
        print("✅ Warm filter applied")
        
        # Blur filter
        blur_img = gpu_blur_filter(original_img, kernel_size=15, device=DEVICE)
        processed_imgs.append(blur_img)
        titles.append("Gaussian Blur")
        print("✅ Gaussian blur applied")
        
        # Edge detection
        edge_img = gpu_edge_detection(original_img, DEVICE)
        processed_imgs.append(edge_img)
        titles.append("Edge Detection")
        print("✅ Edge detection applied")
        
        # Save results (without displaying)
        save_results(original_img, processed_imgs, titles)
        
        # VLA actions have already been executed above
        print("\n🤖 VLA robot actions completed!")
        
        print("\n🎉 BYOVLA with Kinova GPU processing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if bot is not None:
            bot.disconnect()
        
        # Clear GPU memory
        if DEVICE and DEVICE.type == 'cuda':
            try:
                import torch
                torch.cuda.empty_cache()
                print("🧹 GPU memory cleared")
            except:
                pass
        
        print("\n🧹 Cleanup completed")

if __name__ == "__main__":
    main()
