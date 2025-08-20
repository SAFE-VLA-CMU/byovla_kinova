#!/usr/bin/env python3
"""
BYOVLA Kinova - Safe Version
Simplified version that avoids problematic imports causing segmentation faults
"""

import sys
import os
import time
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Core imports that are known to work
print("🚀 BYOVLA Kinova - Safe Version")
print("=" * 40)

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
            
            # Calculate target pose
            target_pose = Base_pb2.CartesianPose()
            target_pose.x = current_pose.x + dx
            target_pose.y = current_pose.y + dy
            target_pose.z = current_pose.z + dz
            target_pose.theta_x = current_pose.theta_x + dtheta_x
            target_pose.theta_y = current_pose.theta_y + dtheta_y
            target_pose.theta_z = current_pose.theta_z + dtheta_z
            
            # Execute movement
            self.base.SendSelectedToolForConstrainedMotion(target_pose)
            
            # Wait for movement to complete
            time.sleep(duration)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to execute action: {e}")
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

def warm_filter(image):
    """Apply a warm filter to the image"""
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

def display_results(original_img, processed_img, title="BYOVLA Results"):
    """Display the results using matplotlib"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        ax2.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        ax2.set_title("Processed Image")
        ax2.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Failed to display results: {e}")

def save_results(original_img, processed_img, results_dir="./results"):
    """Save the results to files"""
    try:
        # Create results directory
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Save images
        cv2.imwrite(f"{results_dir}/original.jpg", original_img)
        cv2.imwrite(f"{results_dir}/processed.jpg", processed_img)
        
        print(f"💾 Results saved to {results_dir}")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

def main():
    """Main function"""
    print("🚀 BYOVLA with Kinova Robot - Safe Version")
    print("=" * 50)
    
    # Initialize Kinova robot
    bot = None
    if KINOVA_AVAILABLE:
        try:
            bot = KinovaController()
            if not bot.connect():
                print("❌ Failed to connect to Kinova robot - running in simulation mode")
                bot = None
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
        
        # Process image
        print("🖼️  Processing image...")
        processed_img = warm_filter(original_img)
        
        # Display results
        print("📊 Displaying results...")
        display_results(original_img, processed_img)
        
        # Save results
        save_results(original_img, processed_img)
        
        # Robot movement demonstration
        if bot is not None:
            print("\n🤖 Running robot actions with Kinova...")
            
            # Get current pose
            curr_pose = bot.get_current_pose()
            if curr_pose:
                print(f"📍 Current pose: {curr_pose}")
                
                # Example action: small movement
                print("🔄 Executing example action...")
                success = bot.execute_action(0.05, 0, 0, 0, 0, 0, 0)
                if success:
                    print("✅ Action completed successfully")
                else:
                    print("❌ Action failed")
                
                # Return to original position
                print("🔄 Returning to original position...")
                success = bot.execute_action(-0.05, 0, 0, 0, 0, 0, 0)
                if success:
                    print("✅ Return movement completed")
                else:
                    print("❌ Return movement failed")
            else:
                print("❌ Could not get current pose")
        else:
            print("⚠️  Skipping robot actions (Kinova not available)")
        
        print("\n🎉 BYOVLA with Kinova completed successfully!")
        
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
        print("\n🧹 Cleanup completed")

if __name__ == "__main__":
    main()
