#!/usr/bin/env python3
"""
Simple Kinova Arm Movement Test Script
Tests basic connection and movement functionality
"""

import time
import sys
import os

# Add the parent directory to path to access the Kinova API wheel
sys.path.append('..')

# Kinova API imports
from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import Base_pb2, DeviceConfig_pb2, Session_pb2
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient


class KinovaTestController:
    """Simple test controller for Kinova arm"""
    
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
        
        print(f"🤖 Kinova Test Controller")
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
            self.session_manager.CreateSession(session_info)
            print("✅ Session created!")
            
            # Create base client
            self.base = BaseClient(self.router)
            
            # Get current pose
            self.get_current_pose()
            print("✅ Connected successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def get_current_pose(self):
        """Get current end-effector pose"""
        try:
            pose = self.base.GetMeasuredCartesianPose()
            print(f"📍 Current pose:")
            print(f"   x: {pose.x:.3f} m")
            print(f"   y: {pose.y:.3f} m") 
            print(f"   z: {pose.z:.3f} m")
            print(f"   theta_x: {pose.theta_x:.1f}°")
            print(f"   theta_y: {pose.theta_y:.1f}°")
            print(f"   theta_z: {pose.theta_z:.1f}°")
            return pose
        except Exception as e:
            print(f"❌ Failed to get pose: {e}")
            return None
    
    def move_to_position(self, x, y, z, theta_x=180, theta_y=0, theta_z=90):
        """Move to specific position"""
        try:
            print(f"\n🔄 Moving to position: ({x:.3f}, {y:.3f}, {z:.3f})")
            print(f"   Orientation: theta_x={theta_x}, theta_y={theta_y}, theta_z={theta_z}")
            
            # Create action
            action = Base_pb2.Action()
            action.name = f"Move to ({x:.3f}, {y:.3f}, {z:.3f})"
            action.application_data = ""
            
            cartesian_pose = action.reach_pose.target_pose
            cartesian_pose.x = x
            cartesian_pose.y = y
            cartesian_pose.z = z
            cartesian_pose.theta_x = theta_x
            cartesian_pose.theta_y = theta_y
            cartesian_pose.theta_z = theta_z
            
            # Execute movement
            success = self._execute_movement(action)
            
            if success:
                print("✅ Movement completed!")
                self.get_current_pose()
            else:
                print("❌ Movement failed!")
            
            return success
            
        except Exception as e:
            print(f"❌ Movement failed: {e}")
            return False
    
    def _execute_movement(self, action):
        """Execute movement action with timeout"""
        import threading
        
        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self._check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )
        
        try:
            print("   Executing movement...")
            self.base.ExecuteAction(action)
            
            print("   Waiting for completion...")
            finished = e.wait(self.TIMEOUT_DURATION)
            self.base.Unsubscribe(notification_handle)
            
            return finished
                
        except Exception as e:
            print(f"   Movement execution failed: {e}")
            return False
    
    def _check_for_end_or_abort(self, e):
        """Check for action completion"""
        def check(notification, e=e):
            event_name = Base_pb2.ActionEvent.Name(notification.action_event)
            print(f"   📡 Event: {event_name}")
            if (notification.action_event == Base_pb2.ACTION_END or 
                notification.action_event == Base_pb2.ACTION_ABORT):
                e.set()
        return check
    
    def test_simple_movements(self):
        """Test simple movements"""
        print("\n🧪 Testing Simple Movements")
        print("=" * 30)
        
        # Get current pose
        current_pose = self.get_current_pose()
        if not current_pose:
            print("❌ Cannot get current pose - aborting test")
            return False
        
        # Test 1: Small forward movement
        print("\n📋 Test 1: Small forward movement")
        test_x = current_pose.x + 0.05  # 5cm forward
        test_y = current_pose.y
        test_z = current_pose.z
        
        if not self.move_to_position(test_x, test_y, test_z):
            print("❌ Test 1 failed")
            return False
        
        time.sleep(2)  # Wait between movements
        
        # Test 2: Return to original position
        print("\n📋 Test 2: Return to original position")
        if not self.move_to_position(current_pose.x, current_pose.y, current_pose.z):
            print("❌ Test 2 failed")
            return False
        
        time.sleep(2)
        
        # Test 3: Small upward movement
        print("\n📋 Test 3: Small upward movement")
        test_x = current_pose.x
        test_y = current_pose.y
        test_z = current_pose.z + 0.05  # 5cm up
        
        if not self.move_to_position(test_x, test_y, test_z):
            print("❌ Test 3 failed")
            return False
        
        time.sleep(2)
        
        # Test 4: Return to original position
        print("\n📋 Test 4: Return to original position")
        if not self.move_to_position(current_pose.x, current_pose.y, current_pose.z):
            print("❌ Test 4 failed")
            return False
        
        print("\n✅ All movement tests completed successfully!")
        
        # Test rotations
        print("\n🧪 Testing Rotations")
        print("=" * 30)
        
        # Test 5: Yaw rotation (left/right)
        print("\n📋 Test 5: Yaw rotation (left)")
        test_theta_z = current_pose.theta_z + 15  # 15 degrees left
        if not self.move_to_position(current_pose.x, current_pose.y, current_pose.z, 
                                   current_pose.theta_x, current_pose.theta_y, test_theta_z):
            print("❌ Test 5 failed")
            return False
        
        time.sleep(2)
        
        # Test 6: Return to original orientation
        print("\n📋 Test 6: Return to original orientation")
        if not self.move_to_position(current_pose.x, current_pose.y, current_pose.z,
                                   current_pose.theta_x, current_pose.theta_y, current_pose.theta_z):
            print("❌ Test 6 failed")
            return False
        
        time.sleep(2)
        
        # Test 7: Pitch rotation (up/down)
        print("\n📋 Test 7: Pitch rotation (up)")
        test_theta_y = current_pose.theta_y + 10  # 10 degrees up
        if not self.move_to_position(current_pose.x, current_pose.y, current_pose.z,
                                   current_pose.theta_x, test_theta_y, current_pose.theta_z):
            print("❌ Test 7 failed")
            return False
        
        time.sleep(2)
        
        # Test 8: Return to original orientation
        print("\n📋 Test 8: Return to original orientation")
        if not self.move_to_position(current_pose.x, current_pose.y, current_pose.z,
                                   current_pose.theta_x, current_pose.theta_y, current_pose.theta_z):
            print("❌ Test 8 failed")
            return False
        
        time.sleep(2)
        
        # Test 9: Roll rotation (sideways)
        print("\n📋 Test 9: Roll rotation (sideways)")
        test_theta_x = current_pose.theta_x + 10  # 10 degrees roll
        if not self.move_to_position(current_pose.x, current_pose.y, current_pose.z,
                                   test_theta_x, current_pose.theta_y, current_pose.theta_z):
            print("❌ Test 9 failed")
            return False
        
        time.sleep(2)
        
        # Test 10: Final return to original position and orientation
        print("\n📋 Test 10: Return to original position and orientation")
        if not self.move_to_position(current_pose.x, current_pose.y, current_pose.z,
                                   current_pose.theta_x, current_pose.theta_y, current_pose.theta_z):
            print("❌ Test 10 failed")
            return False
        
        print("\n🎉 All movement AND rotation tests completed successfully!")
        return True
    
    def test_rotations_only(self):
        """Test only rotation movements"""
        print("\n🧪 Testing Rotations Only")
        print("=" * 30)
        
        # Get current pose
        current_pose = self.get_current_pose()
        if not current_pose:
            print("❌ Cannot get current pose - aborting test")
            return False
        
        # Test Yaw rotation (left/right)
        print("\n📋 Test 1: Yaw rotation (left)")
        test_theta_z = current_pose.theta_z + 15  # 15 degrees left
        if not self.move_to_position(current_pose.x, current_pose.y, current_pose.z, 
                                   current_pose.theta_x, current_pose.theta_y, test_theta_z):
            print("❌ Yaw rotation failed")
            return False
        
        time.sleep(2)
        
        # Test Yaw rotation (right)
        print("\n📋 Test 2: Yaw rotation (right)")
        test_theta_z = current_pose.theta_z - 15  # 15 degrees right
        if not self.move_to_position(current_pose.x, current_pose.y, current_pose.z,
                                   current_pose.theta_x, current_pose.theta_y, test_theta_z):
            print("❌ Yaw rotation failed")
            return False
        
        time.sleep(2)
        
        # Return to original orientation
        print("\n📋 Test 3: Return to original orientation")
        if not self.move_to_position(current_pose.x, current_pose.y, current_pose.z,
                                   current_pose.theta_x, current_pose.theta_y, current_pose.theta_z):
            print("❌ Return to original failed")
            return False
        
        print("\n✅ All rotation tests completed successfully!")
        return True
    
    def disconnect(self):
        """Disconnect from robot"""
        try:
            if self.session_manager:
                self.session_manager.CloseSession()
            if self.transport:
                self.transport.disconnect()
            print("🔌 Disconnected from robot")
        except Exception as e:
            print(f"❌ Error disconnecting: {e}")


def main():
    """Main test function"""
    print("🚀 Kinova Arm Movement Test")
    print("=" * 40)
    
    # Create controller
    controller = KinovaTestController()
    
    try:
        # Connect to robot
        if not controller.connect():
            print("❌ Failed to connect - exiting")
            return
        
        # Wait a moment
        time.sleep(2)
        
        # Ask user what to test
        print("\n🧪 What would you like to test?")
        print("1. Movement tests only (4 tests)")
        print("2. Movement + Rotation tests (10 tests)")
        print("3. Rotation tests only (3 tests)")
        
        try:
            choice = input("Enter your choice (1-3): ").strip()
        except:
            choice = "2"  # Default to full test
        
        # Run selected tests
        if choice == "1":
            success = controller.test_simple_movements()
        elif choice == "3":
            success = controller.test_rotations_only()
        else:  # Default to full test
            success = controller.test_simple_movements()
        
        if success:
            print("\n🎉 All tests passed! Robot is working correctly.")
        else:
            print("\n❌ Some tests failed.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        controller.disconnect()
        print("\n🧹 Test completed")


if __name__ == "__main__":
    main() 