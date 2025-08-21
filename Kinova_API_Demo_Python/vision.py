import sys
import time
# For image processing
import numpy as np
import cv2
from connect import RobotConnect
from scipy.spatial.transform import Rotation
from kortex_api.autogen.messages import VisionConfig_pb2

class BallDetector:
    """
    This class uses the arm's camera to find a green ball and then calculate its center coordinates in the global frame
    """
    def __init__(self, robot_connection: RobotConnect):
        """
        :param robot_connection: object that has established a connection to the arm
        """
        # Save the current attached api
        self.robot_connection = robot_connection

        # You don't need a connection to access the stream itself, just the ip
        self.camera_stream = f"rtsp://{self.robot_connection.ip}/color"
        # Video capture with opencv so you can process the images
        self.video_capture = cv2.VideoCapture(self.camera_stream)

        # Image processing parameters
        # Blur
        self.kernel_size = (11, 11)
        self.sigma = 0
        # Green color threshold
        # Very ad hoc values... might need to be changed depending on lighting
        self.greenLower = (36, 25, 25)
        self.greenUpper = (70, 255, 255)
        # Erosion/dilation param
        self.erosion_iterations = 20
        self.dilation_iterations = 20

        # Ball geometry parameters
        self.ball_radius = 0.03
        self.ball_height = 0.05
        self.kinova_support = 0.02  # height of black base under the arm
        
        # Coordinate mapping calibration factors
        self.x_scale_factor = 0.15  # meters per normalized unit for X-axis
        self.y_scale_factor = 0.15  # meters per normalized unit for Y-axis
        self.x_offset = 0.0         # X-axis offset correction
        self.y_offset = 0.0         # Y-axis offset correction

        # Store the position of the ball in the global frame
        self.global_point = None
        self.detection_confidence = 0  # Track detection confidence
        self.last_detection_time = 0   # Track when ball was last detected

    def get_intrinsic_vision_parameters(self):
        """
        Using the established connection, obtain the intrinsic parameters of the camera device
        This is needed for transforming the 2D info about the image into a 3D point
        :return: intrinsic parameters of the camera on the arm
        """
        sensor_id = VisionConfig_pb2.SensorIdentifier() # Message for communication
        sensor_id.sensor = VisionConfig_pb2.SENSOR_COLOR # Specify that you want the vision (and not depth)
        intrinsics = self.robot_connection.vision_config.GetIntrinsicParameters(sensor_id, self.robot_connection.vision_device_id)
        return intrinsics

    def find_object(self, display_view: bool = True):
        """
        This function will look for a green ball in the camera video stream
        Saves position of the ball center w.r.t global frame as an object attribute

        :param display_view: Whether opencv should show a screen with the tracked ball
        """
        try:
            # Get the current frame from the video (there might be a tiny delay)
            _, image = self.video_capture.read()

            # Blur and convert to HSV
            blurred = cv2.GaussianBlur(image, self.kernel_size, self.sigma)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # Create a mask of color green
            mask = cv2.inRange(hsv, self.greenLower, self.greenUpper)
            mask = cv2.erode(mask, None, self.erosion_iterations)
            mask = cv2.dilate(mask, None, self.dilation_iterations)

            # Find the right contours
            cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # only proceed if at least one contour was found
            if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                # only proceed if the radius meets a minimum size
                if radius > 10:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(image, (int(x), int(y)), int(radius),
                               (0, 255, 255), 2)
                    cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
                    # If the user wants the view displayed
                    if display_view:
                        cv2.imshow("Tracked ball", image)
                        cv2.waitKey(1)

                    # Simplified coordinate calculation - use relative positioning from current robot pose
                    # Get current robot tool position
                    base_to_tool = self.robot_connection.base.GetMeasuredCartesianPose()
                    current_x = base_to_tool.x
                    current_y = base_to_tool.y
                    current_z = base_to_tool.z
                    
                    # Calculate relative ball position based on image coordinates
                    # Convert pixel coordinates to relative world coordinates
                    image_width = image.shape[1]
                    image_height = image.shape[0]
                    
                    # Normalize pixel coordinates to [-1, 1] range
                    normalized_x = (x - image_width/2) / (image_width/2)
                    normalized_y = (y - image_height/2) / (image_height/2)
                    
                    # Convert to world coordinates (assuming ball is on the table at z = 0.02)
                    # Use a simple mapping: normalized coordinates to world coordinates
                    table_z = 0.02  # Ball is on the table
                    
                    # Map normalized coordinates to world coordinates using calibration factors
                    world_x = current_x + normalized_x * self.x_scale_factor + self.x_offset
                    world_y = current_y + normalized_y * self.y_scale_factor + self.y_offset
                    world_z = table_z
                    
                    # Store the calculated position
                    self.global_point = np.array([world_x, world_y, world_z])
                    
                    # Debug output
                    print(f"Image coords: ({x}, {y}), Normalized: ({normalized_x:.3f}, {normalized_y:.3f})")
                    print(f"Current robot pos: ({current_x:.3f}, {current_y:.3f}, {current_z:.3f})")
                    print(f"Calculated ball pos: ({world_x:.3f}, {world_y:.3f}, {world_z:.3f})")
                    print(f"Scale factors: X={self.x_scale_factor:.3f}, Y={self.y_scale_factor:.3f}")
                    print(f"Offsets: X={self.x_offset:.3f}, Y={self.y_offset:.3f}")
        except:
            print('An error has occurred, restarting detection')

    def calibrate_coordinates(self, x_scale=None, y_scale=None, x_offset=None, y_offset=None):
        """
        Calibrate the coordinate mapping factors to improve ball grasping accuracy
        :param x_scale: New X-axis scale factor (meters per normalized unit)
        :param y_scale: New Y-axis scale factor (meters per normalized unit)
        :param x_offset: New X-axis offset correction (meters)
        :param y_offset: New Y-axis offset correction (meters)
        """
        if x_scale is not None:
            self.x_scale_factor = x_scale
        if y_scale is not None:
            self.y_scale_factor = y_scale
        if x_offset is not None:
            self.x_offset = x_offset
        if y_offset is not None:
            self.y_offset = y_offset
        
        print(f"Calibration updated:")
        print(f"  X scale: {self.x_scale_factor:.3f} m/unit, offset: {self.x_offset:.3f} m")
        print(f"  Y scale: {self.y_scale_factor:.3f} m/unit, offset: {self.y_offset:.3f} m")
    
    def test_calibration(self):
        """
        Test the current calibration by showing the coordinate mapping
        Place a green ball at a known position and run this to verify accuracy
        """
        print("\n=== CALIBRATION TEST ===")
        print("Place a green ball at a known position on the table")
        print("Current calibration factors:")
        print(f"  X scale: {self.x_scale_factor:.3f} m/unit, offset: {self.x_offset:.3f} m")
        print(f"  Y scale: {self.y_scale_factor:.3f} m/unit, offset: {self.y_offset:.3f} m")
        print("Run find_object() to test detection accuracy")
        print("=======================\n")

    def is_ball_detection_stable(self, min_confidence=3, min_duration=1.0):
        """
        Check if ball detection is stable enough for automatic catching
        :param min_confidence: Minimum number of consecutive detections
        :param min_duration: Minimum time (seconds) ball should be detected
        :return: True if detection is stable
        """
        current_time = time.time()
        
        # Check if ball is currently detected
        if self.global_point is not None:
            # Increment confidence if ball is detected
            self.detection_confidence += 1
            self.last_detection_time = current_time
        else:
            # Reset confidence if ball is not detected
            self.detection_confidence = 0
        
        # Check if we have stable detection
        if (self.detection_confidence >= min_confidence and 
            (current_time - self.last_detection_time) <= min_duration):
            return True
        
        return False

    def end_vision(self):
        """
        Call this to end vision activities
        :return:
        """
        self.video_capture.release()
        cv2.destroyAllWindows()