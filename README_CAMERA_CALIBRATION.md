# Camera Calibration for BYOVLA with Kinova

This document explains how to use the enhanced camera calibration system that makes BYOVLA work properly with 3D spatial reasoning.

## 🎯 Why Camera Calibration is Essential

The original BYOVLA system was **fundamentally broken** because it lacked:

1. **Camera intrinsics** - focal length, principal point, distortion coefficients
2. **Camera extrinsics** - transformation from camera to robot coordinates  
3. **3D coordinate mapping** - converting 2D image observations to 3D robot actions
4. **Metric measurements** - accurate real-world distances and positions

Without these, the robot was essentially moving randomly relative to objects it was trying to interact with.

## 🚀 What's Been Fixed

### ✅ Camera Intrinsics
- **Real calibration parameters** from your RealSense D435 camera
- **Focal lengths**: fx=605.88, fy=605.09
- **Principal point**: (320.80, 244.40)
- **Resolution**: 640x480
- **Distortion coefficients**: [0.0, 0.0, 0.0, 0.0, 0.0]

### ✅ 3D Coordinate System
- **Pixel coordinates** → **Camera coordinates** → **Robot coordinates**
- **Depth processing** for accurate 3D measurements
- **Coordinate transformations** using proper camera matrices

### ✅ Robot Control
- **3D-aware movements** to specific positions
- **Proper approach trajectories** for grasping
- **Metric precision** in robot actions

## 📋 System Architecture

```
2D Image + Depth → Camera Calibration → 3D World Coordinates → Robot Actions
     ↓                    ↓                      ↓              ↓
Grounded SAM2 → Intrinsics/Extrinsics → Depth Mapping → Octo Policy → Kinova Control
```

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
pip install numpy opencv-python pyyaml pyrealsense2
```

### 2. Run the Test Script
```bash
python3 test_calibration.py
```

This will verify that the camera calibration system is working correctly.

### 3. Calibrate Camera-to-Robot Transformation

**Option A: Interactive Calibration (Recommended)**
```bash
python3 calibrate_camera.py --robot-ip 192.168.2.9 --points 6
```

**Option B: Manual Calibration**
1. Place calibration object at known robot positions
2. Move robot to each position
3. Capture image and click on object
4. Repeat for 6+ points
5. System calculates transformation matrix

### 4. Run Enhanced BYOVLA
```bash
python3 byovla_kinova.py
```

## 🎯 Calibration Process

### Step 1: Camera Setup
- Mount RealSense D435 camera above robot workspace
- Ensure clear view of robot end-effector
- Check depth stream is working

### Step 2: Collect Calibration Points
- **Point 1**: Robot at home position
- **Point 2**: Robot moved 10cm in X direction
- **Point 3**: Robot moved 10cm in Y direction  
- **Point 4**: Robot moved 10cm in Z direction
- **Point 5**: Robot at workspace corner
- **Point 6**: Robot at another workspace corner

### Step 3: Validation
- System calculates transformation matrix
- Validates accuracy (should be <2cm error)
- Saves calibration to `camera_calibration.yaml`

## 🔧 Technical Details

### Camera Parameters
```yaml
intrinsics:
  fx: 605.876220703125  # Focal length X
  fy: 605.091796875     # Focal length Y
  cx: 320.8007507324219 # Principal point X
  cy: 244.4005126953125 # Principal point Y
  width: 640
  height: 480

distortion: [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion
```

### Coordinate Transformations
```python
# Pixel to Camera coordinates
x = (u - cx) * depth / fx
y = (v - cy) * depth / fy
z = depth

# Camera to Robot coordinates  
robot_coords = T_camera_to_robot @ camera_coords
```

### Depth Processing
- **RealSense depth scale**: 0.001 (millimeters to meters)
- **Valid range**: 0.1m to 10.0m
- **Filtering**: Removes invalid depth values
- **Object depth**: Median depth within object mask

## 🧪 Testing the System

### 1. Basic Calibration Test
```bash
python3 test_calibration.py
```

### 2. Coordinate Mapping Test
```python
from byovla_kinova import CameraCalibration, DepthProcessor

# Test pixel to 3D conversion
calibration = CameraCalibration()
depth_processor = DepthProcessor(calibration)

# Convert pixel (320, 240) at depth 0.5m
camera_coords = calibration.pixel_to_camera_coords(320, 240, 0.5)
print(f"Camera coordinates: {camera_coords}")
```

### 3. Robot Movement Test
```python
# Test 3D movement to specific position
target_pos = [0.3, 0.0, 0.5]  # 30cm X, 0cm Y, 50cm Z
success = bot.execute_3d_action(target_pos)
```

## 📊 Expected Results

### Before Calibration
- ❌ Robot moves randomly
- ❌ No 3D spatial awareness
- ❌ Actions not aligned with objects
- ❌ Metric measurements meaningless

### After Calibration
- ✅ Robot moves to precise 3D positions
- ✅ Actions properly aligned with objects
- ✅ Accurate metric measurements
- ✅ Proper grasp poses calculated

## 🚨 Troubleshooting

### Common Issues

1. **"Camera intrinsics not set"**
   - Check that `CameraCalibration` is properly initialized
   - Verify calibration file exists and is readable

2. **"Depth processing failed"**
   - Check RealSense camera connection
   - Verify depth stream is enabled
   - Check depth values are in valid range

3. **"Transform calculation failed"**
   - Ensure at least 3 calibration points
   - Check robot positions are accurate
   - Verify depth measurements are correct

4. **Poor calibration accuracy**
   - Use more calibration points (6-10)
   - Ensure calibration object is clearly visible
   - Check robot positioning accuracy
   - Verify depth measurements

### Debug Mode
```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)

# Check calibration parameters
print(f"Camera matrix: {calibration.get_camera_matrix()}")
print(f"Transform matrix: {calibration.camera_to_robot_transform}")
```

## 🔮 Future Improvements

1. **Automatic calibration** using calibration targets
2. **Multi-camera support** for different viewpoints
3. **Online calibration** during operation
4. **Calibration validation** with known objects
5. **ROS integration** for dynamic calibration

## 📚 References

- [RealSense Camera Calibration](https://github.com/IntelRealSense/librealsense/tree/master/tools/rs-enumerate-devices)
- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Robot Kinematics](https://en.wikipedia.org/wiki/Forward_kinematics)
- [Coordinate Transformations](https://en.wikipedia.org/wiki/Transformation_matrix)

## 🎉 Success!

With proper camera calibration, your BYOVLA system now has:

- **True 3D spatial reasoning**
- **Accurate metric measurements** 
- **Proper robot-object coordination**
- **Professional-grade precision**

The robot will now move exactly where it needs to go to interact with objects detected in the camera view! 