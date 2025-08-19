# BYOVLA with Kinova Robot

This is a modified version of the BYOVLA (Build Your Own Vision-Language-Action) script that replaces WidowX robot actions with Kinova robot actions and includes image display functionality for VLA outputs.

## Overview

The original BYOVLA script was designed to work with WidowX robots and included:
- Vision-Language-Action (VLA) processing using Octo models
- Object detection and segmentation with Grounded SAM2
- Image inpainting with LAMA
- Robot action execution

This modified version:
- ✅ Replaces WidowX robot control with Kinova robot control
- ✅ Adds image display functionality for VLA outputs
- ✅ Shows inpainting results and masks
- ✅ Maintains the same VLA processing pipeline
- ✅ Includes simplified versions for demonstration

## Key Changes

### 1. Robot Control
- **Before**: Used `InterbotixManipulatorXS` for WidowX
- **After**: Uses `KinovaController` class with Kinova API

### 2. Action Execution
- **Before**: `bot.act(dx, dy, dz, droll, dpitch, dyaw, dgrasp)`
- **After**: `bot.execute_action(dx, dy, dz, droll, dpitch, dyaw, dgrasp)`

### 3. Image Display
- Added `display_vla_results()` function to show:
  - Original image
  - Objects inpainted
  - Final VLA output
  - Generated masks

### 4. Results Saving
- Added `save_vla_results()` function to save all outputs
- Creates organized directory structure
- Saves images and masks separately

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements_byovla_kinova.txt
```

**⚠️ Important Note**: There's a known dependency conflict between the Kinova API (which requires protobuf==3.5.1) and TensorFlow (which requires protobuf>=3.20.3). The requirements file has been updated to avoid this conflict by removing TensorFlow dependencies.

If you encounter any protobuf-related errors, you can:
1. Use the provided requirements file (recommended)
2. Create separate environments for Kinova and ML components
3. Use PyTorch alternatives for TensorFlow functionality

### 2. Install Kinova API
The Kinova API wheel should already be in your workspace:
```bash
pip install kortex_api-2.6.0.post3-py3-none-any.whl
```

### 3. Optional: Install Full BYOVLA Dependencies
For full functionality, you'll need:
- Grounded-SAM-2 models
- Inpaint_Anything models
- Octo model files
- GPT-4 API access

## Configuration

### Robot Connection
Edit the robot connection parameters in `byovla_kinova.py`:
```python
bot = KinovaController(
    ip="192.168.2.9",        # Your Kinova robot IP
    port=10000,              # Default TCP port
    credentials=("admin", "admin")  # Your credentials
)
```

### Camera Setup
If using RealSense camera:
```python
try:
    import pyrealsense2 as rs
    pipeline = init_camera()
    camera_available = True
except ImportError:
    camera_available = False
```

## Usage

### 1. Test Dependencies First
Before running the main script, test that all dependencies work correctly:
```bash
python3 test_dependencies.py
```

This will verify that:
- All core libraries can be imported
- Kinova API is accessible
- Image processing functions work
- No dependency conflicts exist

### 2. Basic Usage
```bash
python3 byovla_kinova.py
```

### What It Does
1. **Connects to Kinova robot**
2. **Initializes camera** (if available)
3. **Captures image** (or uses placeholder)
4. **Simulates VLA processing**:
   - Object detection
   - Mask generation
   - Inpainting
5. **Displays results** with matplotlib
6. **Saves outputs** to `./vla_results/`
7. **Executes robot actions** (if models available)

### Output Files
The script creates a `vla_results/` directory with:
- `01_original_image.jpg` - Initial captured image
- `02_detected_objects.jpg` - Objects inpainted
- `03_combined_visualization.jpg` - Final VLA output
- `mask_*.png` - Generated masks

## Customization

### Language Instructions
Modify the task description:
```python
language_instruction = "Pick up the red object"  # Your task here
```

### VLA Parameters
Adjust sensitivity and processing parameters:
```python
thresh = 0.002              # Sensitivity threshold
w = np.array([1, 1, 1, 0, 0, 0, 0])  # Action weights
N = 5                       # Number of samples
dilate_size_vla = 10        # Mask dilation
```

### Robot Actions
Modify the action execution:
```python
# Example: move 5cm forward
success = bot.execute_action(0.05, 0, 0, 0, 0, 0, 0)

# Example: move 5cm back
success = bot.execute_action(-0.05, 0, 0, 0, 0, 0, 0)
```

## Troubleshooting

### Connection Issues
- Check robot IP address and network connectivity
- Verify credentials are correct
- Ensure robot is powered on and ready

### Camera Issues
- Install `pyrealsense2` for RealSense support
- Check camera connections
- Script will use placeholder images if camera unavailable

### Model Issues
- Install required ML models for full functionality
- Script includes simplified versions for demonstration
- Check model paths and dependencies

### Dependency Conflicts
- **Protobuf Version Conflict**: Kinova API requires protobuf==3.5.1, which conflicts with TensorFlow
- **Solution**: Use the provided requirements file that avoids this conflict
- **Alternative**: Create separate conda environments for different components
- **Test**: Run `python3 test_dependencies.py` to verify all dependencies work

### Robot Movement Issues
- Verify robot is not in emergency stop
- Check joint limits and workspace boundaries
- Ensure proper initialization sequence

## Safety Notes

⚠️ **Important Safety Considerations**:
- Always test robot movements in simulation first
- Keep emergency stop button accessible
- Monitor robot during execution
- Start with small movements
- Verify workspace is clear

## Future Enhancements

To make this fully functional, you would need to:
1. **Install actual ML models** (Grounded-SAM2, LAMA, Octo)
2. **Configure API keys** for GPT-4
3. **Set up proper model paths**
4. **Implement full VLA pipeline**
5. **Add error handling and recovery**

## License

This modified script maintains the same license as the original BYOVLA project.

## Contributing

Feel free to modify and improve this script for your specific use case. The modular design makes it easy to:
- Add new robot types
- Implement different VLA models
- Customize image processing
- Add new visualization features
