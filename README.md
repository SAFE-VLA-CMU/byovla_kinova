# BYOVLA with Kinova Robot - Complete Implementation

This repository contains a complete implementation of BYOVLA (Build Your Own Vision-Language-Action) integrated with Kinova robots, RealSense cameras, and advanced computer vision models.

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.8-3.10** (Kinova API compatibility)
- **Ubuntu 18.04+** (recommended)
- **Kinova Gen3 robot** with network access
- **RealSense D435 camera** (optional but recommended)
- **NVIDIA GPU** with CUDA support (for ML models)

### 2. Clone and Setup
```bash
git clone https://github.com/SAFE-VLA-CMU/byovla_kinova.git
cd byovla_kinova
```

### 3. Install Dependencies
```bash
# Create conda environment
conda env create -f environment_byovla_kinova.yml
conda activate byovla_kinova

# Or install Python dependencies directly
pip install -r requirements_byovla_kinova.txt

# Install Kinova API
pip install kortex_api-2.6.0.post3-py3-none-any.whl
```

### 4. Test Setup
```bash
# Test all dependencies
python3 test_dependencies.py

# Test RealSense camera
python3 test_realsense.py

# Test Kinova robot movement
python3 test_kinova_movement.py
```

### 5. Run Main Application
```bash
python3 byovla_kinova.py
```

## 📁 Project Structure

```
byovla_kinova/
├── byovla_kinova.py              # Main application
├── test_realsense.py             # RealSense camera testing
├── test_kinova_movement.py       # Kinova robot testing
├── test_dependencies.py          # Dependency verification
├── requirements_byovla_kinova.txt # Python dependencies
├── environment_byovla_kinova.yml # Conda environment
├── install_byovla_kinova.sh     # Installation script
├── README_byovla_kinova.md      # Detailed documentation
├── SETUP_GUIDE.md               # Setup instructions
├── Grounded-SAM-2/              # Object detection models
├── sam2/                        # SAM2 segmentation
├── Inpaint-Anything/            # Image inpainting
├── byovla/                      # Core BYOVLA implementation
├── Kinova_API_Demo_Python/      # Kinova examples
├── kinova_test/                 # Robot testing files
├── real_vla_results/            # VLA processing results
├── realsense_captures/          # Camera captures
└── demo_results/                # Demo outputs
```

## 🔧 Configuration

### Robot Connection
Edit robot parameters in `byovla_kinova.py`:
```python
bot = KinovaController(
    ip="192.168.2.9",        # Your robot IP
    port=10000,              # TCP port
    credentials=("admin", "admin")  # Username/password
)
```

### Camera Setup
RealSense camera will be automatically detected. If unavailable, the system will use placeholder images.

### Model Paths
Ensure ML models are in the correct directories:
- Grounded-SAM-2 models in `Grounded-SAM-2/`
- SAM2 models in `sam2/`
- Inpaint-Anything models in `Inpaint-Anything/`

## 🎯 Features

### ✅ Working Components
- **Kinova Robot Control**: Full 6-DOF movement control
- **RealSense Camera**: Depth and color stream processing
- **Object Detection**: Grounded-SAM-2 integration
- **Image Segmentation**: SAM2 mask generation
- **Image Inpainting**: LAMA-based inpainting
- **VLA Pipeline**: Vision-Language-Action processing
- **Results Visualization**: Matplotlib-based display
- **Data Saving**: Organized output storage

### 🔄 VLA Processing Pipeline
1. **Image Capture**: RealSense or placeholder
2. **Object Detection**: Grounded-SAM-2
3. **Mask Generation**: SAM2 segmentation
4. **Image Inpainting**: LAMA processing
5. **Action Planning**: Robot movement calculation
6. **Execution**: Kinova robot control
7. **Results**: Visualization and storage

## 🧪 Testing

### Test Dependencies
```bash
python3 test_dependencies.py
```
This verifies:
- All core libraries can be imported
- Kinova API is accessible
- Image processing functions work
- No dependency conflicts exist

### Test Camera
```bash
python3 test_realsense.py
```
Features:
- Automatic stream configuration
- Depth and color visualization
- Frame capture and saving
- Performance monitoring

### Test Robot
```bash
python3 test_kinova_movement.py
```
Features:
- Robot connection testing
- Movement validation
- Safety checks
- Position monitoring

## 🚨 Troubleshooting

### Common Issues

#### 1. Kinova Connection
- **Problem**: Cannot connect to robot
- **Solution**: Check IP address, network, and credentials
- **Debug**: Run `test_kinova_movement.py`

#### 2. Camera Issues
- **Problem**: RealSense not detected
- **Solution**: Install `pyrealsense2` and check connections
- **Debug**: Run `test_realsense.py`

#### 3. Dependency Conflicts
- **Problem**: Protobuf version conflicts
- **Solution**: Use provided requirements file
- **Debug**: Run `test_dependencies.py`

#### 4. Model Loading
- **Problem**: ML models not found
- **Solution**: Download required model files
- **Debug**: Check model paths in code

### Error Codes
- **Connection Error**: Check robot IP and network
- **Camera Error**: Verify RealSense installation
- **Import Error**: Check Python environment
- **Model Error**: Verify model file locations

## 📊 Performance

### Expected Results
- **Camera FPS**: 30fps (RealSense D435)
- **Processing Time**: 2-5 seconds per frame
- **Robot Response**: <100ms movement commands
- **Memory Usage**: 4-8GB RAM

### Optimization Tips
- Use GPU acceleration for ML models
- Reduce image resolution for faster processing
- Optimize robot movement parameters
- Monitor system resources

## 🔒 Safety

### Robot Safety
- Always test in simulation first
- Keep emergency stop accessible
- Monitor robot during execution
- Start with small movements
- Verify workspace is clear

### System Safety
- Regular dependency updates
- Model validation before use
- Error handling and recovery
- Logging and monitoring

## 📚 Documentation

### Additional Resources
- `README_byovla_kinova.md`: Detailed project documentation
- `SETUP_GUIDE.md`: Step-by-step setup instructions
- Code comments: Inline documentation
- Example outputs: In `demo_results/`

### API Reference
- Kinova API: `Kinova_API_Demo_Python/`
- Grounded-SAM-2: `Grounded-SAM-2/`
- SAM2: `sam2/`
- Inpaint-Anything: `Inpaint-Anything/`

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Test thoroughly
4. Submit pull request

### Code Standards
- Python 3.8+ compatibility
- PEP 8 style guide
- Comprehensive error handling
- Clear documentation

## 📄 License

This project maintains the same license as the original BYOVLA project.

## 🆘 Support

### Getting Help
1. Check troubleshooting section
2. Review error logs
3. Test individual components
4. Check GitHub issues

### Reporting Issues
- Include error messages
- Specify system configuration
- Provide reproduction steps
- Attach relevant logs

---

**Ready to get started?** Follow the Quick Start guide above and run `python3 test_dependencies.py` to verify your setup! 