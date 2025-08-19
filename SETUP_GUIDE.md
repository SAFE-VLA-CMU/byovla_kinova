# BYOVLA with Kinova - Dual Environment Setup Guide

## 🎯 **Architecture Overview**

This setup uses **two separate conda environments** to solve the protobuf version conflict:

### **Environment 1: BYOVLA_KINOVA** 
- **Purpose**: Robot control, image capture, basic processing
- **Python**: 3.10
- **Protobuf**: 3.5.1 (Kinova API requirement)
- **Features**: Kinova control, camera, basic image processing

### **Environment 2: BYOVLA_ML**
- **Purpose**: Heavy ML models and inference
- **Python**: 3.10  
- **Protobuf**: 3.20.3+ (TensorFlow requirement)
- **Features**: SAM2, Grounded-SAM-2, Octo, Inpaint Anything

## 🚀 **Quick Start**

### **1. Activate Main Environment (Robot Control)**
```bash
conda activate BYOVLA_KINOVA
python3 byovla_kinova.py
```

### **2. Activate ML Environment (Model Inference)**
```bash
conda activate BYOVLA_ML
# Run ML-specific scripts here
```

## 🔧 **Environment Setup**

### **Main Environment (BYOVLA_KINOVA)**
```bash
# Create environment
conda create -n BYOVLA_KINOVA python=3.10 -y

# Activate
conda activate BYOVLA_KINOVA

# Install dependencies
pip install -r requirements_BYOVLA_KINOVA.txt

# Install Kinova API
pip install kortex_api-2.6.0.post3-py3-none-any.whl
```

### **ML Environment (BYOVLA_ML)**
```bash
# Create environment
conda create -n BYOVLA_ML python=3.10 -y

# Activate
conda activate BYOVLA_ML

# Install dependencies
pip install -r requirements_BYOVLA_ML.txt

# Clone and install ML repositories
git clone https://github.com/facebookresearch/segment-anything-2.git sam2
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
git clone https://github.com/geekyutao/Inpaint-Anything.git

# Install each repository
cd sam2 && pip install -e . && cd ..
cd Grounded-SAM-2 && pip install -e . && cd ..
cd Inpaint-Anything && pip install -e . && cd ..
```

## 📁 **File Structure**

```
gen3/
├── byovla_kinova.py          # Main script (uses BYOVLA_KINOVA env)
├── requirements_BYOVLA_KINOVA.txt
├── requirements_BYOVLA_ML.txt
├── SETUP_GUIDE.md
├── environment_BYOVLA_KINOVA.yml
└── environment_BYOVLA_ML.yml
```

## 🔄 **Workflow**

### **Option A: Separate Execution (Recommended)**
1. **Robot Control**: Use `BYOVLA_KINOVA` environment
2. **ML Processing**: Use `BYOVLA_ML` environment
3. **Data Exchange**: Save/load files between environments

### **Option B: Integrated Execution**
1. **Main Process**: Runs in `BYOVLA_KINOVA` environment
2. **ML Subprocess**: Spawns `BYOVLA_ML` environment for heavy tasks
3. **Communication**: Via files or inter-process communication

## 🧪 **Testing**

### **Test Main Environment**
```bash
conda activate BYOVLA_KINOVA
python3 test_dependencies.py
```

### **Test ML Environment**
```bash
conda activate BYOVLA_ML
python3 -c "import torch; import jax; import transformers; print('ML imports successful')"
```

## ⚠️ **Important Notes**

1. **Never mix environments** - Each has specific protobuf versions
2. **Kinova API** requires protobuf==3.5.1
3. **TensorFlow/SAM2** requires protobuf>=3.20.3
4. **Python 3.10** is the sweet spot for compatibility

## 🚨 **Troubleshooting**

### **Protobuf Conflicts**
- Ensure each environment has the correct protobuf version
- Don't install TensorFlow in BYOVLA_KINOVA environment
- Don't install Kinova API in BYOVLA_ML environment

### **Import Errors**
- Check which environment is active: `conda info --envs`
- Verify package installation: `pip list | grep package_name`
- Ensure correct Python version: `python --version`

## 🎉 **Success Indicators**

- ✅ `BYOVLA_KINOVA` environment: Kinova API + basic packages work
- ✅ `BYOVLA_ML` environment: SAM2 + TensorFlow + JAX work
- ✅ No protobuf version conflicts
- ✅ Both environments can be activated independently

## 🔮 **Future Enhancements**

1. **Automated Environment Switching**: Script to handle environment transitions
2. **Docker Containers**: Isolated environments for production
3. **Model Serving**: REST API for ML model inference
4. **Real-time Integration**: Direct communication between environments 