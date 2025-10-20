# 🌿 C.C.R.O.P - Cannabis Crop Research in Optimization of Plant-Health

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**AI-Powered Cannabis Leaf Disease Detection System** using state-of-the-art deep learning models including YOLOv8 for detection and ResNet/EfficientNet for classification.

<p align="center">
  <img src="https://img.shields.io/badge/Status-Demo%20Ready-success" alt="Status">
  <img src="https://img.shields.io/badge/Version-2.0.0-blue" alt="Version">
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Web Demo](#web-demo)
  - [CLI Tools](#cli-tools)
  - [Python API](#python-api)
- [Model Training](#model-training)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

C.C.R.O.P is an advanced AI system designed to automatically detect and classify cannabis leaf diseases and stress conditions. The system combines:

- **YOLOv8**: For detecting and localizing individual leaves in images
- **Deep Learning Classification**: ResNet/EfficientNet models for disease classification
- **Stress Assessment**: Quantitative stress scoring (0-100 scale)
- **Interactive Demo**: Web-based interface for easy demonstration

### Key Applications

- 🔬 **Research**: Academic studies on plant pathology and AI in agriculture
- 🌱 **Cultivation**: Real-time monitoring of crop health
- 📊 **Data Analysis**: Large-scale plant health assessment
- 🎓 **Education**: Teaching tool for plant disease recognition

---

## ✨ Features

### Core Capabilities

- ✅ **Automatic Leaf Detection** - Uses YOLOv8 to locate individual leaves in complex images
- ✅ **Multi-Class Disease Classification** - Identifies various disease types and severities
- ✅ **Stress Quantification** - Provides 0-100% stress scores with health status
- ✅ **Batch Processing** - Analyze hundreds of images efficiently
- ✅ **Real-time Analysis** - Fast inference suitable for live monitoring
- ✅ **Visual Overlays** - Clear bounding boxes and classification labels

### Technical Features

- 🔧 **Modular Architecture** - Easy to extend and customize
- 🚀 **GPU Acceleration** - CUDA support for fast inference
- 📦 **Multiple Export Formats** - PyTorch, TorchScript, ONNX
- 🎨 **Rich Visualizations** - Confusion matrices, training curves, detection overlays
- 🌐 **Web Interface** - Gradio-based demo for easy sharing

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              Input Image                        │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│         YOLOv8 Leaf Detection                   │
│  Detects and localizes individual leaves        │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│         Leaf Cropping & Preprocessing           │
│  Extracts detected regions for classification   │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│    ResNet/EfficientNet Classification           │
│  Classifies each leaf for disease/stress        │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│      Results Aggregation & Scoring              │
│  Overall health status and recommendations      │
└─────────────────────────────────────────────────┘
```

### Pipeline Modes

1. **Detection + Classification** (Recommended)
   - First detects individual leaves using YOLO
   - Then classifies each detected leaf
   - Best for images with multiple leaves or complex backgrounds

2. **Direct Classification**
   - Classifies the entire image directly
   - Faster but less accurate for multi-leaf images
   - Good for close-up single leaf photos

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster inference)

### Option 1: Install from source (Recommended)

```bash
# Clone the repository
git clone https://github.com/Hosuay/C.C.R.O.P.git
cd C.C.R.O.P

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Option 2: Quick Install

```bash
pip install -r requirements.txt
```

### Optional: Kaggle Dataset Download

If you want to download datasets from Kaggle:

```bash
pip install kaggle

# Setup Kaggle API credentials
# Download kaggle.json from https://www.kaggle.com/settings
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## 🚀 Quick Start

### 1. Run the Web Demo (Easiest)

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the demo
python demo/app.py
```

Then open your browser to `http://localhost:7860`

**Note**: On first run, YOLOv8 weights will be downloaded automatically.

### 2. Command Line Prediction

```bash
# Single image (requires trained classifier)
python cli/predict.py path/to/image.jpg \
    --classifier checkpoints/best_model.pth \
    --visualize --output results.jpg

# Batch processing
python cli/predict.py path/to/images/ \
    --classifier checkpoints/best_model.pth \
    --batch --output results.csv
```

### 3. Python API

```python
import torch
from src.ccrop.models.pipeline import create_pipeline
from src.ccrop.models.classifier import CCROPClassifier
from src.ccrop.models.detector import LeafDetector
from src.ccrop.inference.predictor import StressPredictor

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models
detector = LeafDetector(model_name="yolov8n.pt", device=device)
classifier, metadata = CCROPClassifier.from_checkpoint(
    "checkpoints/best_model.pth", device=device
)

# Create pipeline
from src.ccrop.models.pipeline import CCROPPipeline
pipeline = CCROPPipeline(
    detector=detector,
    classifier=classifier,
    class_names=metadata['classes'],
    stress_mapping=metadata['stress_mapping'],
    device=device
)

# Create predictor
predictor = StressPredictor(pipeline)

# Analyze image
results = predictor.predict_from_path("path/to/leaf.jpg")
print(f"Stress Score: {results['overall_stress']:.2f}%")
print(f"Leaves Detected: {results['num_leaves']}")
```

---

## 💻 Usage

### Web Demo

The Gradio web interface provides the easiest way to demonstrate the system:

```bash
python demo/app.py
```

**Features:**
- ✅ Single image analysis with visual results
- ✅ Batch processing for multiple images
- ✅ Side-by-side comparison of detection vs direct classification
- ✅ Interactive model configuration
- ✅ Real-time stress gauge visualization
- ✅ Detailed results with recommendations

### CLI Tools

#### Prediction

```bash
# Basic prediction with visualization
python cli/predict.py image.jpg \
    --classifier checkpoints/best_model.pth \
    --visualize --output result.jpg

# Batch processing
python cli/predict.py images/ --batch \
    --classifier checkpoints/best_model.pth \
    --output results.csv

# Custom YOLO model with higher accuracy
python cli/predict.py image.jpg \
    --classifier checkpoints/best_model.pth \
    --detector yolov8m.pt \
    --conf-threshold 0.3

# Skip detection (direct classification only)
python cli/predict.py image.jpg \
    --classifier checkpoints/best_model.pth \
    --no-detect
```

### Python API

#### Basic Usage

```python
from src.ccrop import CCROPPipeline, StressPredictor
from src.ccrop.models.classifier import CCROPClassifier
from src.ccrop.models.detector import LeafDetector

# Initialize
device = 'cuda'
detector = LeafDetector(device=device)
classifier, metadata = CCROPClassifier.from_checkpoint("model.pth", device=device)

pipeline = CCROPPipeline(detector, classifier, metadata['classes'],
                        metadata['stress_mapping'], device=device)
predictor = StressPredictor(pipeline)

# Predict
results = predictor.predict_from_path("leaf.jpg")
```

#### Batch Processing

```python
from src.ccrop.inference.predictor import BatchPredictor

batch_predictor = BatchPredictor(pipeline)
df = batch_predictor.predict_directory("images/", output_csv="results.csv")
```

#### Visualization

```python
from src.ccrop.utils.visualization import plot_detections, visualize_predictions

# Plot detections on image
plot_detections(image, results['detections'], save_path="output.jpg")

# Visualize multiple predictions in grid
visualize_predictions(images, predictions, save_path="grid.jpg")
```

---

## 🎓 Model Training

### 1. Prepare Dataset

The original Jupyter notebook (`CCROP.ipynb`) contains the complete training pipeline. You can use it in Google Colab:

```python
from src.ccrop.data import DatasetManager

dm = DatasetManager(root_dir="./dataset")

# Download from Kaggle (optional)
dm.setup_kaggle_credentials(kaggle_json_path="kaggle.json")
dm.download_from_kaggle("engineeringubu/leaf-manifestation-diseases-of-cannabis")
dm.extract_dataset()

# Find dataset
dataset_path = dm.find_dataset_path()
classes, stress_mapping = dm.get_class_info(dataset_path)
```

### 2. Train Classifier

```python
from src.ccrop.data import get_dataloaders
from src.ccrop.models.classifier import create_classifier
import torch.nn as nn
import torch.optim as optim

# Get data loaders
train_loader, val_loader, test_loader, classes = get_dataloaders(
    dataset_path, batch_size=16, num_workers=2
)

# Create model
model = create_classifier(arch="resnet18", num_classes=len(classes))
model = model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=0.5, patience=3)

# Training loop (see CCROP.ipynb for complete implementation)
```

### 3. Train Custom YOLO Detector (Advanced)

```python
from src.ccrop.models.detector import LeafDetector

detector = LeafDetector()
detector.train_custom_model(
    data_yaml="leaf_detection_data.yaml",
    epochs=100,
    imgsz=640,
    batch=16
)
```

---

## 📁 Project Structure

```
C.C.R.O.P/
├── src/ccrop/                 # Main package
│   ├── models/                # Model architectures
│   │   ├── classifier.py      # ResNet/EfficientNet classifier
│   │   ├── detector.py        # YOLO detector
│   │   └── pipeline.py        # Integrated pipeline
│   ├── data/                  # Dataset handling
│   │   ├── dataset.py         # Dataset management
│   │   └── transforms.py      # Image transformations
│   ├── inference/             # Inference tools
│   │   └── predictor.py       # Prediction classes
│   └── utils/                 # Utilities
│       ├── config.py          # Configuration management
│       └── visualization.py   # Plotting functions
├── demo/                      # Web demo
│   └── app.py                 # Gradio interface
├── cli/                       # Command-line tools
│   └── predict.py             # Prediction CLI
├── configs/                   # Configuration files
├── notebooks/                 # Jupyter notebooks
│   └── CCROP.ipynb           # Original training notebook
├── tests/                     # Unit tests
├── examples/                  # Example images
├── checkpoints/               # Saved models (created during training)
├── dataset/                   # Dataset directory (auto-created)
├── results/                   # Output directory (auto-created)
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
├── .gitignore                # Git ignore file
└── README.md                  # This file
```

---

## 📊 Results

### Model Performance (Example)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| ResNet18 | 94.5% | 0.943 | 0.945 | 0.944 |
| ResNet50 | 95.8% | 0.956 | 0.958 | 0.957 |
| EfficientNet-B0 | 96.2% | 0.961 | 0.962 | 0.961 |

*Note: Actual performance depends on dataset and training configuration*

### Health Status Categories

- 🟢 **Healthy** (0-33%): No visible signs of disease or stress
- 🟠 **Moderate Stress** (33-66%): Early symptoms, monitoring recommended
- 🔴 **Severe Stress** (66-100%): Significant disease, immediate action needed

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/ demo/ cli/

# Lint
flake8 src/
```

---

## 📖 Citation

If you use C.C.R.O.P in your research, please cite:

```bibtex
@software{ccrop2024,
  title={C.C.R.O.P: Cannabis Crop Research in Optimization of Plant-Health},
  author={C.C.R.O.P Research Team},
  year={2024},
  url={https://github.com/Hosuay/C.C.R.O.P},
  version={2.0.0}
}
```

**Dataset Citation:**
```bibtex
@dataset{engineeringubu_leaf_manifestation_diseases_2023,
  title        = {Leaf Manifestation Diseases of Cannabis},
  author       = {EngineeringUBU},
  year         = {2023},
  url          = {https://www.kaggle.com/datasets/engineeringubu/leaf-manifestation-diseases-of-cannabis}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Dataset usage must comply with Kaggle's terms of service.

---

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics for object detection
- **PyTorch** team for the deep learning framework
- **Kaggle** and EngineeringUBU for the cannabis leaf disease dataset
- Cannabis research community for domain expertise
- Open-source contributors

---

## 📞 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/Hosuay/C.C.R.O.P/issues)
- **Pull Requests**: Contributions are welcome!

---

## 🌱 Research Context

This model supports C.C.R.O.P's mission to advance AI research in the cultivation of hemp and cannabis, providing open-source datasets and predictive models to improve crop health and sustainability.

The project focuses on:
- Early stress detection in cannabis plants
- Disease classification using computer vision
- AI integration in agricultural monitoring systems
- Open-source tools for researchers and cultivators

---

## 🚀 What's New in Version 2.0

- ✨ **YOLO Integration**: Added YOLOv8 for automatic leaf detection
- 🎨 **Web Demo**: Interactive Gradio interface for easy demonstrations
- 🔧 **Modular Architecture**: Restructured as a proper Python package
- 📦 **Multiple Formats**: Support for PyTorch, TorchScript, and ONNX
- 🖥️ **CLI Tools**: Command-line interface for batch processing
- 📊 **Enhanced Visualizations**: Better detection overlays and result formatting
- 🐍 **Better API**: Clean Python API for integration into other projects

---

<p align="center">
  <b>Built with ❤️ for the plant science community</b>
</p>

<p align="center">
  <sub>C.C.R.O.P - Advancing plant health through artificial intelligence</sub>
</p>
