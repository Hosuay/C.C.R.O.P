# 🚀 Quick Start Guide

This guide will get you up and running with C.C.R.O.P in under 5 minutes!

## Option 1: Web Demo (Easiest - No Training Required)

Perfect for demonstrations and quick testing.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the demo
python demo/app.py

# 3. Open browser to http://localhost:7860
```

**Note**: The demo will work with default YOLOv8 detection. For disease classification, you'll need a trained classifier checkpoint (see below).

---

## Option 2: Use Pre-trained Models

If you have access to a pre-trained classifier checkpoint:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your checkpoint
mkdir -p checkpoints
# Copy your best_model.pth to checkpoints/

# 3. Test with CLI
python cli/predict.py path/to/image.jpg \
    --classifier checkpoints/best_model.pth \
    --visualize --output result.jpg

# 4. Or launch web demo
python demo/app.py
# Then configure the checkpoint path in the web interface
```

---

## Option 3: Train Your Own Model

### Step 1: Setup Kaggle Credentials

```bash
# Download kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 2: Open Training Notebook

The easiest way to train is using Google Colab:

1. Open `notebooks/CCROP.ipynb` in Google Colab
2. Upload your `kaggle.json` when prompted
3. Run all cells
4. Download the trained model from `checkpoints/best_model.pth`

### Step 3: Use Your Trained Model

```bash
# Test your model
python cli/predict.py test_image.jpg \
    --classifier checkpoints/best_model.pth

# Launch web demo
python demo/app.py
```

---

## Common Use Cases

### 1. Analyze a Single Image

```bash
python cli/predict.py leaf.jpg --classifier checkpoints/best_model.pth
```

### 2. Batch Process Multiple Images

```bash
python cli/predict.py images_folder/ \
    --batch \
    --classifier checkpoints/best_model.pth \
    --output results.csv
```

### 3. Get Visual Results

```bash
python cli/predict.py leaf.jpg \
    --classifier checkpoints/best_model.pth \
    --visualize \
    --output annotated_result.jpg
```

### 4. Use Different YOLO Models

```bash
# Faster (less accurate)
python cli/predict.py leaf.jpg --detector yolov8n.pt

# More accurate (slower)
python cli/predict.py leaf.jpg --detector yolov8x.pt
```

### 5. Skip Detection (Direct Classification)

```bash
python cli/predict.py leaf.jpg \
    --classifier checkpoints/best_model.pth \
    --no-detect
```

---

## Python API Quick Example

```python
import torch
from src.ccrop.models.pipeline import CCROPPipeline
from src.ccrop.models.classifier import CCROPClassifier
from src.ccrop.models.detector import LeafDetector
from src.ccrop.inference.predictor import StressPredictor

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models
detector = LeafDetector(model_name="yolov8n.pt", device=device)
classifier, metadata = CCROPClassifier.from_checkpoint(
    "checkpoints/best_model.pth", device=device
)

# Create pipeline
pipeline = CCROPPipeline(
    detector=detector,
    classifier=classifier,
    class_names=metadata['classes'],
    stress_mapping=metadata['stress_mapping'],
    device=device
)

# Predict
predictor = StressPredictor(pipeline)
results = predictor.predict_from_path("leaf.jpg")

# Print results
print(f"Overall Stress: {results['overall_stress']:.2f}%")
print(f"Number of Leaves: {results['num_leaves']}")

# Get health status
health = predictor.get_health_status(results['overall_stress'])
print(f"Status: {health['status']}")
print(f"Recommendation: {health['recommendation']}")
```

---

## Troubleshooting

### Issue: "No module named 'ultralytics'"

```bash
pip install ultralytics
```

### Issue: "Kaggle authentication failed"

```bash
# Make sure kaggle.json is in the right place
ls ~/.kaggle/kaggle.json

# Check permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Issue: "CUDA out of memory"

```bash
# Use CPU instead
python demo/app.py  # Will auto-detect and use CPU if CUDA unavailable

# Or explicitly:
python cli/predict.py image.jpg --device cpu
```

### Issue: "No checkpoint found"

You need to either:
1. Train a model using the notebook
2. Download a pre-trained checkpoint
3. Use the demo without classification (detection only)

---

## Next Steps

- 📖 Read the full [README.md](README.md) for detailed documentation
- 🎓 Check out the training [notebook](notebooks/CCROP.ipynb)
- 💻 Explore the [Python API](README.md#python-api)
- 🌐 Try the [web demo](demo/app.py)

---

## Need Help?

- 🐛 Report bugs: [GitHub Issues](https://github.com/Hosuay/C.C.R.O.P/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/Hosuay/C.C.R.O.P/discussions)

---

**Happy detecting! 🌿**
