"""
C.C.R.O.P - Center for Cannabaceae Research in Optimization of Plant-Health
AI-powered Cannabis Leaf Disease Detection System

This package provides tools for detecting and classifying cannabis leaf diseases
using state-of-the-art deep learning models including YOLO for detection and
ResNet/EfficientNet for classification.
"""

__version__ = "2.0.0"
__author__ = "C.C.R.O.P Research Team"

from .models.classifier import CCROPClassifier
from .models.detector import LeafDetector
from .models.pipeline import CCROPPipeline
from .inference.predictor import StressPredictor
from .utils.config import CCROPConfig

__all__ = [
    'CCROPClassifier',
    'LeafDetector',
    'CCROPPipeline',
    'StressPredictor',
    'CCROPConfig',
]
