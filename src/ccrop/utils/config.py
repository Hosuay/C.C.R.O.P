"""
Configuration management for CCROP system
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import yaml
import json


@dataclass
class CCROPConfig:
    """Centralized configuration for CCROP pipeline"""

    # Dataset settings
    dataset_main: str = "engineeringubu/leaf-manifestation-diseases-of-cannabis"
    dataset_fallback: str = "vipoooool/new-plant-diseases-dataset"
    root_dir: str = "./dataset"

    # Model settings
    classifier_arch: str = "resnet18"  # Options: resnet18, resnet34, resnet50, efficientnet_b0
    yolo_model: str = "yolov8n.pt"  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    input_size: int = 224
    num_workers: int = 2

    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 15
    early_stopping_patience: int = 5

    # Data split
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # YOLO settings
    yolo_conf_threshold: float = 0.25
    yolo_iou_threshold: float = 0.45
    yolo_max_detections: int = 10

    # Detection + Classification pipeline
    use_detection: bool = True  # If True, use YOLO first, else direct classification
    min_leaf_area: int = 100  # Minimum pixel area for detected leaves

    # Paths
    checkpoint_dir: str = "./checkpoints"
    logs_dir: str = "./logs"
    results_dir: str = "./results"
    yolo_weights_dir: str = "./weights/yolo"

    # Kaggle credentials (optional)
    kaggle_username: Optional[str] = None
    kaggle_key: Optional[str] = None

    # Device
    device: str = "auto"  # auto, cuda, cpu

    # Visualization
    max_viz_images: int = 16
    save_visualizations: bool = True

    def __post_init__(self):
        """Setup directories after initialization"""
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories"""
        for dir_path in [self.root_dir, self.checkpoint_dir, self.logs_dir,
                        self.results_dir, self.yolo_weights_dir]:
            os.makedirs(dir_path, exist_ok=True)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'CCROPConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> 'CCROPConfig':
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        config_dict = self.__dict__.copy()
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def to_json(self, json_path: str):
        """Save configuration to JSON file"""
        config_dict = self.__dict__.copy()
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def get_device(self):
        """Get the appropriate device"""
        import torch
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


def create_default_config(save_path: str = "configs/default_config.yaml"):
    """Create and save default configuration"""
    config = CCROPConfig()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    config.to_yaml(save_path)
    print(f"Default configuration saved to {save_path}")
    return config


if __name__ == "__main__":
    # Create default configuration file
    create_default_config()
