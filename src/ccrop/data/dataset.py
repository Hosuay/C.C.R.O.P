"""
Dataset management and loading utilities
"""

import os
import zipfile
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import torch
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from torchvision import datasets
import numpy as np

from .transforms import get_train_transforms, get_val_transforms


class CCROPDataset(Dataset):
    """Custom dataset wrapper for CCROP"""

    def __init__(self, root_dir: str, transform=None):
        """
        Initialize dataset

        Args:
            root_dir: Root directory containing class folders
            transform: Image transformations
        """
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_class_counts(self):
        """Get count of samples per class"""
        targets = [self.dataset.targets[i] for i in range(len(self.dataset))]
        return np.bincount(targets)


class DatasetManager:
    """Manages dataset download, extraction, and organization"""

    def __init__(self, root_dir: str = "./dataset"):
        """
        Initialize dataset manager

        Args:
            root_dir: Root directory for dataset
        """
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)

    def download_from_kaggle(self, dataset_name: str, fallback_name: Optional[str] = None):
        """
        Download dataset from Kaggle

        Args:
            dataset_name: Kaggle dataset name (user/dataset-name)
            fallback_name: Fallback dataset name if primary fails
        """
        print(f"Downloading dataset: {dataset_name}")
        status = os.system(f"kaggle datasets download -d {dataset_name} -p {self.root_dir} 2>/dev/null")

        if status != 0 and fallback_name:
            print(f"Primary dataset failed. Trying fallback: {fallback_name}")
            status = os.system(f"kaggle datasets download -d {fallback_name} -p {self.root_dir} 2>/dev/null")

        if status != 0:
            raise RuntimeError("Failed to download dataset. Check Kaggle credentials and dataset availability.")

        print("Dataset downloaded successfully")

    def extract_dataset(self):
        """Extract ZIP files in dataset directory"""
        zip_files = list(Path(self.root_dir).glob("*.zip"))

        if not zip_files:
            print("No ZIP files found. Dataset may already be extracted.")
            return

        for zip_file in zip_files:
            print(f"Extracting: {zip_file.name}")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
            print(f"Extracted: {zip_file.name}")

    def find_dataset_path(self) -> str:
        """
        Auto-detect dataset folder with class subdirectories

        Returns:
            Path to dataset folder
        """
        def has_multiple_class_folders(path):
            try:
                subdirs = [d for d in os.listdir(path)
                          if os.path.isdir(os.path.join(path, d)) and not d.startswith('.')]
                return len(subdirs) > 1
            except:
                return False

        # Search for valid dataset folder
        for root, dirs, _ in os.walk(self.root_dir):
            if has_multiple_class_folders(root):
                # Check for 'color' subfolder (Plant Diseases dataset structure)
                color_dir = os.path.join(root, "color")
                if os.path.exists(color_dir) and has_multiple_class_folders(color_dir):
                    return color_dir
                return root

        raise FileNotFoundError("Unable to locate valid dataset folder with multiple class directories")

    def get_class_info(self, dataset_path: str) -> Tuple[List[str], Dict[str, float]]:
        """
        Get class names and create stress mapping

        Args:
            dataset_path: Path to dataset

        Returns:
            Tuple of (class_names, stress_mapping)
        """
        classes = sorted([d for d in os.listdir(dataset_path)
                         if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')])

        if len(classes) == 0:
            raise ValueError("No class folders found in dataset")

        # Create stress mapping (0-100 scale)
        # Assumes classes are sorted such that healthier classes come first
        if len(classes) == 1:
            stress_mapping = {classes[0]: 50.0}
        else:
            stress_mapping = {cls: idx * 100.0 / (len(classes) - 1)
                            for idx, cls in enumerate(classes)}

        return classes, stress_mapping

    def setup_kaggle_credentials(self, kaggle_json_path: Optional[str] = None,
                                 username: Optional[str] = None,
                                 api_key: Optional[str] = None) -> bool:
        """
        Setup Kaggle credentials

        Args:
            kaggle_json_path: Path to kaggle.json file
            username: Kaggle username
            api_key: Kaggle API key

        Returns:
            True if credentials are set up successfully
        """
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_config_path = kaggle_dir / "kaggle.json"

        # Check if kaggle.json already exists
        if kaggle_config_path.exists():
            print(f"Using existing kaggle.json from {kaggle_config_path}")
            return True

        # Copy from provided path
        if kaggle_json_path and Path(kaggle_json_path).exists():
            import shutil
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(kaggle_json_path, kaggle_config_path)
            kaggle_config_path.chmod(0o600)
            print(f"Kaggle credentials installed to {kaggle_config_path}")
            return True

        # Use username and API key
        if username and api_key:
            os.environ['KAGGLE_USERNAME'] = username
            os.environ['KAGGLE_KEY'] = api_key
            print("Kaggle credentials set from environment")
            return True

        print("No Kaggle credentials found")
        return False


def get_dataloaders(dataset_path: str, batch_size: int = 16, num_workers: int = 2,
                   train_split: float = 0.8, val_split: float = 0.1,
                   input_size: int = 224, use_weighted_sampler: bool = True,
                   seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create train, validation, and test dataloaders

    Args:
        dataset_path: Path to dataset directory
        batch_size: Batch size
        num_workers: Number of worker processes
        train_split: Fraction for training
        val_split: Fraction for validation
        input_size: Input image size
        use_weighted_sampler: Use weighted sampling for class balance
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    # Get transforms
    train_transform = get_train_transforms(input_size)
    val_transform = get_val_transforms(input_size)

    # Load full dataset
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transform)
    class_names = full_dataset.classes

    # Calculate split sizes
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Apply different transforms for validation and test
    val_dataset.dataset = datasets.ImageFolder(root=dataset_path, transform=val_transform)
    test_dataset.dataset = datasets.ImageFolder(root=dataset_path, transform=val_transform)

    # Create dataloaders
    if use_weighted_sampler:
        # Calculate class weights for balanced training
        all_targets = [full_dataset.targets[i] for i in range(len(full_dataset))]
        class_counts = np.bincount(all_targets)
        class_weights = 1.0 / torch.Tensor(class_counts)
        sample_weights = [class_weights[full_dataset.targets[i]] for i in train_dataset.indices]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    print(f"Number of classes: {len(class_names)}")

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    # Test dataset management
    print("Testing DatasetManager...")

    dm = DatasetManager(root_dir="./test_dataset")
    print(f"Dataset manager created with root: {dm.root_dir}")

    print("\nDataset management test passed!")
