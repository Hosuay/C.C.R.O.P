"""
Visualization utilities for CCROP system
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from sklearn.metrics import confusion_matrix
import torch


def plot_training_history(history: Dict, save_path: Optional[str] = None, show: bool = True):
    """
    Plot training history including loss, accuracy, and learning rate

    Args:
        history: Dictionary containing training metrics
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history['epochs'], history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['epochs'], history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['epochs'], history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['epochs'], history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Learning Rate
    axes[2].plot(history['epochs'], history['learning_rates'], linewidth=2, color='green')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names: List[str],
                         save_path: Optional[str] = None, show: bool = True):
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(max(12, len(class_names) * 0.8), max(10, len(class_names) * 0.7)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_predictions(images: List[np.ndarray], predictions: List[Dict],
                         save_path: Optional[str] = None, show: bool = True,
                         max_images: int = 16):
    """
    Visualize predictions on multiple images

    Args:
        images: List of images (numpy arrays in RGB format)
        predictions: List of prediction dictionaries
        save_path: Path to save the visualization
        show: Whether to display the plot
        max_images: Maximum number of images to display
    """
    n_images = min(len(images), max_images)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for idx in range(n_images):
        image = images[idx]
        pred = predictions[idx]

        # Display image
        axes[idx].imshow(image)
        axes[idx].axis('off')

        # Create title with results
        stress = pred.get('stress_score', 0)
        conf = pred.get('confidence', 0)
        top_class = pred.get('top_class', 'Unknown')

        # Determine color based on stress level
        if stress < 33:
            color = 'green'
            status = 'Healthy'
        elif stress < 66:
            color = 'orange'
            status = 'Moderate'
        else:
            color = 'red'
            status = 'Severe'

        title = f"{status}\nStress: {stress:.1f}% | Conf: {conf:.1f}%\n{top_class[:25]}"
        axes[idx].set_title(title, fontsize=9, color=color, weight='bold')

    # Hide extra subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_detections(image: np.ndarray, detections: List[Dict],
                   save_path: Optional[str] = None, show: bool = True) -> np.ndarray:
    """
    Plot YOLO detections with bounding boxes and labels

    Args:
        image: Input image (RGB format)
        detections: List of detection dictionaries with 'bbox', 'confidence', 'class'
        save_path: Path to save the image
        show: Whether to display the image

    Returns:
        Image with drawn detections
    """
    img_draw = image.copy()

    for det in detections:
        bbox = det['bbox']  # [x1, y1, x2, y2]
        conf = det.get('confidence', 0)
        cls = det.get('class', 'leaf')

        # Get stress info if available
        stress = det.get('stress_score')
        disease = det.get('disease_class')

        # Determine color based on stress or default
        if stress is not None:
            if stress < 33:
                color = (0, 255, 0)  # Green
            elif stress < 66:
                color = (255, 165, 0)  # Orange
            else:
                color = (255, 0, 0)  # Red
        else:
            color = (0, 255, 255)  # Yellow (default)

        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

        # Prepare label
        if stress is not None and disease is not None:
            label = f"{disease[:20]}: {stress:.1f}% ({conf:.2f})"
        else:
            label = f"{cls}: {conf:.2f}"

        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_draw, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1), color, -1)

        # Draw label text
        cv2.putText(img_draw, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
        print(f"Detection visualization saved to {save_path}")

    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(img_draw)
        plt.axis('off')
        plt.title('Leaf Detection and Classification', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    return img_draw


def create_comparison_grid(images: List[np.ndarray], titles: List[str],
                          save_path: Optional[str] = None, show: bool = True):
    """
    Create a grid comparing multiple images (e.g., original vs detection vs classification)

    Args:
        images: List of images to compare
        titles: List of titles for each image
        save_path: Path to save the comparison
        show: Whether to display the comparison
    """
    n_images = len(images)

    fig, axes = plt.subplots(1, n_images, figsize=(6*n_images, 6))
    if n_images == 1:
        axes = [axes]

    for idx, (img, title) in enumerate(zip(images, titles)):
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison grid saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_stress_distribution(stress_scores: List[float], save_path: Optional[str] = None,
                            show: bool = True):
    """
    Plot distribution of stress scores

    Args:
        stress_scores: List of stress scores
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(stress_scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(stress_scores), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(stress_scores):.1f}%')
    axes[0].set_xlabel('Stress Score (%)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Stress Score Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Box plot by category
    categories = []
    for score in stress_scores:
        if score < 33:
            categories.append('Healthy')
        elif score < 66:
            categories.append('Moderate')
        else:
            categories.append('Severe')

    category_order = ['Healthy', 'Moderate', 'Severe']
    colors = ['green', 'orange', 'red']

    box_data = []
    for cat in category_order:
        cat_scores = [s for s, c in zip(stress_scores, categories) if c == cat]
        if cat_scores:
            box_data.append(cat_scores)
        else:
            box_data.append([])

    bp = axes[1].boxplot([d for d in box_data if d], labels=[c for c, d in zip(category_order, box_data) if d],
                         patch_artist=True)

    for patch, color in zip(bp['boxes'], [c for c, d in zip(colors, box_data) if d]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[1].set_ylabel('Stress Score (%)', fontsize=12)
    axes[1].set_title('Stress Scores by Category', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Stress distribution plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
