"""
YOLO-based leaf detection model
"""

import torch
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from PIL import Image


class LeafDetector:
    """
    YOLO-based leaf detector for localizing leaves in images

    Uses Ultralytics YOLOv8 for detecting individual leaves in cannabis plant images.
    """

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45, device: str = 'cpu'):
        """
        Initialize the leaf detector

        Args:
            model_name: YOLO model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Ultralytics YOLO not installed. Install with: pip install ultralytics")

        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Load YOLO model
        print(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)

        # For custom trained models, we might have specific leaf classes
        self.leaf_classes = []  # Will be populated if using custom trained model

    def detect(self, image: Union[np.ndarray, str, Path],
               return_crops: bool = False) -> Dict:
        """
        Detect leaves in an image

        Args:
            image: Input image (numpy array, file path, or PIL Image)
            return_crops: Whether to return cropped leaf images

        Returns:
            Dictionary containing:
                - 'boxes': List of bounding boxes [x1, y1, x2, y2]
                - 'confidences': List of confidence scores
                - 'classes': List of class names
                - 'crops': List of cropped images (if return_crops=True)
        """
        # Run inference
        results = self.model(image, conf=self.conf_threshold,
                           iou=self.iou_threshold, device=self.device)

        detections = {
            'boxes': [],
            'confidences': [],
            'classes': [],
            'crops': [] if return_crops else None
        }

        # Process results
        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            # Extract detection information
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections['boxes'].append([float(x1), float(y1), float(x2), float(y2)])

                # Get confidence
                conf = float(box.conf[0].cpu().numpy())
                detections['confidences'].append(conf)

                # Get class name
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = result.names[cls_id]
                detections['classes'].append(cls_name)

                # Optionally crop detected regions
                if return_crops:
                    if isinstance(image, (str, Path)):
                        img = cv2.imread(str(image))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    elif isinstance(image, np.ndarray):
                        img = image.copy()
                    else:
                        img = np.array(image)

                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    crop = img[y1:y2, x1:x2]
                    detections['crops'].append(crop)

        return detections

    def detect_batch(self, images: List[Union[np.ndarray, str, Path]],
                    return_crops: bool = False) -> List[Dict]:
        """
        Detect leaves in multiple images

        Args:
            images: List of images
            return_crops: Whether to return cropped leaf images

        Returns:
            List of detection dictionaries
        """
        results = []
        for img in images:
            det = self.detect(img, return_crops=return_crops)
            results.append(det)
        return results

    def visualize_detections(self, image: Union[np.ndarray, str, Path],
                            save_path: Optional[str] = None,
                            show: bool = True) -> np.ndarray:
        """
        Visualize detections on image

        Args:
            image: Input image
            save_path: Path to save visualization
            show: Whether to display the image

        Returns:
            Image with drawn detections
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            img = np.array(image)

        # Get detections
        detections = self.detect(image)

        # Draw bounding boxes
        for bbox, conf, cls in zip(detections['boxes'],
                                   detections['confidences'],
                                   detections['classes']):
            x1, y1, x2, y2 = map(int, bbox)

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{cls}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Detection visualization saved to {save_path}")

        if show:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Leaf Detection Results')
            plt.tight_layout()
            plt.show()

        return img

    def train_custom_model(self, data_yaml: str, epochs: int = 100,
                          imgsz: int = 640, batch: int = 16,
                          save_dir: str = './runs/detect'):
        """
        Train a custom YOLO model for leaf detection

        Args:
            data_yaml: Path to dataset YAML configuration
            epochs: Number of training epochs
            imgsz: Image size
            batch: Batch size
            save_dir: Directory to save training results
        """
        print(f"Training custom YOLO model...")
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=save_dir,
            name='leaf_detector'
        )
        print(f"Training complete. Results saved to {save_dir}")
        return results

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, conf_threshold: float = 0.25,
                       iou_threshold: float = 0.45, device: str = 'cpu'):
        """
        Load detector from custom checkpoint

        Args:
            checkpoint_path: Path to YOLO checkpoint (.pt file)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            device: Device to run on

        Returns:
            LeafDetector instance
        """
        detector = cls(model_name=checkpoint_path, conf_threshold=conf_threshold,
                      iou_threshold=iou_threshold, device=device)
        return detector


def create_detector(model_name: str = "yolov8n.pt", conf_threshold: float = 0.25,
                   device: str = 'cpu') -> LeafDetector:
    """
    Factory function to create a leaf detector

    Args:
        model_name: YOLO model name
        conf_threshold: Confidence threshold
        device: Device to run on

    Returns:
        LeafDetector instance
    """
    return LeafDetector(model_name=model_name, conf_threshold=conf_threshold, device=device)


if __name__ == "__main__":
    # Test detector
    print("Testing LeafDetector...")

    try:
        detector = create_detector(model_name="yolov8n.pt")
        print(f"\nDetector created with model: {detector.model_name}")
        print(f"Confidence threshold: {detector.conf_threshold}")
        print("\nDetector test passed!")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install ultralytics: pip install ultralytics")
