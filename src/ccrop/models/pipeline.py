"""
Integrated pipeline combining YOLO detection and disease classification
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Union, Optional, Tuple
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

from .detector import LeafDetector
from .classifier import CCROPClassifier


class CCROPPipeline:
    """
    End-to-end pipeline for cannabis leaf disease detection and classification

    Pipeline:
    1. Detect leaves using YOLO
    2. Crop detected leaf regions
    3. Classify each leaf for disease/stress
    4. Aggregate results
    """

    def __init__(self, detector: LeafDetector, classifier: CCROPClassifier,
                 class_names: List[str], stress_mapping: Dict[str, float],
                 transform: Optional[transforms.Compose] = None,
                 device: str = 'cpu', min_leaf_area: int = 100):
        """
        Initialize the pipeline

        Args:
            detector: LeafDetector instance
            classifier: CCROPClassifier instance
            class_names: List of disease class names
            stress_mapping: Mapping from class names to stress scores (0-100)
            transform: Image transforms for classifier input
            device: Device to run inference on
            min_leaf_area: Minimum leaf area in pixels to process
        """
        self.detector = detector
        self.classifier = classifier.to(device)
        self.classifier.eval()

        self.class_names = sorted(class_names)
        self.stress_mapping = stress_mapping
        self.device = device
        self.min_leaf_area = min_leaf_area

        # Default transform if not provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def process_image(self, image: Union[np.ndarray, str, Path],
                     detect_first: bool = True,
                     top_k: int = 3) -> Dict:
        """
        Process a single image through the complete pipeline

        Args:
            image: Input image (numpy array, file path, or PIL Image)
            detect_first: If True, use YOLO detection first; otherwise classify entire image
            top_k: Number of top predictions to return per detection

        Returns:
            Dictionary containing detection and classification results
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Cannot read image: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()

        results = {
            'image_shape': img.shape,
            'detections': [],
            'overall_stress': 0.0,
            'num_leaves': 0,
            'processing_mode': 'detection' if detect_first else 'direct_classification'
        }

        if detect_first:
            # Step 1: Detect leaves
            detections = self.detector.detect(img, return_crops=True)

            if len(detections['boxes']) == 0:
                print("No leaves detected in image. Falling back to direct classification.")
                return self._classify_whole_image(img, top_k)

            # Step 2: Filter detections by area
            valid_detections = []
            for i, bbox in enumerate(detections['boxes']):
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)

                if area >= self.min_leaf_area:
                    valid_detections.append({
                        'bbox': bbox,
                        'confidence': detections['confidences'][i],
                        'class': detections['classes'][i],
                        'crop': detections['crops'][i] if detections['crops'] else None
                    })

            if len(valid_detections) == 0:
                print("No valid leaf detections. Falling back to direct classification.")
                return self._classify_whole_image(img, top_k)

            # Step 3: Classify each detected leaf
            stress_scores = []

            for det in valid_detections:
                if det['crop'] is not None:
                    # Classify the cropped leaf
                    classification = self._classify_crop(det['crop'], top_k)

                    # Add classification results to detection
                    det.update({
                        'disease_class': classification['top_class'],
                        'disease_confidence': classification['confidence'],
                        'stress_score': classification['stress_score'],
                        'top_predictions': classification['top_predictions']
                    })

                    stress_scores.append(classification['stress_score'])

                results['detections'].append(det)

            # Calculate overall metrics
            results['num_leaves'] = len(valid_detections)
            results['overall_stress'] = np.mean(stress_scores) if stress_scores else 0.0
            results['stress_std'] = np.std(stress_scores) if len(stress_scores) > 1 else 0.0
            results['max_stress'] = np.max(stress_scores) if stress_scores else 0.0
            results['min_stress'] = np.min(stress_scores) if stress_scores else 0.0

        else:
            # Direct classification without detection
            return self._classify_whole_image(img, top_k)

        return results

    def _classify_crop(self, crop: np.ndarray, top_k: int = 3) -> Dict:
        """
        Classify a cropped leaf image

        Args:
            crop: Cropped leaf image (RGB numpy array)
            top_k: Number of top predictions to return

        Returns:
            Classification results dictionary
        """
        # Convert to PIL for transform
        pil_img = Image.fromarray(crop)

        # Transform and add batch dimension
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.classifier(img_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

        # Calculate stress score
        stress_scores = np.array([self.stress_mapping.get(c, 50.0) for c in self.class_names])
        stress_score = float(np.sum(probs * stress_scores))

        # Get top-k predictions
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_predictions = [
            {
                'class': self.class_names[i],
                'probability': float(probs[i]),
                'stress': float(self.stress_mapping.get(self.class_names[i], 50.0))
            }
            for i in top_indices
        ]

        return {
            'top_class': self.class_names[top_indices[0]],
            'confidence': float(probs[top_indices[0]] * 100),
            'stress_score': stress_score,
            'top_predictions': top_predictions,
            'all_probabilities': {c: float(p) for c, p in zip(self.class_names, probs)}
        }

    def _classify_whole_image(self, image: np.ndarray, top_k: int = 3) -> Dict:
        """
        Classify the entire image without detection

        Args:
            image: Input image (RGB numpy array)
            top_k: Number of top predictions to return

        Returns:
            Classification results dictionary
        """
        classification = self._classify_crop(image, top_k)

        return {
            'image_shape': image.shape,
            'detections': [{
                'bbox': [0, 0, image.shape[1], image.shape[0]],  # Full image
                'confidence': 1.0,
                'class': 'whole_image',
                'disease_class': classification['top_class'],
                'disease_confidence': classification['confidence'],
                'stress_score': classification['stress_score'],
                'top_predictions': classification['top_predictions']
            }],
            'overall_stress': classification['stress_score'],
            'num_leaves': 1,
            'processing_mode': 'direct_classification'
        }

    def process_batch(self, images: List[Union[np.ndarray, str, Path]],
                     detect_first: bool = True, top_k: int = 3) -> List[Dict]:
        """
        Process multiple images

        Args:
            images: List of images
            detect_first: Whether to use detection first
            top_k: Number of top predictions per detection

        Returns:
            List of result dictionaries
        """
        results = []
        for img in images:
            try:
                result = self.process_image(img, detect_first=detect_first, top_k=top_k)
                results.append(result)
            except Exception as e:
                print(f"Error processing image: {e}")
                results.append({'error': str(e)})

        return results

    def get_health_status(self, stress_score: float) -> Dict[str, str]:
        """
        Get health status based on stress score

        Args:
            stress_score: Stress score (0-100)

        Returns:
            Dictionary with status, color, and recommendation
        """
        if stress_score < 33:
            return {
                'status': 'Healthy',
                'color': 'green',
                'level': 'low',
                'recommendation': 'Plant is healthy. Continue current care routine.'
            }
        elif stress_score < 66:
            return {
                'status': 'Moderate Stress',
                'color': 'orange',
                'level': 'medium',
                'recommendation': 'Monitor closely. Consider adjusting water, light, or nutrients.'
            }
        else:
            return {
                'status': 'Severe Stress',
                'color': 'red',
                'level': 'high',
                'recommendation': 'Immediate attention required. Check for pests, diseases, or environmental issues.'
            }

    @classmethod
    def from_checkpoints(cls, detector_path: str, classifier_path: str,
                        device: str = 'cpu', conf_threshold: float = 0.25) -> 'CCROPPipeline':
        """
        Create pipeline from checkpoint files

        Args:
            detector_path: Path to YOLO detector checkpoint
            classifier_path: Path to classifier checkpoint
            device: Device to run on
            conf_threshold: Detection confidence threshold

        Returns:
            CCROPPipeline instance
        """
        # Load detector
        detector = LeafDetector.from_checkpoint(detector_path, conf_threshold=conf_threshold,
                                               device=device)

        # Load classifier
        classifier, metadata = CCROPClassifier.from_checkpoint(classifier_path, device=device)

        # Create pipeline
        pipeline = cls(
            detector=detector,
            classifier=classifier,
            class_names=metadata['classes'],
            stress_mapping=metadata['stress_mapping'],
            device=device
        )

        return pipeline


def create_pipeline(detector_model: str = "yolov8n.pt",
                   classifier_checkpoint: str = None,
                   device: str = 'cpu') -> CCROPPipeline:
    """
    Factory function to create a pipeline

    Args:
        detector_model: YOLO model name or path
        classifier_checkpoint: Path to classifier checkpoint
        device: Device to run on

    Returns:
        CCROPPipeline instance
    """
    # Create detector
    detector = LeafDetector(model_name=detector_model, device=device)

    # Load classifier
    if classifier_checkpoint:
        classifier, metadata = CCROPClassifier.from_checkpoint(classifier_checkpoint, device=device)
        class_names = metadata['classes']
        stress_mapping = metadata['stress_mapping']
    else:
        # Create default classifier (for testing)
        print("Warning: No classifier checkpoint provided. Creating default classifier.")
        classifier = CCROPClassifier(arch="resnet18", num_classes=38)
        class_names = [f"class_{i}" for i in range(38)]
        stress_mapping = {c: i * 100.0 / 37 for i, c in enumerate(class_names)}

    # Create pipeline
    pipeline = CCROPPipeline(
        detector=detector,
        classifier=classifier,
        class_names=class_names,
        stress_mapping=stress_mapping,
        device=device
    )

    return pipeline


if __name__ == "__main__":
    # Test pipeline creation
    print("Testing CCROPPipeline...")

    try:
        # Create a test pipeline
        print("\nCreating test pipeline...")
        pipeline = create_pipeline(detector_model="yolov8n.pt", device='cpu')

        print(f"Pipeline created successfully!")
        print(f"Number of disease classes: {len(pipeline.class_names)}")
        print(f"Detection mode enabled: True")
        print(f"Device: {pipeline.device}")

        print("\nPipeline test passed!")
    except Exception as e:
        print(f"Error: {e}")
