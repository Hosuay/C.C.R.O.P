"""
Inference and prediction utilities
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Union, Optional
from pathlib import Path
from PIL import Image
import pandas as pd

from ..models.pipeline import CCROPPipeline


class StressPredictor:
    """Enhanced stress prediction with confidence scores"""

    def __init__(self, pipeline: CCROPPipeline):
        """
        Initialize predictor

        Args:
            pipeline: CCROPPipeline instance
        """
        self.pipeline = pipeline

    def predict_from_path(self, img_path: str, detect_first: bool = True,
                         top_k: int = 3) -> Dict:
        """
        Predict stress from image path

        Args:
            img_path: Path to image
            detect_first: Whether to use YOLO detection first
            top_k: Number of top predictions

        Returns:
            Prediction results dictionary
        """
        return self.pipeline.process_image(img_path, detect_first=detect_first, top_k=top_k)

    def predict_from_array(self, image: np.ndarray, detect_first: bool = True,
                          top_k: int = 3) -> Dict:
        """
        Predict stress from numpy array (RGB format)

        Args:
            image: Input image (RGB)
            detect_first: Whether to use YOLO detection first
            top_k: Number of top predictions

        Returns:
            Prediction results dictionary
        """
        return self.pipeline.process_image(image, detect_first=detect_first, top_k=top_k)

    def get_health_status(self, stress_score: float) -> Dict:
        """
        Get health status from stress score

        Args:
            stress_score: Stress score (0-100)

        Returns:
            Health status dictionary
        """
        return self.pipeline.get_health_status(stress_score)

    def format_results(self, results: Dict) -> str:
        """
        Format results for display

        Args:
            results: Prediction results

        Returns:
            Formatted string
        """
        output = []
        output.append("="*60)
        output.append("CCROP - Leaf Disease Detection Results")
        output.append("="*60)

        output.append(f"\nProcessing Mode: {results['processing_mode']}")
        output.append(f"Number of Leaves Detected: {results['num_leaves']}")
        output.append(f"Overall Stress Score: {results['overall_stress']:.2f}%")

        health = self.get_health_status(results['overall_stress'])
        output.append(f"Health Status: {health['status']} ({health['color'].upper()})")
        output.append(f"Recommendation: {health['recommendation']}")

        if results['num_leaves'] > 1:
            output.append(f"\nStress Statistics:")
            output.append(f"  Min: {results.get('min_stress', 0):.2f}%")
            output.append(f"  Max: {results.get('max_stress', 0):.2f}%")
            output.append(f"  Std: {results.get('stress_std', 0):.2f}%")

        output.append(f"\n{'='*60}")
        output.append("Individual Leaf Detections:")
        output.append("="*60)

        for i, det in enumerate(results['detections']):
            output.append(f"\nLeaf #{i+1}:")
            output.append(f"  Disease Class: {det.get('disease_class', 'N/A')}")
            output.append(f"  Confidence: {det.get('disease_confidence', 0):.2f}%")
            output.append(f"  Stress Score: {det.get('stress_score', 0):.2f}%")

            if 'top_predictions' in det:
                output.append(f"  Top Predictions:")
                for j, pred in enumerate(det['top_predictions'][:3]):
                    output.append(f"    {j+1}. {pred['class']}: {pred['probability']*100:.2f}%")

        return "\n".join(output)


class BatchPredictor:
    """Batch prediction for multiple images"""

    def __init__(self, pipeline: CCROPPipeline):
        """
        Initialize batch predictor

        Args:
            pipeline: CCROPPipeline instance
        """
        self.pipeline = pipeline
        self.predictor = StressPredictor(pipeline)

    def predict_directory(self, image_dir: str, detect_first: bool = True,
                         output_csv: Optional[str] = None,
                         image_extensions: List[str] = None) -> pd.DataFrame:
        """
        Run inference on all images in a directory

        Args:
            image_dir: Directory containing images
            detect_first: Whether to use YOLO detection first
            output_csv: Path to save results CSV
            image_extensions: List of image extensions to process

        Returns:
            DataFrame with results
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        image_paths = []
        for ext in image_extensions:
            image_paths.extend(Path(image_dir).glob(f"*{ext}"))
            image_paths.extend(Path(image_dir).glob(f"*{ext.upper()}"))

        if not image_paths:
            print(f"No images found in {image_dir}")
            return pd.DataFrame()

        print(f"Processing {len(image_paths)} images...")

        results = []
        for i, img_path in enumerate(image_paths):
            try:
                result = self.predictor.predict_from_path(str(img_path), detect_first=detect_first)

                # Extract summary info
                results.append({
                    'filename': img_path.name,
                    'filepath': str(img_path),
                    'num_leaves': result['num_leaves'],
                    'overall_stress': result['overall_stress'],
                    'max_stress': result.get('max_stress', result['overall_stress']),
                    'min_stress': result.get('min_stress', result['overall_stress']),
                    'health_status': self.predictor.get_health_status(result['overall_stress'])['status']
                })

                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(image_paths)} images")

            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
                results.append({
                    'filename': img_path.name,
                    'filepath': str(img_path),
                    'error': str(e)
                })

        # Create DataFrame
        df_results = pd.DataFrame(results)

        # Save to CSV if requested
        if output_csv:
            df_results.to_csv(output_csv, index=False)
            print(f"\nResults saved to {output_csv}")

        # Print summary
        self._print_summary(df_results)

        return df_results

    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("Batch Inference Summary")
        print("="*60)

        if 'error' in df.columns:
            errors = df['error'].notna().sum()
            if errors > 0:
                print(f"Errors encountered: {errors}")

        valid_results = df[df['overall_stress'].notna()]

        if len(valid_results) > 0:
            print(f"Total images processed: {len(valid_results)}")
            print(f"Average stress score: {valid_results['overall_stress'].mean():.2f}%")
            print(f"Std deviation: {valid_results['overall_stress'].std():.2f}%")
            print(f"Min stress: {valid_results['overall_stress'].min():.2f}%")
            print(f"Max stress: {valid_results['overall_stress'].max():.2f}%")

            # Stress distribution
            if 'health_status' in valid_results.columns:
                print(f"\nHealth Status Distribution:")
                status_counts = valid_results['health_status'].value_counts()
                for status, count in status_counts.items():
                    print(f"  {status}: {count} ({count/len(valid_results)*100:.1f}%)")


def create_predictor(pipeline: CCROPPipeline) -> StressPredictor:
    """
    Factory function to create a predictor

    Args:
        pipeline: CCROPPipeline instance

    Returns:
        StressPredictor instance
    """
    return StressPredictor(pipeline)


if __name__ == "__main__":
    print("Predictor module loaded successfully!")
