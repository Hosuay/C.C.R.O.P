"""
CLI tool for running predictions with CCROP
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ccrop.models.pipeline import create_pipeline
from src.ccrop.models.classifier import CCROPClassifier
from src.ccrop.models.detector import LeafDetector
from src.ccrop.inference.predictor import StressPredictor, BatchPredictor
import torch


def main():
    parser = argparse.ArgumentParser(description="CCROP - Cannabis Leaf Disease Prediction")

    parser.add_argument('input', type=str,
                       help='Input image path or directory')
    parser.add_argument('--classifier', type=str, required=True,
                       help='Path to classifier checkpoint')
    parser.add_argument('--detector', type=str, default='yolov8n.pt',
                       help='YOLO detector model (default: yolov8n.pt)')
    parser.add_argument('--batch', action='store_true',
                       help='Process directory in batch mode')
    parser.add_argument('--no-detect', action='store_true',
                       help='Skip detection, classify entire image directly')
    parser.add_argument('--output', type=str,
                       help='Output file for results (CSV for batch mode)')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization of detections')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to run on (default: auto)')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='Detection confidence threshold (default: 0.25)')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load models
    print("Loading models...")
    detector = LeafDetector(model_name=args.detector, conf_threshold=args.conf_threshold,
                           device=device)
    classifier, metadata = CCROPClassifier.from_checkpoint(args.classifier, device=device)

    # Create pipeline
    from src.ccrop.models.pipeline import CCROPPipeline
    pipeline = CCROPPipeline(
        detector=detector,
        classifier=classifier,
        class_names=metadata['classes'],
        stress_mapping=metadata['stress_mapping'],
        device=device
    )

    print("Models loaded successfully!")

    # Run prediction
    if args.batch:
        print(f"\nProcessing directory: {args.input}")
        batch_predictor = BatchPredictor(pipeline)
        results_df = batch_predictor.predict_directory(
            args.input,
            detect_first=not args.no_detect,
            output_csv=args.output
        )

        if args.output:
            print(f"\nResults saved to: {args.output}")

    else:
        print(f"\nProcessing image: {args.input}")
        predictor = StressPredictor(pipeline)
        results = predictor.predict_from_path(
            args.input,
            detect_first=not args.no_detect
        )

        # Print results
        print("\n" + predictor.format_results(results))

        # Save visualization if requested
        if args.visualize and args.output:
            import cv2
            from src.ccrop.utils.visualization import plot_detections

            image = cv2.imread(args.input)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            viz = plot_detections(image, results['detections'],
                                save_path=args.output, show=False)
            print(f"\nVisualization saved to: {args.output}")


if __name__ == "__main__":
    main()
