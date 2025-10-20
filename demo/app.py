"""
Gradio Web Demo for CCROP - Cannabis Leaf Disease Detection System

This interactive demo allows users to:
- Upload and analyze single images
- Process multiple images in batch
- View real-time webcam analysis
- Compare detection vs direct classification
- Explore model statistics
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ccrop.models.pipeline import CCROPPipeline, create_pipeline
from src.ccrop.models.classifier import CCROPClassifier
from src.ccrop.models.detector import LeafDetector
from src.ccrop.inference.predictor import StressPredictor
from src.ccrop.utils.visualization import plot_detections


# Global variables for model
pipeline = None
predictor = None


def initialize_models(classifier_path: str = None, detector_model: str = "yolov8n.pt"):
    """Initialize the models"""
    global pipeline, predictor

    print("Initializing CCROP models...")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        if classifier_path and Path(classifier_path).exists():
            # Load from checkpoint
            detector = LeafDetector(model_name=detector_model, device=device)
            classifier, metadata = CCROPClassifier.from_checkpoint(classifier_path, device=device)

            pipeline = CCROPPipeline(
                detector=detector,
                classifier=classifier,
                class_names=metadata['classes'],
                stress_mapping=metadata['stress_mapping'],
                device=device
            )
        else:
            # Create with default models
            print("Warning: No classifier checkpoint provided. Using default model.")
            pipeline = create_pipeline(detector_model=detector_model, device=device)

        predictor = StressPredictor(pipeline)
        print("Models initialized successfully!")

        return "Models loaded successfully!"

    except Exception as e:
        error_msg = f"Error initializing models: {str(e)}"
        print(error_msg)
        return error_msg


def analyze_image(image: np.ndarray, use_detection: bool = True):
    """
    Analyze a single image

    Args:
        image: Input image (RGB numpy array)
        use_detection: Whether to use YOLO detection first

    Returns:
        Tuple of (annotated_image, results_text, stress_score)
    """
    if pipeline is None:
        return None, "Error: Models not initialized. Please initialize models first.", 0

    try:
        # Get predictions
        results = predictor.predict_from_array(image, detect_first=use_detection)

        # Create annotated image
        if use_detection and len(results['detections']) > 0:
            annotated_img = plot_detections(image, results['detections'],
                                           save_path=None, show=False)
        else:
            annotated_img = image

        # Format results
        results_text = predictor.format_results(results)

        # Get overall stress score
        stress_score = results['overall_stress']

        return annotated_img, results_text, stress_score

    except Exception as e:
        return image, f"Error during analysis: {str(e)}", 0


def create_stress_gauge(stress_score: float):
    """Create a visual stress gauge"""
    health = predictor.get_health_status(stress_score) if predictor else {'status': 'Unknown', 'color': 'gray'}

    gauge_html = f"""
    <div style="text-align: center; padding: 20px;">
        <h2 style="color: {health['color']};">{health['status']}</h2>
        <div style="width: 100%; background-color: #ddd; border-radius: 10px; overflow: hidden;">
            <div style="width: {stress_score}%; background-color: {health['color']};
                        height: 40px; line-height: 40px; color: white; font-weight: bold;">
                {stress_score:.1f}%
            </div>
        </div>
        <p style="margin-top: 10px; font-size: 14px;">{health.get('recommendation', '')}</p>
    </div>
    """
    return gauge_html


def batch_analyze(files):
    """Analyze multiple images"""
    if pipeline is None:
        return "Error: Models not initialized."

    if not files:
        return "No files uploaded."

    results_summary = []
    results_summary.append("="*60)
    results_summary.append("BATCH ANALYSIS RESULTS")
    results_summary.append("="*60)

    for i, file in enumerate(files):
        try:
            image = Image.open(file.name)
            image_np = np.array(image)

            results = predictor.predict_from_array(image_np, detect_first=True)

            filename = Path(file.name).name
            stress = results['overall_stress']
            health = predictor.get_health_status(stress)

            results_summary.append(f"\n{i+1}. {filename}")
            results_summary.append(f"   Stress: {stress:.2f}% | Status: {health['status']}")
            results_summary.append(f"   Leaves Detected: {results['num_leaves']}")

        except Exception as e:
            results_summary.append(f"\n{i+1}. {Path(file.name).name}")
            results_summary.append(f"   Error: {str(e)}")

    return "\n".join(results_summary)


def create_demo():
    """Create the Gradio demo interface"""

    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    """

    with gr.Blocks(css=custom_css, title="CCROP - Cannabis Leaf Disease Detection") as demo:

        # Header
        gr.HTML("""
        <div class="header">
            <h1>🌿 C.C.R.O.P - Cannabis Leaf Disease Detection System</h1>
            <p>AI-Powered Plant Health Analysis using YOLO Detection + Deep Learning Classification</p>
        </div>
        """)

        # Model initialization section
        with gr.Accordion("⚙️ Model Configuration", open=False):
            with gr.Row():
                classifier_path_input = gr.Textbox(
                    label="Classifier Checkpoint Path (optional)",
                    placeholder="./checkpoints/best_model.pth",
                    value=""
                )
                detector_model_input = gr.Dropdown(
                    label="YOLO Detector Model",
                    choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                    value="yolov8n.pt"
                )

            init_button = gr.Button("Initialize Models", variant="primary")
            init_status = gr.Textbox(label="Initialization Status", interactive=False)

            init_button.click(
                fn=initialize_models,
                inputs=[classifier_path_input, detector_model_input],
                outputs=init_status
            )

        # Main tabs
        with gr.Tabs():

            # Tab 1: Single Image Analysis
            with gr.Tab("📸 Single Image Analysis"):
                gr.Markdown("""
                ### Upload an image of a cannabis leaf for disease analysis
                The system will detect individual leaves (if multiple) and classify each for disease/stress indicators.
                """)

                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="Upload Image", type="numpy")
                        use_detection_checkbox = gr.Checkbox(
                            label="Use YOLO Detection (detect individual leaves first)",
                            value=True
                        )
                        analyze_button = gr.Button("🔍 Analyze Image", variant="primary")

                    with gr.Column():
                        output_image = gr.Image(label="Analysis Results")
                        stress_gauge = gr.HTML(label="Stress Level")

                results_text = gr.Textbox(
                    label="Detailed Results",
                    lines=20,
                    max_lines=30
                )

                analyze_button.click(
                    fn=lambda img, use_det: (
                        analyze_image(img, use_det)[0],
                        analyze_image(img, use_det)[1],
                        create_stress_gauge(analyze_image(img, use_det)[2])
                    ),
                    inputs=[input_image, use_detection_checkbox],
                    outputs=[output_image, results_text, stress_gauge]
                )

            # Tab 2: Batch Processing
            with gr.Tab("📁 Batch Processing"):
                gr.Markdown("""
                ### Upload multiple images for batch analysis
                Process multiple leaf images at once and get a summary of results.
                """)

                batch_input = gr.File(
                    label="Upload Images",
                    file_count="multiple",
                    file_types=["image"]
                )
                batch_button = gr.Button("🔄 Process Batch", variant="primary")
                batch_output = gr.Textbox(
                    label="Batch Results",
                    lines=20,
                    max_lines=40
                )

                batch_button.click(
                    fn=batch_analyze,
                    inputs=batch_input,
                    outputs=batch_output
                )

            # Tab 3: Model Comparison
            with gr.Tab("⚖️ Detection vs Direct Classification"):
                gr.Markdown("""
                ### Compare results with and without YOLO detection
                See how the detection step affects the final classification.
                """)

                with gr.Row():
                    compare_input = gr.Image(label="Upload Image", type="numpy")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**With YOLO Detection**")
                        compare_detect_img = gr.Image(label="Detected Leaves")
                        compare_detect_text = gr.Textbox(label="Results", lines=10)

                    with gr.Column():
                        gr.Markdown("**Direct Classification**")
                        compare_direct_img = gr.Image(label="Direct Classification")
                        compare_direct_text = gr.Textbox(label="Results", lines=10)

                compare_button = gr.Button("⚖️ Compare Methods", variant="primary")

                def compare_methods(image):
                    if image is None:
                        return None, "No image provided", None, "No image provided"

                    # With detection
                    detect_img, detect_text, _ = analyze_image(image, use_detection=True)

                    # Without detection
                    direct_img, direct_text, _ = analyze_image(image, use_detection=False)

                    return detect_img, detect_text, direct_img, direct_text

                compare_button.click(
                    fn=compare_methods,
                    inputs=compare_input,
                    outputs=[compare_detect_img, compare_detect_text,
                            compare_direct_img, compare_direct_text]
                )

            # Tab 4: About
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                # C.C.R.O.P - Center for Cannabaceae Research in Optimization of Plant-Health

                ## Overview
                This AI-powered system provides automated cannabis leaf disease detection and stress analysis
                using state-of-the-art computer vision and deep learning techniques.

                ## Technology Stack
                - **Detection**: YOLOv8 for leaf localization and detection
                - **Classification**: ResNet/EfficientNet for disease classification
                - **Framework**: PyTorch for deep learning
                - **Interface**: Gradio for interactive web demo

                ## Features
                - ✅ Automatic leaf detection and localization
                - ✅ Multi-class disease classification
                - ✅ Stress score calculation (0-100 scale)
                - ✅ Batch processing support
                - ✅ Real-time analysis
                - ✅ Visual detection overlays

                ## How It Works
                1. **Detection Phase**: YOLOv8 identifies individual leaves in the image
                2. **Classification Phase**: Each detected leaf is analyzed for disease/stress
                3. **Aggregation**: Results are combined to provide overall plant health assessment

                ## Health Status Categories
                - 🟢 **Healthy** (0-33%): Plant shows no signs of stress or disease
                - 🟠 **Moderate Stress** (33-66%): Early signs of stress, monitoring recommended
                - 🔴 **Severe Stress** (66-100%): Significant disease/stress, immediate action needed

                ## Citation
                If you use this system in your research, please cite:
                ```
                C.C.R.O.P - Cannabis Leaf Disease Detection System
                AI-powered plant health monitoring using YOLO and deep learning
                ```

                ## Contact & Support
                For questions, collaborations, or support, please open an issue on our GitHub repository.

                ---
                **Version**: 2.0.0
                **Last Updated**: 2024
                """)

        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #666;">
            <p>© 2024 C.C.R.O.P Research Team | AI-Powered Plant Health Monitoring</p>
        </div>
        """)

    return demo


if __name__ == "__main__":
    print("="*60)
    print("Starting CCROP Demo Application")
    print("="*60)

    # Create and launch demo
    demo = create_demo()

    # Launch with sharing enabled for easy access
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create a public link
        show_error=True
    )
