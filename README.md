# C.C.R.O.P
🌿 Cannabis Leaf Disease Detection (AI Vision Model)
🚀 Deep Learning Project using PyTorch & Kaggle Dataset

This project uses a Convolutional Neural Network (CNN) trained on the Leaf Manifestation Diseases of Cannabis dataset
 to detect early signs of leaf stress and disease in Cannabis sativa.

The purpose of this model is to contribute toward open-source research in AI-assisted cultivation monitoring — forming part of CCROP’s (Center for Cannabaceae Research in Optimization of Plant-Health) initiative for sustainable plant health optimization.

🧠 Project Overview
🔍 Goal

Train a CNN to classify cannabis leaves into health categories based on visual symptoms, providing a foundation for AI-based early warning systems for cultivators and researchers.

🧩 Features

✅ Auto-downloads dataset directly from Kaggle (no manual uploads)

🖼️ Preprocessing with image augmentation for better generalization

🧠 Transfer learning via pretrained model (ResNet or EfficientNet)

📊 Real-time training visualization and progress output

💾 Model export to .pt for future deployment or fine-tuning

📦 Dataset

Dataset: Leaf Manifestation Diseases of Cannabis

Source: Kaggle
License: Open use for research and education
Classes: Includes various categories of healthy and diseased cannabis leaves.

⚙️ Setup Instructions
1️⃣ Open in Google Colab

You can run this entire project in Google Colab — no local setup required.

2️⃣ Set Up Kaggle API Key

To automatically download the dataset, you’ll need your Kaggle API key.

Go to Kaggle → Account → Create API Token

It will download a file called kaggle.json

Upload it to your Colab environment using the cell:

from google.colab import files
files.upload()  # Select kaggle.json

3️⃣ Run the Notebook

Once uploaded, the notebook will:

Install dependencies

Set up Kaggle API access

Download the dataset

Prepare data loaders

Train and validate the CNN

Save your trained model

🧪 Example Training Output
Epoch [1/15], Loss: 0.5371, Accuracy: 87.4%
Epoch [2/15], Loss: 0.3128, Accuracy: 91.2%
...
Training complete! Model saved to cannabis_leaf_model.pt

🧬 Project Structure
📂 cannabis-leaf-disease-detection/
│
├── 📄 README.md
├── 📓 cannabis_disease_detection.ipynb   # Main Colab notebook
├── 📁 dataset/                           # Auto-downloaded from Kaggle
├── 📁 outputs/
│   └── cannabis_leaf_model.pt            # Saved model
└── 📁 logs/                              # Training logs

🧩 Model Details

Architecture: CNN (ResNet / EfficientNet)

Framework: PyTorch

Optimizer: Adam

Loss Function: CrossEntropyLoss

Metrics: Accuracy, Loss, Validation Accuracy

Image Size: 224x224 (normalized)

🌱 Research Context

This model supports CCROP’s mission to:

Advance AI research in the cultivation of hemp and cannabis, providing open-source datasets and predictive models to improve crop health and sustainability.

The project focuses on early stress detection, disease classification, and AI integration in agricultural monitoring systems.

🤝 Contributing

Pull requests are welcome!
If you’d like to collaborate or test model variants, open an issue or contact CCROP through GitHub.

🧾 License

This project is distributed under the MIT License.
Dataset usage must comply with Kaggle’s terms of service
.

💡 Citation

If you use this work in your research, please cite:

@dataset{engineeringubu_leaf_manifestation_diseases_2023,
  title        = {Leaf Manifestation Diseases of Cannabis},
  author       = {EngineeringUBU},
  year         = {2023},
  url          = {https://www.kaggle.com/datasets/engineeringubu/leaf-manifestation-diseases-of-cannabis}
}
