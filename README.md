🧠 NeuroScan Pro: Clinical-Grade MRI Diagnostics

NeuroScan Pro is an end-to-end deep learning pipeline designed to assist radiologists in the rapid classification of brain tumors. It utilizes Transfer Learning (EfficientNet-B0) to detect Gliomas, Meningiomas, and Pituitary tumors with >99% accuracy.

Crucially, this system prioritizes Explainability (XAI). It integrates Grad-CAM to generate heatmaps, visually highlighting the specific tissue regions influencing the AI's decision, ensuring that the model is "looking" at the tumor and not background artifacts.

📸 Demo & Explainability

AI Diagnosis with Grad-CAM

The model correctly identifies a Pituitary Tumor (99.8% Confidence) and highlights the sellar region in red.

Performance Matrix

Achieved 0% False Positive Rate on healthy patients (Specificity).

⚡ Key Features

Clinical-Grade Accuracy: 99.13% Test Accuracy on 1,311 unseen samples.

Zero False Positives: The model demonstrated 100% Specificity for the "No Tumor" class in testing.

Explainable AI: Integrated Grad-CAM visualization to ensure trust and transparency.

Modern UI: A dark-mode enabled Streamlit dashboard for real-time inference.

Report Generation: Automatic generation of PDF-style text reports for clinical records.

🛠️ Installation & Setup

Note: The MRI dataset is not included in this repository due to size constraints.

1. Clone the Repository

git clone [https://github.com/YOUR_USERNAME/NeuroScan-AI.git](https://github.com/YOUR_USERNAME/NeuroScan-AI.git)
cd NeuroScan-AI


2. Install Dependencies

pip install -r requirements.txt


3. Download Data

Download the Brain Tumor MRI Dataset from Kaggle:

Link to Dataset (Masoud Nickparvar)

Extract the files so your folder structure looks like this:

data/
└── raw/
    └── MRI_images/
        ├── Training/
        └── Testing/


4. Run the App

streamlit run app.py


🧠 Model Architecture

Component

Specification

Reason for Choice

Backbone

EfficientNet-B0

Compound scaling offers high accuracy with low inference latency (120ms).

Optimizer

Adam (lr=1e-4)

Fast convergence with adaptive learning rates.

Loss Function

CrossEntropy

Standard for multi-class classification.

Augmentation

RandomRotate, Flip

Prevents overfitting and ensures geometric invariance.

📂 Project Structure

├── src/
│   ├── data_loader.py    # Clinical-grade augmentations & loading
│   ├── model.py          # EfficientNet architecture definition
│   └── explain.py        # Grad-CAM engine logic
├── saved_models/         # Trained model weights (best_model_finetuned.pth)
├── app.py                # Streamlit Web Application
├── train.py              # Training loop with validation
├── predict.py            # CLI Diagnostic tool
└── requirements.txt      # Dependency list


📜 Disclaimer

This project is for research and educational purposes only. It is not FDA-approved and should not be used as a substitute for professional medical diagnosis.
