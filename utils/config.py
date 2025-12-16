import os

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Models
CNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_model.pth")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer")

# Image processing
IMAGE_SIZE = 128

# Device configuration
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels (example: modify based on your dataset)
DISEASE_CLASSES = [
    "Healthy",
    "Leaf Cancer - Early Stage",
    "Leaf Cancer - Moderate",
    "Leaf Cancer - Severe"
]

# Suggestions for each class (offline knowledge base)
PREVENTION_GUIDE = {
    "Healthy": "Your plant is healthy. Keep monitoring regularly.",
    "Leaf Cancer - Early Stage": "Apply organic fungicides, prune affected leaves.",
    "Leaf Cancer - Moderate": "Use copper-based fungicide, improve air circulation.",
    "Leaf Cancer - Severe": "Remove infected plants, disinfect tools, rotate crops."
}
