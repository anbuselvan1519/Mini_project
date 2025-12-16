import torch
from torchvision import transforms
from PIL import Image

# Define preprocessing pipeline for input images
def preprocess_image(image_path, img_size=128):
    """
    Preprocess an input image for CNN inference.
    
    Args:
        image_path (str): Path to input image
        img_size (int): Image resizing dimension
    
    Returns:
        torch.Tensor: Preprocessed tensor ready for CNN
    """
    transform_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])  # Standard ImageNet values
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform_pipeline(image).unsqueeze(0)  # Add batch dimension
    return image
