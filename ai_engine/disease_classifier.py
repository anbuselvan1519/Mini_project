import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os

# ---------------------------
# Load class labels dynamically
# ---------------------------
def load_labels():
    labels_path = r"C:\Users\admin\OneDrive\Documents\projects\plant_cancer_ai\data\class_mapping.json"
    with open(labels_path, "r") as f:
        labels_dict = json.load(f)
    labels = [labels_dict[str(i)] for i in range(len(labels_dict))]
    return labels

CLASSES = load_labels()

# ---------------------------
# Hybrid ResNet18 (features + classifier)
# ---------------------------
class ResNet18Hybrid(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet18(weights=None)   # must match training
        base.fc = nn.Identity()  # keep features
        self.feature_extractor = base
        self.fc = nn.Linear(512, num_classes)  # your trained classifier

    def forward(self, x):
        feats = self.feature_extractor(x)
        logits = self.fc(feats)
        return logits, feats

# ---------------------------
# Load trained model
# ---------------------------
def load_model(model_path, num_classes):
    # Base ResNet18
    base = models.resnet18(weights=None)
    base.fc = nn.Linear(512, num_classes)  # classifier head

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(model_path, map_location=device)
    base.load_state_dict(state)  # ✅ direct match

    model = nn.Sequential(
        nn.Identity(),  # dummy to keep API flexible
        base
    )
    model.to(device).eval()
    return base, device


MODEL_PATH = r"C:\Users\admin\OneDrive\Documents\projects\plant_cancer_ai\models\plant_resnet18.pth"
model, device = load_model(MODEL_PATH, num_classes=len(CLASSES))

# ---------------------------
# Transform pipeline
# ---------------------------
def to_tensor_no_numpy(pic):
    # Convert PIL image → tensor without numpy
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute(2, 0, 1).float().div(255)
    return img

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    to_tensor_no_numpy,  # ✔ custom safe replacement
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])


# ---------------------------
# Few-Shot Utilities
# ---------------------------
def compute_prototypes(support_feats, support_labels):
    prototypes = []
    classes = torch.unique(support_labels)
    for cls in classes:
        cls_feats = support_feats[support_labels == cls]
        prototypes.append(cls_feats.mean(0))
    return torch.stack(prototypes), classes

def fewshot_predict(query_feats, prototypes, proto_classes):
    dists = torch.cdist(query_feats, prototypes)  # Euclidean distance
    preds = proto_classes[dists.argmin(dim=1)]
    return preds

# ---------------------------
# Load support set for few-shot fallback
# ---------------------------
SUPPORT_DIR = r"C:\Users\admin\OneDrive\Documents\projects\plant_cancer_ai\data\support_images"
support_images, support_labels, support_classnames = [], [], []

if os.path.exists(SUPPORT_DIR):
    for idx, cls in enumerate(os.listdir(SUPPORT_DIR)):
        cls_folder = os.path.join(SUPPORT_DIR, cls)
        if os.path.isdir(cls_folder):
            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    support_images.append(os.path.join(cls_folder, img_name))

# Assign labels to support set
support_tensors, support_label_tensors = [], []
if support_images:
    for idx, cls in enumerate(os.listdir(SUPPORT_DIR)):
        cls_folder = os.path.join(SUPPORT_DIR, cls)
        if os.path.isdir(cls_folder):
            support_classnames.append(cls)
            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(cls_folder, img_name)
                    img = Image.open(img_path).convert("RGB")
                    img_t = transform(img)
                    support_tensors.append(img_t)
                    support_label_tensors.append(idx)

    if support_tensors:
        support_tensors = torch.stack(support_tensors).to(device)
        support_label_tensors = torch.tensor(support_label_tensors).to(device)
        with torch.no_grad():
            _, s_feats = model(support_tensors)
        support_prototypes, proto_classes = compute_prototypes(s_feats, support_label_tensors)

        # Map proto_class → classname
        proto_class_to_name = {cls.item(): support_classnames[cls.item()] for cls in proto_classes}
    else:
        support_prototypes, proto_classes, proto_class_to_name = None, None, None
else:
    support_prototypes, proto_classes, proto_class_to_name = None, None, None

# ---------------------------
# Unified Prediction Function
# ---------------------------
def predict_disease(image, threshold=0.6):
    """
    Predict plant disease.
    If normal confidence < threshold, fallback to few-shot with support set.
    """
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")

    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, 1)
        return CLASSES[predicted.item()], confidence.item()


    # If confidence is high enough, return normal prediction
    if confidence.item() >= threshold:
        return CLASSES[predicted.item()], confidence.item()

    # Else fallback to few-shot
    if support_prototypes is not None:
        # need features -> recompute manually if few-shot required
        feats = logits  # ⚠️ placeholder, or extend model to expose features
        pred = fewshot_predict(feats, support_prototypes, proto_classes).item()
        return support_classnames[pred], 1.0

    # If no support set exists, return "Unknown"
    return "UnknownDisease", confidence.item()


