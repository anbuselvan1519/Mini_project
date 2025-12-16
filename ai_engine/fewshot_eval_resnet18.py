import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
import os, random

# ======================
# 1. Config
# ======================
DATA_DIR = r"C:\Users\admin\OneDrive\Documents\projects\plant_cancer_ai\data\dataset"
MODEL_PATH = r"C:\Users\admin\OneDrive\Documents\projects\plant_cancer_ai\models\plant_resnet18.pth"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

val_data = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# 2. Few-Shot Wrapper
# ======================
class FewShotResNet(nn.Module):
    def __init__(self, backbone, feature_dim):
        super(FewShotResNet, self).__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim

    def forward(self, x):
        feats = self.backbone(x).view(x.size(0), -1)
        return feats

    def compute_prototypes(self, support_feats, support_labels):
        prototypes = []
        for c in torch.unique(support_labels):
            class_mask = support_labels == c
            class_embeds = support_feats[class_mask]
            prototypes.append(class_embeds.mean(0))
        return torch.stack(prototypes)

    def fewshot_predict(self, query_feats, prototypes):
        dists = torch.cdist(query_feats, prototypes)
        return torch.argmin(dists, dim=1)

# ======================
# 3. Load Pretrained ResNet18 Backbone
# ======================
base = models.resnet18(pretrained=False)
feature_dim = base.fc.in_features
base.fc = nn.Linear(feature_dim, len(val_data.classes))  # original classifier

# load your trained weights
base.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# drop the classifier, keep backbone
backbone = nn.Sequential(*list(base.children())[:-1])
model = FewShotResNet(backbone, feature_dim).to(device)
model.eval()

# ======================
# 4. Few-Shot Episode Sampler
# ======================
def sample_episode(dataset, n_way=3, k_shot=2, q_query=2):
    classes = random.sample(dataset.classes, n_way)
    support_imgs, support_labels, query_imgs, query_labels = [], [], [], []

    for i, cls in enumerate(classes):
        cls_idx = dataset.class_to_idx[cls]
        cls_samples = [x[0] for x in dataset.samples if x[1] == cls_idx]
        chosen = random.sample(cls_samples, k_shot + q_query)

        for p in chosen[:k_shot]:
            img = transform(datasets.folder.default_loader(p))
            support_imgs.append(img)
            support_labels.append(i)

        for p in chosen[k_shot:]:
            img = transform(datasets.folder.default_loader(p))
            query_imgs.append(img)
            query_labels.append(i)

    return (torch.stack(support_imgs), torch.tensor(support_labels),
            torch.stack(query_imgs), torch.tensor(query_labels))

# ======================
# 5. Run Few-Shot Evaluation
# ======================
support_imgs, support_labels, query_imgs, query_labels = sample_episode(val_data, n_way=3, k_shot=3, q_query=5)
support_imgs, support_labels = support_imgs.to(device), support_labels.to(device)
query_imgs, query_labels = query_imgs.to(device), query_labels.to(device)

with torch.no_grad():
    support_feats = model(support_imgs)
    query_feats = model(query_imgs)
    prototypes = model.compute_prototypes(support_feats, support_labels)
    preds = model.fewshot_predict(query_feats, prototypes)

acc = (preds == query_labels).float().mean().item() * 100
print(f"🌱 Few-Shot Episodic Accuracy: {acc:.2f}%")
