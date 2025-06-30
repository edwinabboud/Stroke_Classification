import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from PIL import Image
from torchvision import transforms
from models.efficientnet_transformer import StrokeClassifier
from preprocess.preprocess import get_transforms

# === Load config ===
import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Custom Dataset for External_Test (Binary) ===
class ExternalTestDataset(Dataset):
    def __init__(self, image_dir, label_csv, transform=None):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(label_csv)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_filename = f"{row['image_id']}.png"
        img_path = os.path.join(self.image_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        label = 1 if row['Stroke'] != 0 else 0  # Stroke or No Stroke
        if self.transform:
            image = self.transform(image)
        return image, label

# === Load dataset ===
transform = get_transforms(config['data']['image_size'])
dataset = ExternalTestDataset(
    image_dir=os.path.join(config['data']['dataset_path'], 'External_Test/PNG'),
    label_csv=os.path.join(config['data']['dataset_path'], 'External_Test/labels.csv'),
    transform=transform
)
loader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=False)

# === Load model ===
model = StrokeClassifier(
    backbone=config['model']['backbone'],
    pretrained=False,
    num_classes=config['model']['num_classes']
)
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
model.to(device)
model.eval()

# === Evaluation ===
y_true = []
y_pred = []

with torch.no_grad():
    for imgs, labels in tqdm(loader, desc="Evaluating External_Test"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# === Metrics ===
print("\nClassification Report (External_Test — Binary):\n")
print(classification_report(y_true, y_pred, target_names=['No Stroke', 'Stroke']))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix — External Test (Binary)")
plt.tight_layout()
plt.show()
