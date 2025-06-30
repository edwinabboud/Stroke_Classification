import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Set random seed for reproducibility
SEED = 42

def preprocess_dataset(src_root, dst_root, val_ratio=0.1, test_ratio=0.1):
    """
    Organizes the dataset into train/val/test directories with binary labels:
    - Normal = 0 (No Stroke)
    - Bleeding or Ischemia = 1 (Stroke)
    """
    class_map = {
        "Normal": 0,         # No Stroke
        "Bleeding": 1,       # Stroke
        "Ischemia": 1        # Stroke
    }

    for cls, label in class_map.items():
        img_dir = os.path.join(src_root, cls, "PNG")
        if not os.path.exists(img_dir):
            continue
        images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")]

        if len(images) == 0:
            continue

        train_imgs, temp_imgs = train_test_split(images, test_size=val_ratio + test_ratio, random_state=SEED)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_ratio / (val_ratio + test_ratio), random_state=SEED)

        for split_name, split_imgs in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
            class_dir = "Stroke" if label == 1 else "NoStroke"
            split_class_dir = os.path.join(dst_root, split_name, class_dir)
            os.makedirs(split_class_dir, exist_ok=True)

            for img_path in split_imgs:
                shutil.copy(img_path, os.path.join(split_class_dir, os.path.basename(img_path)))


class StrokeCTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = list(self.root_dir.glob("*/*.png"))
        self.class_to_idx = {"NoStroke": 0, "Stroke": 1}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.class_to_idx[img_path.parent.name]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])


if __name__ == "__main__":
    preprocess_dataset(
        src_root="./Data",
        dst_root="./Data/processed",
        val_ratio=0.1,
        test_ratio=0.1
    )

    # Optional: Save model for deployment
    import torch
    from models.efficientnet_transformer import StrokeClassifier
    import yaml

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = StrokeClassifier(
        backbone=config['model']['backbone'],
        pretrained=False,
        num_classes=config['model']['num_classes']
    )
    checkpoint_path = "checkpoints/best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
    model.eval()

    # Save to TorchScript
    dummy_input = torch.randn(1, 3, config['data']['image_size'], config['data']['image_size'])
    traced = torch.jit.trace(model, dummy_input)
    traced.save("checkpoints/stroke_model_scripted.pt")

    # Save to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "checkpoints/stroke_model.onnx",
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )
