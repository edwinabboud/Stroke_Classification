import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb
import yaml
from dotenv import load_dotenv

from models.efficientnet_transformer import StrokeClassifier
from preprocess.preprocess import StrokeCTDataset, get_transforms

# === Load .env for wandb ===
load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)

# === Load config.yaml ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# === Set device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Set seed ===
torch.manual_seed(42)

# === Define transforms ===
transform = get_transforms(config['data']['image_size'])

# === Load datasets ===
train_dataset = StrokeCTDataset(os.path.join(config['data']['processed_path'], 'train'), transform)
val_dataset   = StrokeCTDataset(os.path.join(config['data']['processed_path'], 'val'), transform)
train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False)

print(f"[DEBUG] Loaded {len(train_dataset)} training samples")
print(f"[DEBUG] Loaded {len(val_dataset)} validation samples")

# === Model ===
model = StrokeClassifier(
    backbone=config['model']['backbone'],
    pretrained=config['model']['pretrained'],
    num_classes=config['model']['num_classes']
).to(device)

# === Optimizer, Loss, Scheduler ===
optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'], weight_decay=config['train']['weight_decay'])
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'])

# === wandb init ===
if config['wandb']['use']:
    wandb.init(
        project=config['wandb']['project'],
        name=config['run_name'],
        config=config
    )

# === Training loop ===
print(f"[DEBUG] Entering training loop")
best_val_acc = 0
for epoch in range(config['train']['epochs']):
    print(f"[DEBUG] Starting epoch {epoch+1}")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    # === Validation ===
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= total
    val_acc = correct / total

    # === Logging ===
    print(f"Epoch {epoch+1}/{config['train']['epochs']} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    if config['wandb']['use']:
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0]
        })

    scheduler.step()

    # === Save best model ===
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/best_model.pt")

if config['wandb']['use']:
    wandb.finish()