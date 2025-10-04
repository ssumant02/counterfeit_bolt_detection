# fine_tune_ResNet_authentic.py

import os, glob, math, random
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Folders: use both head + thread crops of authentic bolts
AUTH_DIRS = [
    "data/authentic/authentic_head",
    "data/authentic/authentic_threads"
]

SAVE_PATH = "models/resnet50_finetuned_authentic.pth"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
WD = 1e-5
IMG_SIZE = (224, 224)
SEED = 123

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------- Dataset that returns TWO augmented views of the SAME image -------- #
class TwoViewAuthenticSet(Dataset):
    def __init__(self, roots):
        self.paths = []
        for r in roots:
            self.paths += sorted(
                [p for p in glob.glob(os.path.join(r, "*.*")) if p.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
            )
        assert len(self.paths) > 0, f"No authentic images found in {roots}"

        # Two independent augmentation pipelines (mild but effective)
        self.tfm1 = T.Compose([
            T.Resize(IMG_SIZE),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.RandomRotation(degrees=5),
            T.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.tfm2 = T.Compose([
            T.Resize(IMG_SIZE),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=5),
            T.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.tfm1(img), self.tfm2(img)

# -------- Backbone that outputs 2048-D global embedding -------- #
class ResNet50Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Unfreeze what you want to fine-tune
        for p in self.net.parameters():
            p.requires_grad = False
        for p in self.net.layer4.parameters():
            p.requires_grad = True
        for p in self.net.conv1.parameters():
            p.requires_grad = True
        for p in self.net.bn1.parameters():
            p.requires_grad = True

        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        # Forward through the body (conv1..layer4) manually:
        x = self.net.conv1(x); x = self.net.bn1(x); x = self.net.relu(x); x = self.net.maxpool(x)
        x = self.net.layer1(x); x = self.net.layer2(x); x = self.net.layer3(x); x = self.net.layer4(x)
        g = self.pool(x).flatten(1)
        g = nn.functional.normalize(g, dim=1)
        return g

def cosine_align_loss(z1, z2):
    # Positive-pair loss: maximize cosine similarity -> minimize (1 - cos)
    cos_sim = nn.functional.cosine_similarity(z1, z2, dim=1)  # (B,)
    loss = (1.0 - cos_sim).mean()
    return loss

def main():
    ds = TwoViewAuthenticSet(AUTH_DIRS)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)

    model = ResNet50Embedder().to(DEVICE)
    # Only optimize the unfrozen params
    params = [p for n,p in model.net.named_parameters() if p.requires_grad]
    optimiz = optim.AdamW(params, lr=LR, weight_decay=WD)

    model.train()
    best_loss = math.inf

    for epoch in range(1, EPOCHS+1):
        run_loss = 0.0
        for (x1, x2) in dl:
            x1 = x1.to(DEVICE, non_blocking=True)
            x2 = x2.to(DEVICE, non_blocking=True)

            z1 = model(x1)   # (B,2048) normalized
            z2 = model(x2)   # (B,2048) normalized

            loss = cosine_align_loss(z1, z2)

            optimiz.zero_grad(set_to_none=True)
            loss.backward()
            optimiz.step()

            run_loss += loss.item()

        avg = run_loss / len(dl)
        print(f"Epoch {epoch:02d}/{EPOCHS}  loss={avg:.4f}")

        # Save best
        if avg < best_loss:
            best_loss = avg
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            # Save torchvision ResNet-50 compatible weights
            torch.save(model.stem.state_dict(), SAVE_PATH)
            print(f"  ↳ saved: {SAVE_PATH}")

    print("Done.")

if __name__ == "__main__":
    torch.set_grad_enabled(True)
    main()