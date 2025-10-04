# bolt_pipeline.py
# Minimal template for counterfeit-bolt detection
# Variants:
#   A) Gallery + Distance (simple baseline)
#   B) PaDiM-style (patchwise, optional upgrade)

import os
import glob
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

# =========================
# 0) CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FINETUNED_WEIGHTS = "resnet50_finetuned_authentic.pth"

# Folders (heads/threads ROIs). Use your own paths or CLI args.
AUTH_HEAD_DIR   = "data/authentic/authentic_head"     # authentic McMaster head crops
AUTH_THREAD_DIR = "data/authentic/authentic_threads"  # authentic McMaster thread crops
TEST_HEAD_DIR   = "data/test/test_head"          # test head crops
TEST_THREAD_DIR = "data/test/test_threads"       # test thread crops

IMG_SIZE = (224, 224)
TOPK = 5                       # nearest neighbors to average
#This is our threshold that we are evaluating currently (need to modify)
#If calculated distance/mean is larger than we classify it as counterfeit

THRESH_HEAD_DIST   = 0.25      # tune on authentic validation (99th percentile)
THRESH_THREAD_DIST = 0.20      # threashold for cos distance
THRESH_HEAD_Z      = 25.0      # optional z-score norm threshold
THRESH_THREAD_Z    = 30.0      # threshold for z score

# PaDiM options (optional)
#If USE_PADIM is false then it will run the simple Gallery + Distance Method
USE_PADIM = False              # turn on if you want patchwise modeling (if we want to run the PaDim anomaly detection method) 
PADIM_D = 100                  # out of the 7x7x2048 feature map that ResNet50 produces, we only extract a 7x7x100 feature map to speed up evaluation
PADIM_REDUCE = "mean"          # creates one final anomaly score by finding the mean for all of the anomaly scores for each patch


# =========================
# 1) MODEL: ResNet-50 Feature Extractor
# =========================
class ResNet50Backbone(nn.Module):
    """ResNet-50 up to (but not including) avgpool+fc."""
    def __init__(self):
        super().__init__()
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.body = nn.Sequential(*list(net.children())[:-2])  # (B, 2048, H, W) this outputs the feature map
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))             # compresses feature map to produce 1x2048 feature vector

    @torch.no_grad()
    def embed_global(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Return 1x2048 embedding (global vector)."""
        self.eval()
        fmap = self.body(img_tensor)           # (B, 2048, H, W)
        pooled = self.avgpool(fmap)            # (B, 2048, 1, 1)
        vec = pooled.view(pooled.size(0), -1)  # (B, 2048)
        v = vec[0].cpu().numpy()
        v = v / (np.linalg.norm(v) + 1e-8)     # L2-normalize
        return v

    @torch.no_grad()
    def feature_map(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Return last conv feature map (B, 2048, H, W) for PaDiM."""
        self.eval()
        fmap = self.body(img_tensor)
        return fmap


# =========================
# 2) PREPROCESS
# =========================
TFM = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return TFM(img).unsqueeze(0).to(DEVICE)   # (1,3,H,W)


# =========================
# 3) GALLERY + DISTANCE UTILITIES (Simple Baseline)
# =========================
def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - (a @ b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8)

def topk_avg_distance(vec: np.ndarray, gallery: np.ndarray, k: int = TOPK) -> float:
    dists = np.array([cosine_dist(vec, g) for g in gallery])
    return np.sort(dists)[:k].mean()

def zscore_norm(vec: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> float:
    z = (vec - mu) / (sd + 1e-6)
    return float(np.linalg.norm(z))

def build_gallery(backbone: ResNet50Backbone, folder: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (gallery Nx2048, mean 2048, std 2048) for a folder of authentic ROI images."""
    vecs = []
    paths = sorted(glob.glob(os.path.join(folder, "*.*")))
    assert len(paths) > 0, f"No images in {folder}"
    for p in paths:
        x = load_image(p)
        v = backbone.embed_global(x)
        vecs.append(v)
    gallery = np.stack(vecs)         # (N,2048)
    mu = gallery.mean(axis=0)
    sd = gallery.std(axis=0) + 1e-6
    return gallery, mu, sd


# =========================
# 4) PaDiM-LIKE (Patchwise) UTILITIES (Optional)
# =========================
def select_channels(C: int = 2048, D: int = PADIM_D, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(C, size=D, replace=False))
    return idx

def fmap_to_patches(fmap: torch.Tensor, channel_idx: np.ndarray) -> np.ndarray:
    """
    fmap: (1, C, H, W)
    return: (H*W, D) numpy
    """
    fmap = fmap[0, channel_idx, :, :]            # (D,H,W)
    D, H, W = fmap.shape
    patches = fmap.permute(1,2,0).contiguous().view(H*W, D)
    return patches.cpu().numpy(), H, W

def fit_padim(backbone: ResNet50Backbone, folder: str, channel_idx: np.ndarray):
    """
    Fit per-patch Gaussian (mu, cov_inv) from authentic images in folder.
    Returns: mu_list, invcov_list, H, W
    """
    patch_bags = []
    paths = sorted(glob.glob(os.path.join(folder, "*.*")))
    assert len(paths) > 0, f"No images in {folder}"
    H = W = None

    for p in paths:
        x = load_image(p)
        fmap = backbone.feature_map(x)  # (1,2048,H,W)
        patches, H, W = fmap_to_patches(fmap, channel_idx)  # (H*W, D)
        patch_bags.append(patches)

    stack = np.stack(patch_bags, axis=0)  # (Nimg, H*W, D)
    Nimg, Npatch, D = stack.shape

    mu_list, invcov_list = [], []
    # Using Ledoit-Wolf covariance is better, but to keep the template
    # dependency-light we’ll do a pinv of sample covariance.
    for p in range(Npatch):
        X = stack[:, p, :]                         # (Nimg, D)
        mu = X.mean(axis=0)
        X0 = X - mu
        cov = (X0.T @ X0) / max(Nimg-1, 1)         # (D,D)
        invcov = np.linalg.pinv(cov)               # pseudo-inverse for stability
        mu_list.append(mu)
        invcov_list.append(invcov)

    return mu_list, invcov_list, H, W

def score_padim(backbone: ResNet50Backbone, img_path: str,
                channel_idx: np.ndarray, mu_list, invcov_list, H, W,
                reduce: str = PADIM_REDUCE) -> Tuple[np.ndarray, float]:
    x = load_image(img_path)
    fmap = backbone.feature_map(x)                 # (1,2048,H,W)
    patches, H2, W2 = fmap_to_patches(fmap, channel_idx)
    assert (H2, W2) == (H, W)

    dists = []
    for p in range(H*W):
        xv = patches[p]
        mu = mu_list[p]
        invc = invcov_list[p]
        diff = (xv - mu)
        md2 = float(diff @ invc @ diff)
        dists.append(np.sqrt(max(md2, 0.0)))

    amap = np.array(dists).reshape(H, W)

    if reduce == "mean":
        score = float(amap.mean())
    elif reduce == "max":
        score = float(amap.max())
    else:  # 'p90'
        score = float(np.percentile(amap, 90))
    return amap, score


# =========================
# 5) MAIN FLOW (pick your variant)
# =========================
def run_gallery_distance():
    print("=== A) Gallery + Distance ===")

    # Build a base torchvision ResNet-50 first
    base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Load the fine-tuned weights if available INTO THE BASE MODEL
    if os.path.exists(FINETUNED_WEIGHTS):
        state = torch.load(FINETUNED_WEIGHTS, map_location=DEVICE)
        # Load whatever matches (layer4/conv1/bn1 etc.); skip non-matching keys
        incompatible = base.load_state_dict(state, strict=False)
        print("Loaded fine-tuned weights.")
        print("  Missing keys:", incompatible.missing_keys)
        print("  Unexpected  :", incompatible.unexpected_keys)
    else:
        print("Fine-tuned weights not found; using ImageNet weights.")

    # Now build your backbone FROM the (possibly fine-tuned) base model
    backbone = ResNet50Backbone().to(DEVICE)
    backbone.body = nn.Sequential(*list(base.children())[:-2]).to(DEVICE)

    # Build authentic galleries
    print("Building galleries...")
    gal_head, mu_head, sd_head = build_gallery(backbone, AUTH_HEAD_DIR)
    gal_thr,  mu_thr,  sd_thr  = build_gallery(backbone, AUTH_THREAD_DIR)

    # Score all test images (example)
    head_tests = sorted(glob.glob(os.path.join(TEST_HEAD_DIR, "*.*")))
    thr_tests  = sorted(glob.glob(os.path.join(TEST_THREAD_DIR, "*.*")))

    def embed_path(p): return backbone.embed_global(load_image(p))

    print("\nScoring HEAD ROIs:")
    for p in head_tests:
        v = embed_path(p)
        dist = topk_avg_distance(v, gal_head, k=TOPK)
        z    = zscore_norm(v, mu_head, sd_head)
        verdict = "PASS" if (dist <= THRESH_HEAD_DIST and z <= THRESH_HEAD_Z) else "FLAG"
        print(f"{os.path.basename(p):30s}  dist={dist:.3f}  z={z:.1f}  -> {verdict}")

    print("\nScoring THREAD ROIs:")
    for p in thr_tests:
        v = embed_path(p)
        dist = topk_avg_distance(v, gal_thr, k=TOPK)
        z    = zscore_norm(v, mu_thr, sd_thr)
        verdict = "PASS" if (dist <= THRESH_THREAD_DIST and z <= THRESH_THREAD_Z) else "FLAG"
        print(f"{os.path.basename(p):30s}  dist={dist:.3f}  z={z:.1f}  -> {verdict}")

    print("\nFuse decision in your app: FLAG if threads FLAG; else check head.")

def run_padim():
    print("=== B) PaDiM-style (patchwise) ===")
    assert USE_PADIM, "Set USE_PADIM=True at top to run this block."
    backbone = ResNet50Backbone().to(DEVICE)
    ch_idx = select_channels(C=2048, D=PADIM_D)

    print("Fitting PaDiM on authentic HEAD patches...")
    muH, invH, Hh, Wh = fit_padim(backbone, AUTH_HEAD_DIR, ch_idx)
    print("Fitting PaDiM on authentic THREAD patches...")
    muT, invT, Ht, Wt = fit_padim(backbone, AUTH_THREAD_DIR, ch_idx)

    head_tests = sorted(glob.glob(os.path.join(TEST_HEAD_DIR, "*.*")))
    thr_tests  = sorted(glob.glob(os.path.join(TEST_THREAD_DIR, "*.*")))

    # Choose thresholds from validation authentic set in practice.
    THR_PADIM_HEAD   = 0.80
    THR_PADIM_THREAD = 0.80

    print("\nScoring HEAD ROIs:")
    for p in head_tests:
        amap, score = score_padim(backbone, p, ch_idx, muH, invH, Hh, Wh, PADIM_REDUCE)
        verdict = "PASS" if score <= THR_PADIM_HEAD else "FLAG"
        print(f"{os.path.basename(p):30s}  padim_score={score:.3f} -> {verdict}")

    print("\nScoring THREAD ROIs:")
    for p in thr_tests:
        amap, score = score_padim(backbone, p, ch_idx, muT, invT, Ht, Wt, PADIM_REDUCE)
        verdict = "PASS" if score <= THR_PADIM_THREAD else "FLAG"
        print(f"{os.path.basename(p):30s}  padim_score={score:.3f} -> {verdict}")

if __name__ == "__main__":
    if USE_PADIM:
        run_padim()
    else:
        run_gallery_distance()