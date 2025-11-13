import torch
from torch import optim, nn
from torchvision import transforms as T
import matplotlib.pyplot as plt
from ModelCNN import TinyUNet
from PH2_Data_weak import PH2Dataset, make_ph2_loaders
from seg_metrics import dice, iou, accuracy, sensitivity, specificity
from Unet import UNet
from ablation import estimate_pos_weight, run_ablation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinyUNet(in_ch=3, base_ch=32, out_ch=1).to(device)
model2 = UNet(in_ch=3, out_ch=1, base_ch=64, depth=4).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer2 = optim.Adam(model2.parameters(), lr=1e-4)


def train_one_epoch(model, loader, optimizer, criterion, device, threshold=0.5):
    model.train()
    total_loss = 0.0
    M = {"dice": 0.0, "iou": 0.0, "acc": 0.0, "sen": 0.0, "spe": 0.0}
    n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()  # (B,1,H,W) in {0,1}

        optimizer.zero_grad()
        logits = model(xb)  # logits
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # --- metrics on current batch (no grad needed) ---
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            # squeeze channel if needed to (B,H,W)
            yb_ = yb[:, 0] if yb.ndim == 4 and yb.size(1) == 1 else yb
            pr_ = preds[:, 0] if preds.ndim == 4 and preds.size(1) == 1 else preds
            M["dice"] += dice(pr_, yb_).item()
            M["iou"] += iou(pr_, yb_).item()
            M["acc"] += accuracy(pr_, yb_).item()
            M["sen"] += sensitivity(pr_, yb_).item()
            M["spe"] += specificity(pr_, yb_).item()
            n += 1

    avg_loss = total_loss / max(1, len(loader))
    avg_metrics = {k: M[k] / max(1, n) for k in M}
    return avg_loss, avg_metrics


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    M = {"dice": 0.0, "iou": 0.0, "acc": 0.0, "sen": 0.0, "spe": 0.0}
    n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()

        logits = model(xb)  # logits
        total_loss += criterion(logits, yb).item()

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        yb_ = yb[:, 0] if yb.ndim == 4 and yb.size(1) == 1 else yb
        pr_ = preds[:, 0] if preds.ndim == 4 and preds.size(1) == 1 else preds
        M["dice"] += dice(pr_, yb_).item()
        M["iou"] += iou(pr_, yb_).item()
        M["acc"] += accuracy(pr_, yb_).item()
        M["sen"] += sensitivity(pr_, yb_).item()
        M["spe"] += specificity(pr_, yb_).item()
        n += 1

    avg_loss = total_loss / max(1, len(loader))
    avg_metrics = {k: M[k] / max(1, n) for k in M}
    return avg_loss, avg_metrics


@torch.no_grad()
def test_epoch(model, loader, device, threshold=0.5):
    model.eval()
    N = 0
    sums = {"dice": 0.0, "iou": 0.0, "acc": 0.0, "sen": 0.0, "spe": 0.0}

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if yb.ndim == 4 and yb.size(1) == 1:
            yb = yb[:, 0]
        yb = yb.long()

        logits = model(xb)
        if logits.ndim == 4 and logits.size(1) == 1:
            logits = logits[:, 0]
        preds = (torch.sigmoid(logits) >= threshold).long()

        bsz = xb.size(0)
        N += bsz
        sums["dice"] += dice(preds, yb).item() * bsz
        sums["iou"] += iou(preds, yb).item() * bsz
        sums["acc"] += accuracy(preds, yb).item() * bsz
        sums["sen"] += sensitivity(preds, yb).item() * bsz
        sums["spe"] += specificity(preds, yb).item() * bsz

    # dataset-weighted averages
    for k in sums:
        sums[k] /= max(N, 1)
    return sums


import os
import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def save_all_preds(model, loader, device, out_dir="preds", threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    idx_global = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        B = xb.size(0)
        for i in range(B):
            x = xb[i].detach().cpu()
            y = yb[i].detach().cpu()
            pr = preds[i].detach().cpu()

            # squeeze channel for masks
            if y.ndim == 3 and y.size(0) == 1:
                y = y[0]
            if pr.ndim == 3 and pr.size(0) == 1:
                pr = pr[0]

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.title("Image")
            plt.imshow(x.permute(1, 2, 0))
            plt.axis("off")
            plt.subplot(1, 3, 2)
            plt.title("Label")
            plt.imshow(y, cmap="gray")
            plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.title("Prediction")
            plt.imshow(pr, cmap="gray")
            plt.axis("off")
            plt.tight_layout()

            out_path = os.path.join(out_dir, f"sample_{idx_global:05d}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
            idx_global += 1

    print(f"Saved {idx_global} images to: {out_dir}")

@torch.no_grad()
def save_all_preds_point_supervision(model, loader, device, out_dir="preds", threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    idx_global = 0
    for xb, yb, pos_xy, neg_xy in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        B = xb.size(0)
        for i in range(B):
            x = xb[i].detach().cpu()
            y = yb[i].detach().cpu()
            pr = preds[i].detach().cpu()

            # squeeze channel for masks
            if y.ndim == 3 and y.size(0) == 1:
                y = y[0]
            if pr.ndim == 3 and pr.size(0) == 1:
                pr = pr[0]

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.title("Image")
            plt.imshow(x.permute(1, 2, 0))
            plt.axis("off")
            plt.subplot(1, 3, 2)
            plt.title("Label")
            plt.imshow(y, cmap="gray")
            plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.title("Prediction")
            plt.imshow(pr, cmap="gray")
            plt.axis("off")
            plt.tight_layout()

            out_path = os.path.join(out_dir, f"sample_{idx_global:05d}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
            idx_global += 1

    print(f"Saved {idx_global} images to: {out_dir}")  



def point_level_loss(logits, mask, pos_xy, neg_xy):
    bce = nn.BCEWithLogitsLoss(reduction="none")
    losses = []
    total_points = 0

    for b in range(logits.size(0)):
        pos = pos_xy[b]
        neg = neg_xy[b]

        def sample_points(points):
            if not points or points[0] == (-1, -1):
                return torch.tensor([], device=logits.device), torch.tensor([], device=logits.device)
            ys = [y for x, y in points if x >= 0 and y >= 0]
            xs = [x for x, y in points if x >= 0 and y >= 0]
            if len(xs) == 0:
                return torch.tensor([], device=logits.device), torch.tensor([], device=logits.device)
            return logits[b, 0, ys, xs], mask[b, 0, ys, xs]

        pos_logits, pos_gt = sample_points(pos)
        neg_logits, neg_gt = sample_points(neg)

        if pos_logits.numel() > 0:
            losses.append(bce(pos_logits, pos_gt).sum())
            total_points += pos_logits.numel()
        if neg_logits.numel() > 0:
            losses.append(bce(neg_logits, neg_gt).sum())
            total_points += neg_logits.numel()

    if len(losses) > 0:
        total_loss = torch.stack(losses).sum()
        return total_loss / total_points # we average over total points
    else:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

def train_one_epoch_point_supervision(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for xb, yb, pos_xy, neg_xy in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()  # (B,1,H,W) in {0,1}

        optimizer.zero_grad()
        logits = model(xb)  # logits
        loss = point_level_loss(logits, yb, pos_xy, neg_xy)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / max(1, len(loader))
    return avg_loss

@torch.no_grad()
def eval_epoch_point_supervision(model, loader, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    M = {"dice": 0.0, "iou": 0.0, "acc": 0.0, "sen": 0.0, "spe": 0.0}
    n = 0

    for xb, yb, pos_xy, neg_xy in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()

        logits = model(xb)  # logits
        total_loss += point_level_loss(logits, yb, pos_xy, neg_xy)

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        yb_ = yb[:, 0] if yb.ndim == 4 and yb.size(1) == 1 else yb
        pr_ = preds[:, 0] if preds.ndim == 4 and preds.size(1) == 1 else preds
        M["dice"] += dice(pr_, yb_).item()
        M["iou"] += iou(pr_, yb_).item()
        M["acc"] += accuracy(pr_, yb_).item()
        M["sen"] += sensitivity(pr_, yb_).item()
        M["spe"] += specificity(pr_, yb_).item()
        n += 1

    avg_loss = total_loss / max(1, len(loader))
    avg_metrics = {k: M[k] / max(1, n) for k in M}
    return avg_loss, avg_metrics

@torch.no_grad()
def test_epoch_point_supervision(model, loader, device, threshold=0.5):
    model.eval()
    N = 0
    sums = {"dice": 0.0, "iou": 0.0, "acc": 0.0, "sen": 0.0, "spe": 0.0}

    for xb, yb, pos_xy, neg_xy in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if yb.ndim == 4 and yb.size(1) == 1:
            yb = yb[:, 0]
        yb = yb.long()

        logits = model(xb)
        if logits.ndim == 4 and logits.size(1) == 1:
            logits = logits[:, 0]
        preds = (torch.sigmoid(logits) >= threshold).long()

        bsz = xb.size(0)
        N += bsz
        sums["dice"] += dice(preds, yb).item() * bsz
        sums["iou"] += iou(preds, yb).item() * bsz
        sums["acc"] += accuracy(preds, yb).item() * bsz
        sums["sen"] += sensitivity(preds, yb).item() * bsz
        sums["spe"] += specificity(preds, yb).item() * bsz

    # dataset-weighted averages
    for k in sums:
        sums[k] /= max(N, 1)
    return sums


# UNET --------------------

# Transform PH2 to 512x512
img_trans = T.Compose(
    [
        T.Resize((512, 512)),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406],
        #             std =[0.229, 0.224, 0.225])
    ]
)

mask_trans = T.Compose(
    [
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Lambda(lambda m: (m > 0.5).float()),  # binarize {0,1}
    ]
)

# ================== SAMPLING STRATEGY CONFIGURATION ==================
# Available sampling methods for click simulation:
# - "random": Random sampling from valid regions (original method)
# - "grid": Grid-based sampling with auto-calculated spacing based on number of clicks
# - "boundary": Samples points from lesion boundaries/edges using morphological operations
# - "poisson": Poisson disk sampling with minimum distance constraints for better distribution
# 
# Each method has different advantages:
# - boundary: Good for edge-focused supervision, emphasizes lesion boundaries
# - poisson: Avoids clustering, ensures minimum distance between points
# - grid: Systematic coverage, predictable distribution
# - random: Simple baseline, good for general supervision
SAMPLING_METHOD = "poisson"  # Change this to test different sampling methods

# Load UNET

train_loader_PH2, val_loader_PH2, test_loader_PH2 = make_ph2_loaders(
    root_dir="/dtu/datasets1/02516/PH2_Dataset_images",
    batch_size=2,
    img_transform=T.ToTensor(),
    mask_transform=None,
    sampling_method=SAMPLING_METHOD,
    poisson_min_dist=30,  # Only used when sampling_method="poisson"
)

# test clicks and save the picture
train_ds = PH2Dataset(
    root_dir="/dtu/datasets1/02516/PH2_Dataset_images",
    split="train",
    val_ratio=0.2,
    test_ratio=0.2,
    seed=42,
    transform_img=T.ToTensor(),
    transform_mask=mask_trans,
    clicks_pos=10,
    clicks_neg=10,
    sampling_method=SAMPLING_METHOD,
    poisson_min_dist=30  # Only used when sampling_method="poisson"
)

fig, axes = plt.subplots(1, 2)
axes = axes.flatten()
for i in range(1):
    x,y,_,_ = train_ds[i]

    xi = x.detach().cpu()
    yi = y.detach().cpu()
    # prepare image array
    if xi.ndim == 3 and xi.shape[0] == 3:
        img_np = xi.permute(1, 2, 0).clamp(0, 1).numpy()
        img_cmap = None
    elif xi.ndim == 3 and xi.shape[0] == 1:
        img_np = xi[0].clamp(0, 1).numpy()
        img_cmap = "gray"
    else:
        # fallback
        img_np = xi.squeeze().clamp(0, 1).numpy()
        img_cmap = "gray"
    # mask array
    mask_np = yi[0].float().numpy() if yi.ndim == 3 else yi.float().numpy()

    axes[2*i].set_title(f"Image {i+1}")
    axes[2*i].imshow(img_np, cmap=img_cmap)
    axes[2*i].axis("off")
    axes[2*i + 1].set_title(f"Mask {i+1}")
    axes[2*i + 1].imshow(mask_np, cmap="gray")
    axes[2*i + 1].axis("off")

plt.tight_layout()
plt.savefig(f"/zhome/9c/f/221532/Project3/part2/Ablation_results/clicks_{SAMPLING_METHOD}.png", dpi=150)
plt.close()

# click annotations, training loop
print("Started training")
for epoch in range(10):
    tr_loss = train_one_epoch_point_supervision(model2, train_loader_PH2, optimizer2, device)
    va_loss, va_m = eval_epoch_point_supervision(model2, val_loader_PH2, device)
    print(f"{train_ds.clicks_pos} {SAMPLING_METHOD} clicks:"
          f"[PH2][{epoch+1:02d}] "
          f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f} | "
        #   f"train: Dice={tr_m['dice']:.3f} IoU={tr_m['iou']:.3f} Acc={tr_m['acc']:.3f} Sen={tr_m['sen']:.3f} Spec={tr_m['spe']:.3f} | "
          f"val: Dice={va_m['dice']:.3f} IoU={va_m['iou']:.3f} Acc={va_m['acc']:.3f} Sen={va_m['sen']:.3f} Spec={va_m['spe']:.3f}")


metrics = test_epoch_point_supervision(model2, test_loader_PH2, device, threshold=0.5)
print(metrics)

#for visual inspection, save predictions on test set
save_all_preds_point_supervision(model2, test_loader_PH2, device, out_dir="test_ph2_unet", threshold=0.5)

# # Ablation -------------

# new_unet = lambda: UNet(in_ch=3, out_ch=1, base_ch=64, depth=4)

# # PH2
# ph2_results = run_ablation("PH2",(train_loader_PH2, val_loader_PH2, test_loader_PH2),
#     device, new_unet, train_one_epoch, eval_epoch, epochs=10, lr=1e-3)

# # DRIVE
# drive_results = run_ablation("DRIVE", (train_loader_DRIVE, val_loader_DRIVE, test_loader_DRIVE),
#     device, new_unet, train_one_epoch, eval_epoch, epochs=10, lr=1e-3)
