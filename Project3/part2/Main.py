import torch
from torch import optim, nn
from torchvision import transforms as T
import matplotlib.pyplot as plt
from ModelCNN import TinyUNet
from Project3.part2.PH2_Data_weak import PH2Dataset, make_ph2_loaders
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
    M = {"dice":0.0, "iou":0.0, "acc":0.0, "sen":0.0, "spe":0.0}
    n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()      # (B,1,H,W) in {0,1}

        optimizer.zero_grad()
        logits = model(xb)              # logits
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # --- metrics on current batch (no grad needed) ---
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            # squeeze channel if needed to (B,H,W)
            yb_ = yb[:,0] if yb.ndim==4 and yb.size(1)==1 else yb
            pr_ = preds[:,0] if preds.ndim==4 and preds.size(1)==1 else preds
            M["dice"] += dice(pr_, yb_).item()
            M["iou"]  += iou(pr_,  yb_).item()
            M["acc"]  += accuracy(pr_, yb_).item()
            M["sen"]  += sensitivity(pr_, yb_).item()
            M["spe"]  += specificity(pr_, yb_).item()
            n += 1

    avg_loss = total_loss / max(1, len(loader))
    avg_metrics = {k: M[k] / max(1, n) for k in M}
    return avg_loss, avg_metrics

@torch.no_grad()
def eval_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    M = {"dice":0.0, "iou":0.0, "acc":0.0, "sen":0.0, "spe":0.0}
    n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()

        logits = model(xb)              # logits
        total_loss += criterion(logits, yb).item()

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        yb_ = yb[:,0] if yb.ndim==4 and yb.size(1)==1 else yb
        pr_ = preds[:,0] if preds.ndim==4 and preds.size(1)==1 else preds
        M["dice"] += dice(pr_, yb_).item()
        M["iou"]  += iou(pr_,  yb_).item()
        M["acc"]  += accuracy(pr_, yb_).item()
        M["sen"]  += sensitivity(pr_, yb_).item()
        M["spe"]  += specificity(pr_, yb_).item()
        n += 1

    avg_loss = total_loss / max(1, len(loader))
    avg_metrics = {k: M[k] / max(1, n) for k in M}
    return avg_loss, avg_metrics


@torch.no_grad()
def test_epoch(model, loader, device, threshold=0.5):
    model.eval()
    N = 0
    sums = {"dice":0.0, "iou":0.0, "acc":0.0, "sen":0.0, "spe":0.0}

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if yb.ndim == 4 and yb.size(1) == 1: 
            yb = yb[:,0]
        yb = yb.long()

        logits = model(xb)   
        if logits.ndim == 4 and logits.size(1) == 1:
            logits = logits[:,0]             
        preds = (torch.sigmoid(logits) >= threshold).long()

        bsz = xb.size(0)
        N += bsz
        sums["dice"] += dice(preds, yb).item() * bsz
        sums["iou"]  += iou(preds, yb).item()  * bsz
        sums["acc"]  += accuracy(preds, yb).item() * bsz
        sums["sen"]  += sensitivity(preds, yb).item() * bsz
        sums["spe"]  += specificity(preds, yb).item() * bsz

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
        probs  = torch.sigmoid(logits)          
        preds  = (probs >= threshold).float()  

        B = xb.size(0)
        for i in range(B):
            x  = xb[i].detach().cpu()        
            y  = yb[i].detach().cpu()        
            pr = preds[i].detach().cpu()     

            # squeeze channel for masks
            if y.ndim == 3 and y.size(0) == 1: y = y[0]
            if pr.ndim == 3 and pr.size(0) == 1: pr = pr[0]

            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1); plt.title("Image"); plt.imshow(x.permute(1,2,0)); plt.axis("off")
            plt.subplot(1,3,2); plt.title("Label"); plt.imshow(y, cmap="gray"); plt.axis("off")
            plt.subplot(1,3,3); plt.title("Prediction"); plt.imshow(pr, cmap="gray"); plt.axis("off")
            plt.tight_layout()

            out_path = os.path.join(out_dir, f"sample_{idx_global:05d}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
            idx_global += 1

    print(f"Saved {idx_global} images to: {out_dir}")



# Example usage with your loaders (Drive or PH2):

# train_loader_PH2, val_loader_PH2,test_loader_PH2 = make_ph2_loaders(
#     root_dir="/dtu/datasets1/02516/PH2_Dataset_images",
#     batch_size=2,
#     img_transform=T.ToTensor(),
#     mask_transform=None
# )

# train_loader_DRIVE, val_loader_DRIVE,test_loader_DRIVE = make_drive_loaders(
#     root_dir="/dtu/datasets1/02516/DRIVE",
#     batch_size=2,
#     img_transform=T.ToTensor(),
#     mask_transform=None
# )


# for epoch in range(10):
#     tr_loss, tr_m = train_one_epoch(model, train_loader_PH2, optimizer, criterion, device)
#     va_loss, va_m = eval_epoch(model, val_loader_PH2, criterion, device)
#     print(f"[PH2][{epoch+1:02d}] "
#           f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")



# metrics = test_epoch(model, test_loader_PH2, device, threshold=0.5)
# print(metrics) 

# #for visual inspection, save predictions on test set
# save_all_preds(model, test_loader_PH2, device, out_dir="test_PH2_CNN", threshold=0.5)



# UNET --------------------

# Transform PH2 to 512x512
img_trans = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406],
    #             std =[0.229, 0.224, 0.225])
])

mask_trans = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Lambda(lambda m: (m > 0.5).float()),  # binarize {0,1}
])


# Load UNET

train_loader_PH2, val_loader_PH2,test_loader_PH2 = make_ph2_loaders(
    root_dir="/dtu/datasets1/02516/PH2_Dataset_images",
    batch_size=2,
    img_transform=T.ToTensor(),
    mask_transform=None
)

# # DRIVE is already cropped to 512x512
# train_loader_DRIVE, val_loader_DRIVE,test_loader_DRIVE = make_drive_loaders(
#     root_dir="/dtu/datasets1/02516/DRIVE",
#     batch_size=2,
#     img_transform=img_trans,
#     mask_transform=mask_trans
# )


# # UNET PH2

for epoch in range(10):
    tr_loss, tr_m = train_one_epoch(model2, train_loader_PH2, optimizer2, criterion, device)
    va_loss, va_m = eval_epoch(model2, val_loader_PH2, criterion, device)
    print(f"[PH2][{epoch+1:02d}] "
          f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f} | "
          f"train: Dice={tr_m['dice']:.3f} IoU={tr_m['iou']:.3f} Acc={tr_m['acc']:.3f} Sen={tr_m['sen']:.3f} Spec={tr_m['spe']:.3f} | "
          f"val: Dice={va_m['dice']:.3f} IoU={va_m['iou']:.3f} Acc={va_m['acc']:.3f} Sen={va_m['sen']:.3f} Spec={va_m['spe']:.3f}")


metrics = test_epoch(model2, test_loader_PH2, device, threshold=0.5)
print(metrics) 

#for visual inspection, save predictions on test set
save_all_preds(model2, test_loader_PH2, device, out_dir="test_ph2_unet", threshold=0.5)

# # UNET DRIVE

# for epoch in range(10):
#     tr_loss, tr_m = train_one_epoch(model2, train_loader_DRIVE, optimizer2, criterion, device)
#     va_loss, va_m = eval_epoch(model2, val_loader_DRIVE, criterion, device)
#     print(f"[DRIVE][{epoch+1:02d}] "
#           f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f} | "
#           f"train: Dice={tr_m['dice']:.3f} IoU={tr_m['iou']:.3f} Acc={tr_m['acc']:.3f} Sen={tr_m['sen']:.3f} Spec={tr_m['spe']:.3f} | "
#           f"val: Dice={va_m['dice']:.3f} IoU={va_m['iou']:.3f} Acc={va_m['acc']:.3f} Sen={va_m['sen']:.3f} Spec={va_m['spe']:.3f}")


# metrics = test_epoch(model2, test_loader_DRIVE, device, threshold=0.5)
# print(metrics) 

# #for visual inspection, save predictions on test set
# save_all_preds(model2, test_loader_DRIVE, device, out_dir="test_drive_unet", threshold=0.5)


# # Ablation -------------

# new_unet = lambda: UNet(in_ch=3, out_ch=1, base_ch=64, depth=4)

# # PH2
# ph2_results = run_ablation("PH2",(train_loader_PH2, val_loader_PH2, test_loader_PH2),
#     device, new_unet, train_one_epoch, eval_epoch, epochs=10, lr=1e-3)

# # DRIVE
# drive_results = run_ablation("DRIVE", (train_loader_DRIVE, val_loader_DRIVE, test_loader_DRIVE),
#     device, new_unet, train_one_epoch, eval_epoch, epochs=10, lr=1e-3)
