import torch
from torch import optim, nn
from torchvision import transforms as T
import matplotlib.pyplot as plt
from ModelCNN import TinyUNet
from DRIVE_Data import DriveDataset, make_drive_loaders
from PH2_Data import PH2Dataset, make_ph2_loaders
from seg_metrics import dice, iou, accuracy, sensitivity, specificity


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinyUNet(in_ch=3, base_ch=32, out_ch=1).to(device)
criterion = nn.BCEWithLogitsLoss()                        
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()                      
        optimizer.zero_grad()
        logits = model(xb)                                
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()
        logits = model(xb)
        loss = criterion(logits, yb)
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)

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
train_loader_PH2, val_loader_PH2,test_loader_PH2 = make_ph2_loaders(
    root_dir="/dtu/datasets1/02516/PH2_Dataset_images",
    batch_size=2,
    img_transform=T.ToTensor(),
    mask_transform=None
)

train_loader_DRIVE, val_loader_DRIVE,test_loader_DRIVE = make_drive_loaders(
    root_dir="/dtu/datasets1/02516/DRIVE",
    batch_size=2,
    img_transform=T.ToTensor(),
    mask_transform=None
)

for epoch in range(10):
    tr = train_one_epoch(model, train_loader_PH2, optimizer, criterion)
    va = eval_epoch(model, val_loader_PH2, criterion)
    print(f"[{epoch+1:02d}] train_loss={tr:.4f}  val_loss={va:.4f}")



metrics = test_epoch(model, test_loader_PH2, device, threshold=0.5)
print(metrics) 

#for visual inspection, save predictions on test set
save_all_preds(model, test_loader_PH2, device, out_dir="test_preds", threshold=0.5)
