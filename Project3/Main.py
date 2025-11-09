import torch
from torch import optim, nn
from torchvision import transforms as T
from ModelCNN import TinyUNet
from DRIVE_Data import DriveDataset, make_drive_loaders
from PH2_Data import PH2Dataset, make_ph2_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinyUNet(in_ch=3, base_ch=32, out_ch=1).to(device)
criterion = nn.BCEWithLogitsLoss()                        # binary
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()                        # (B,1,H,W), float for BCE
        optimizer.zero_grad()
        logits = model(xb)                                # (B,1,H,W)
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

# Example usage with your loaders (Drive or PH2):
# train_loader, val_loader = make_ph2_loaders("/path/to/PH2", batch_size=4, img_transform=T.ToTensor())
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
    tr = train_one_epoch(model, train_loader_DRIVE, optimizer, criterion)
    va = eval_epoch(model, val_loader_DRIVE, criterion)
    print(f"[{epoch+1:02d}] train_loss={tr:.4f}  val_loss={va:.4f}")
