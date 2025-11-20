import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple

class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt)**self.gamma * bce
        return loss.mean() if self.reduction=="mean" else loss.sum()

def get_loss(name, pos_weight=None):
    name = name.lower()
    if name == "ce":
        return nn.BCEWithLogitsLoss()
    if name == "focal":
        return FocalLossWithLogits(alpha=0.8, gamma=2.0)
    if name == "weighted_ce":
        if pos_weight is None:
            raise ValueError("pos_weight is required for weighted_ce")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    raise ValueError(f"Unknown loss: {name}")

@torch.no_grad()
def estimate_pos_weight(train_loader, max_batches=50):
    pos, neg = 0.0, 0.0
    for i, (_, m) in enumerate(train_loader):
        m = m.float()
        pos += m.sum().item()
        neg += (1 - m).sum().item()
        if i + 1 >= max_batches: break
    pos = max(pos, 1.0)
    w = neg / pos
    return torch.tensor([w], dtype=torch.float32)



# Run Ablation
def run_ablation(
    dataset_name: str,
    loaders: Tuple,                        # train_loader, val_loader, test_loader
    device: torch.device,
    model_ctor: Callable[[], nn.Module],   # lambda: UNet(...)
    train_one_epoch,                       
    eval_epoch,                            
    epochs: int = 10,
    lr: float = 1e-3,
    threshold: float = 0.5,
):
    train_loader, val_loader, test_loader = loaders
    results = {}

    for loss_name in ["ce", "focal", "weighted_ce"]:
        print(f"\n=== {dataset_name} | Loss: {loss_name} ===")

        # fresh model + optimizer per loss
        model = model_ctor().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        if loss_name == "weighted_ce":
            pw = estimate_pos_weight(train_loader).to(device)
            criterion = get_loss("weighted_ce", pos_weight=pw)
            print(f"pos_weight ≈ {pw.item():.2f}")
        else:
            criterion = get_loss(loss_name)

        best_dice = -1.0
        best_state = None

        for epoch in range(1, epochs + 1):
            tr_loss, tr_m = train_one_epoch(model, train_loader, optimizer, criterion, device, threshold=threshold)
            va_loss, va_m = eval_epoch(model, val_loader, criterion, device, threshold=threshold)
            print(f"[{epoch:02d}] train={tr_loss:.4f}  val={va_loss:.4f}  val_Dice={va_m['dice']:.4f}")
            if va_m["dice"] > best_dice:
                best_dice = va_m["dice"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        # Evaluate best on all splits
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        tr_loss, tr_m = eval_epoch(model, train_loader, criterion, device, threshold=threshold)
        va_loss, va_m = eval_epoch(model,   val_loader, criterion, device, threshold=threshold)
        te_loss, te_m = eval_epoch(model,  test_loader, criterion, device, threshold=threshold)

        results[loss_name] = {"train": tr_m, "val": va_m, "test": te_m}

        def fmt(m): return f"Dice={m['dice']:.3f} IoU={m['iou']:.3f} Acc={m['acc']:.3f} Sens={m['sen']:.3f} Spec={m['spe']:.3f}"
        print("FINAL:")
        print("  train:", fmt(tr_m))
        print("  val  :", fmt(va_m))
        print("  test :", fmt(te_m))

    return results