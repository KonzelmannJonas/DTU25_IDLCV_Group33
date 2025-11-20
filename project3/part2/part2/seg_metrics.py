import torch

_EPS = 1e-7

def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.size(0), -1)

def _confusion(preds: torch.Tensor, targets: torch.Tensor):
    p = _flatten(preds).to(torch.long)
    t = _flatten(targets).to(torch.long)
    tp = (p & t).sum(dim=1)
    tn = ((1 - p) & (1 - t)).sum(dim=1)
    fp = (p & (1 - t)).sum(dim=1)
    fn = ((1 - p) & t).sum(dim=1)
    return tp, tn, fp, fn

def dice(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    tp, tn, fp, fn = _confusion(preds, targets)
    return ((2 * tp + _EPS) / (2 * tp + fp + fn + _EPS)).mean()

def iou(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    tp, tn, fp, fn = _confusion(preds, targets)
    return ((tp + _EPS) / (tp + fp + fn )).mean()

def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    tp, tn, fp, fn = _confusion(preds, targets)
    return ((tp + tn + _EPS) / (tp + tn + fp + fn + _EPS)).mean()

def sensitivity(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    tp, tn, fp, fn = _confusion(preds, targets)
    return ((tp + _EPS) / (tp + fn + _EPS)).mean()

def specificity(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    tp, tn, fp, fn = _confusion(preds, targets)
    return ((tn + _EPS) / (tn + fp + _EPS)).mean()
