import os
import re
import random
from glob import glob
from typing import Optional  # added for Optional type

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import ImageDraw  # for drawing clicks


class PH2Dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
        transform_img=None,
        transform_mask=None,
        clicks_pos=5,
        clicks_neg=5,
    ):
        assert split in ("train", "val","test"), "split must be 'train' or 'val' or 'test'"
        self.root_dir = root_dir

        # collect cases (non-contiguous IDs allowed)
        case_dirs = sorted(
            d for d in glob(os.path.join(root_dir, "IMD*")) if os.path.isdir(d)
        )
        if not case_dirs:
            raise FileNotFoundError(f"No IMD* folders under {root_dir}")

        pairs = []
        for cdir in case_dirs:
            cid = os.path.basename(cdir)  # e.g., IMD002

            # image: usually exactly this file; fall back to any .bmp in the dermoscopic folder
            derm_dir = os.path.join(cdir, f"{cid}_Dermoscopic_Image")
            img_path = os.path.join(derm_dir, f"{cid}.bmp")
            if not os.path.exists(img_path):
                cand = sorted(glob(os.path.join(derm_dir, "*.bmp")))
                img_path = cand[0] if cand else None

            # mask: usually this file; fall back to first bmp in lesion folder
            lesion_dir = os.path.join(cdir, f"{cid}_lesion")
            mask_path = os.path.join(lesion_dir, f"{cid}_lesion.bmp")
            if not os.path.exists(mask_path):
                cand = sorted(glob(os.path.join(lesion_dir, "*.bmp")))
                mask_path = cand[0] if cand else None

            if img_path and mask_path and os.path.exists(img_path) and os.path.exists(mask_path):
                pairs.append((img_path, mask_path))

        if not pairs:
            raise RuntimeError("Found no (image, lesion) pairs. Check folder names/casing.")

        # reproducible split
        random.Random(seed).shuffle(pairs)
        n_total = len(pairs)
        n_val = int(round(val_ratio * n_total))
        n_test = int(round(test_ratio * n_total))
        val_pairs = pairs[:n_val]
        test_pairs = pairs[n_val:n_val + n_test]
        train_pairs = pairs[n_val+n_test:]

        if split == "train":
            self.pairs = train_pairs
        elif split == "val":
            self.pairs = val_pairs
        else:  # "test"
            self.pairs = test_pairs

        # transforms
        self.transform_img = transform_img if transform_img is not None else T.ToTensor()
        self.transform_mask = transform_mask or self._default_mask_transform
        
        # click simulation params
        self.clicks_pos = clicks_pos
        self.clicks_neg = clicks_neg

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale

        img  = F.center_crop(img,  (512, 512))
        mask = F.center_crop(mask, (512, 512))
        
        # implement click simulation 
        m_pos = (T.ToTensor()(mask)[0] > 0.5)
        m_pos_eroded = self._square_erode(m_pos, radius=10)
        m_neg = ~m_pos
        m_neg_eroded = self._square_erode(m_neg, radius=10)
        img_drawn = img.copy()
        draw = ImageDraw.Draw(img_drawn)
        
        # draw positive clicks
        for _ in range(self.clicks_pos):
            ys, xs = torch.nonzero(m_pos_eroded, as_tuple=True)
            i = torch.randint(0, xs.numel(), (1,)).item()
            pt = (int(xs[i]), int(ys[i]))
            draw.circle((pt[0], pt[1]), 10, fill=(0, 255, 0))
            
        # draw negative clicks
        for _ in range(self.clicks_neg):
            ys, xs = torch.nonzero(m_neg_eroded, as_tuple=True)
            i = torch.randint(0, xs.numel(), (1,)).item()
            pt = (int(xs[i]), int(ys[i]))
            draw.circle((pt[0], pt[1]), 10, fill=(255, 0, 0))

        x = self.transform_img(img_drawn)
        y = self.transform_mask(mask)  # 1xHxW, values {0,1}
        return x, y
    
    @staticmethod
    def _square_erode(mask_2d: torch.Tensor, radius: int) -> torch.Tensor: 
        if radius <= 0: 
            return mask_2d
        x = mask_2d.unsqueeze(0).unsqueeze(0).float()
        inv = 1.0 - x
        inv_dil = torch.nn.functional.max_pool2d(inv, kernel_size=2*radius+1, stride=1, padding=radius)
        y = 1.0 - inv_dil
        return (y[0, 0] > 0.5).to(mask_2d.dtype)

    @staticmethod
    def _default_mask_transform(mask_pil: Image.Image) -> torch.Tensor:
        """
        Convert PIL mask to {0,1} tensor (1,H,W). Any pixel > 0 -> 1.
        """
        m = T.ToTensor()(mask_pil)      # float in [0,1], shape (1,H,W)
        m = (m > 0.5).to(torch.long)    # binarize; use .float() if training with BCE
        return m

def make_ph2_loaders(
    root_dir: str,
    batch_size: int = 4,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    img_transform=None,
    mask_transform=None,
):
    from torch.utils.data import DataLoader
    train_ds = PH2Dataset(root_dir, split="train", val_ratio=val_ratio,test_ratio=test_ratio, seed=seed,
                          transform_img=img_transform, transform_mask=mask_transform)
    val_ds = PH2Dataset(root_dir, split="val", val_ratio=val_ratio,test_ratio=test_ratio, seed=seed,
                        transform_img=img_transform, transform_mask=mask_transform)
    test_ds = PH2Dataset(root_dir, split="test", val_ratio=val_ratio,test_ratio=test_ratio, seed=seed,
                            transform_img=img_transform, transform_mask=mask_transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
# Example usage:
