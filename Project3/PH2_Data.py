import os
import re
import random
from glob import glob

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import functional as F


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

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale

        img  = F.center_crop(img,  (512, 512))
        mask = F.center_crop(mask, (512, 512))

        x = self.transform_img(img)
        y = self.transform_mask(mask)  # 1xHxW, values {0,1}
        return x, y

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
