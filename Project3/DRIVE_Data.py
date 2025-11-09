import os
import re
import random
from glob import glob

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class DriveDataset(Dataset):
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
        self.train_img_dir = os.path.join(root_dir, "training", "images")
        self.train_mask_dir = os.path.join(root_dir, "training", "1st_manual")

        # Find all training images
        img_paths = sorted(glob(os.path.join(self.train_img_dir, "*.tif")))
        if len(img_paths) == 0:
            raise FileNotFoundError(f"No .tif images found in {self.train_img_dir}")

        # Build (image_path, mask_path) pairs by matching the numeric ID
        # Example mapping:  "21_training.tif"  -> "21_manual1.gif"
        pairs = []
        for ip in img_paths:
            fname = os.path.basename(ip)
            # Extract leading digits (e.g., "21" from "21_training.tif" or "21_test.tif")
            m = re.match(r"(\d+)_", fname)
            if not m:
                # fallback: try just digits anywhere
                m = re.search(r"(\d+)", fname)
            if not m:
                continue
            img_id = m.group(1)  # e.g., "21"
            mask_path = os.path.join(self.train_mask_dir, f"{img_id}_manual1.gif")
            if os.path.exists(mask_path):
                pairs.append((ip, mask_path))


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

        # Load image (RGB)
        img = Image.open(img_path).convert("RGB")
        # Load mask (grayscale)
        mask = Image.open(mask_path).convert("L")

        # Apply transforms
        x = self.transform_img(img)
        y = self.transform_mask(mask)  # expected to be 1xHxW with {0,1}

        return x, y

    @staticmethod
    def _default_mask_transform(mask_pil: Image.Image) -> torch.Tensor:
        """
        Convert PIL mask to a {0,1} tensor of shape (1, H, W), dtype=torch.long.
        Any pixel > 0 becomes 1.
        """
        m = T.ToTensor()(mask_pil)  # float tensor in [0,1], shape (1,H,W)
        m = (m > 0.5).to(torch.long)  # binarize to 0/1
        return m


# (Optional) tiny helper to get DataLoaders quickly
def make_drive_loaders(
    root_dir: str,
    batch_size: int = 4,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 2,
    img_transform=None,
    mask_transform=None,
):
    from torch.utils.data import DataLoader
    train_ds = DriveDataset(root_dir, split="train", val_ratio=val_ratio,test_ratio=test_ratio, seed=seed,
                          transform_img=img_transform, transform_mask=mask_transform)
    val_ds = DriveDataset(root_dir, split="val", val_ratio=val_ratio,test_ratio=test_ratio, seed=seed,
                        transform_img=img_transform, transform_mask=mask_transform)
    test_ds = DriveDataset(root_dir, split="test", val_ratio=val_ratio,test_ratio=test_ratio, seed=seed,
                            transform_img=img_transform, transform_mask=mask_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# Example usage:

