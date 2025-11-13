import os
import re
import random
from glob import glob
from typing import Optional  # added for Optional type

import numpy as np
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
        sampling_method: str = "random",
        poisson_min_dist: int = 50,
    ):
        assert split in ("train", "val","test"), "split must be 'train' or 'val' or 'test'"
        assert sampling_method in ("random", "grid", "boundary", "poisson"), "sampling_method must be 'random', 'grid', 'boundary', or 'poisson'"
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
        self.sampling_method = sampling_method
        self.poisson_min_dist = poisson_min_dist

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
        m_pos_eroded = self._square_erode(m_pos, radius=1)
        m_neg = ~m_pos
        m_neg_eroded = self._square_erode(m_neg, radius=1)
        img_drawn = img.copy()
        draw = ImageDraw.Draw(img_drawn)
        
        # generate points based on sampling method
        if self.sampling_method == "random":
            pos_pts = self._sample_random_points(m_pos_eroded, self.clicks_pos)
            neg_pts = self._sample_random_points(m_neg_eroded, self.clicks_neg)
        elif self.sampling_method == "grid":
            pos_pts = self._sample_grid_points(m_pos_eroded, self.clicks_pos)
            neg_pts = self._sample_grid_points(m_neg_eroded, self.clicks_neg)
        elif self.sampling_method == "boundary":
            pos_pts = self._sample_boundary_points(m_pos_eroded, self.clicks_pos)
            neg_pts = self._sample_boundary_points(m_neg_eroded, self.clicks_neg)
        else:  # poisson sampling
            pos_pts = self._sample_poisson_points(m_pos_eroded, self.clicks_pos)
            neg_pts = self._sample_poisson_points(m_neg_eroded, self.clicks_neg)
        
        # draw positive clicks
        for pt in pos_pts:
            if pt != (-1, -1):
                draw.circle((pt[0], pt[1]), 10, fill=(0, 255, 0))
            
        # draw negative clicks
        for pt in neg_pts:
            if pt != (-1, -1):
                draw.circle((pt[0], pt[1]), 10, fill=(255, 0, 0))

        x = self.transform_img(img_drawn)
        y = self.transform_mask(mask)  # 1xHxW, values {0,1}
        return x, y, pos_pts, neg_pts
    
    def _sample_random_points(self, mask: torch.Tensor, num_points: int):
        """Sample points randomly from the mask."""
        pts = []
        for _ in range(num_points):
            ys, xs = torch.nonzero(mask, as_tuple=True)
            if xs.numel() != 0:
                i = torch.randint(0, xs.numel(), (1,)).item()
                pt = (int(xs[i]), int(ys[i]))
                pts.append(pt)
            else:
                pts.append((-1, -1))  # indicate no valid point available
        return pts
    
    def _sample_grid_points(self, mask: torch.Tensor, max_points: int):
        """Sample points using grid-based sampling from the mask with auto-calculated spacing."""
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if len(xs) == 0:
            return [(-1, -1)] * max_points

        h, w = mask.shape
        
        # Calculate optimal spacing based on desired number of points
        # Assuming we want a roughly square grid distribution
        total_area = h * w
        points_per_unit_area = max_points / total_area
        ideal_spacing = max(1, int((1.0 / points_per_unit_area) ** 0.5))
        
        # Try different spacing values to get close to desired number of points
        best_spacing = ideal_spacing
        best_diff = float('inf')
        
        # Test spacing values around the ideal
        for test_spacing in range(max(1, ideal_spacing - 20), ideal_spacing + 21):
            if test_spacing <= 0:
                continue
                
            # Count how many valid points we'd get with this spacing
            count = 0
            for y in range(0, h, test_spacing):
                for x in range(0, w, test_spacing):
                    if y < h and x < w and mask[y, x]:
                        count += 1
                    if count >= max_points:
                        break
                if count >= max_points:
                    break
            
            # Check if this spacing gives us closer to the desired number
            diff = abs(count - max_points)
            if diff < best_diff:
                best_diff = diff
                best_spacing = test_spacing
        
        # Now sample points using the best spacing
        pts = []
        for y in range(0, h, best_spacing):
            for x in range(0, w, best_spacing):
                if y < h and x < w and mask[y, x]:
                    pts.append((x, y))
                if len(pts) >= max_points:
                    break
            if len(pts) >= max_points:
                break
        
        # pad with invalid points if we don't have enough
        while len(pts) < max_points:
            pts.append((-1, -1))
        
        return pts[:max_points]
    
    def _sample_boundary_points(self, mask: torch.Tensor, n_points: int):
        """Sample points from the boundary of the mask."""
        from torch.nn.functional import max_pool2d

        # detect boundary
        mask_float = mask.float().unsqueeze(0).unsqueeze(0)
        dil = max_pool2d(mask_float, 3, stride=1, padding=1)
        boundary = (dil - mask_float).squeeze().bool()

        ys, xs = torch.nonzero(boundary, as_tuple=True)
        pts = []
        if len(xs) > 0:
            idxs = torch.randperm(len(xs))[:n_points]
            for i in idxs:
                pts.append((int(xs[i]), int(ys[i])))
        
        # pad with invalid points if we don't have enough
        while len(pts) < n_points:
            pts.append((-1, -1))
        
        return pts[:n_points]
    
    def _sample_poisson_points(self, mask: torch.Tensor, max_points: int):
        """Sample points using Poisson disk sampling for better distribution."""
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if len(xs) == 0:
            return [(-1, -1)] * max_points
            
        coords = np.stack([xs.numpy(), ys.numpy()], axis=1)
        np.random.shuffle(coords)
        pts = []

        for (x, y) in coords:
            if all((x - px)**2 + (y - py)**2 > self.poisson_min_dist**2 for px, py in pts):
                pts.append((int(x), int(y)))
            if len(pts) >= max_points:
                break
        
        # pad with invalid points if we don't have enough
        while len(pts) < max_points:
            pts.append((-1, -1))
        
        return pts[:max_points]
    
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
    sampling_method: str = "random",
    poisson_min_dist: int = 30,
):
    from torch.utils.data import DataLoader
    train_ds = PH2Dataset(root_dir, split="train", val_ratio=val_ratio,test_ratio=test_ratio, seed=seed,
                          transform_img=img_transform, transform_mask=mask_transform,
                          sampling_method=sampling_method, poisson_min_dist=poisson_min_dist)
    val_ds = PH2Dataset(root_dir, split="val", val_ratio=val_ratio,test_ratio=test_ratio, seed=seed,
                        transform_img=img_transform, transform_mask=mask_transform,
                        sampling_method=sampling_method, poisson_min_dist=poisson_min_dist)
    test_ds = PH2Dataset(root_dir, split="test", val_ratio=val_ratio,test_ratio=test_ratio, seed=seed,
                            transform_img=img_transform, transform_mask=mask_transform,
                            sampling_method=sampling_method, poisson_min_dist=poisson_min_dist)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
# Example usage:
