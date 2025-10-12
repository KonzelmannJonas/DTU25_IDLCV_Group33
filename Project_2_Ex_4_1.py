from glob import glob
import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F




class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir = '/work3/ppar/data/ucf101', 
    split = 'train', 
    transform = None,
    stack_frames = True
):

        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]
        
        if self.stack_frames:
            frames = torch.stack(frames).permute(0, 1, 2, 3)


        return frames, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames


class FrameCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Simple convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [B, 3, H, W]
        feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        logits = self.classifier(feats)
        return logits


class VideoAveragingModel(nn.Module):
    """
    Apply CNN on each frame, average softmax probabilities,
    and output a video-level prediction.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.frame_cnn = FrameCNN(num_classes=num_classes)

    def forward(self, frames):
        B, N, C, H, W = frames.shape
        frames = frames.view(B * N, C, H, W)
        logits = self.frame_cnn(frames)                # [B*N, num_classes]
        probs = F.softmax(logits, dim=1)               # frame-level probabilities
        probs = probs.view(B, N, -1)
        avg_probs = probs.mean(dim=1)                  # average probabilities
        return torch.log(avg_probs + 1e-8)   # to avoid log(0)
    





# TRAIN ONE EPOCH

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for frames, y in loader:
        frames, y = frames.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total

# EVALUATE

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for frames, y in loader:
            frames, y = frames.to(device), y.to(device)
            outputs = model(frames)
            loss = criterion(outputs, y)
            running_loss += loss.item() * y.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
    return running_loss / total, correct / total



# MAIN

if __name__ == "__main__":
    ROOT = "/dtu/datasets1/02516/ucf101_noleakage"
    BATCH_SIZE = 8
    N_FRAMES = 10
    NUM_CLASSES = 10
    NUM_EPOCHS = 10

    print("Starting...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Frame_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    train_dataset = FrameVideoDataset(ROOT, split='train', transform=Frame_transform)
    val_dataset   = FrameVideoDataset(ROOT, split='val', transform=Frame_transform)
    test_dataset  = FrameVideoDataset(ROOT, split='test', transform=Frame_transform)


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = VideoAveragingModel(num_classes=NUM_CLASSES).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Training started")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"[{epoch+1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")