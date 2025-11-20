from glob import glob
import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """
    Plots training & validation loss and accuracy curves.
    Args:
        train_losses (list): Training loss per epoch
        val_losses (list): Validation loss per epoch
        train_accs (list): Training accuracy per epoch
        val_accs (list): Validation accuracy per epoch
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # ---- Loss Plot ----
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'o-', label='Train Loss')
    plt.plot(epochs, val_losses, 'o-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)

    # ---- Accuracy Plot ----
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'o-', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'o-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("late_fusion.png")   # saves plot to file
    plt.close()



class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir = '/dtu/datasets1/02516/ufc10', #'/work3/ppar/data/ucf101', 
    split = 'train', 
    transform_rgb=None,
    transform_gray=None,
    stack_frames = True,
    dual_mode=False
):

        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform_rgb = transform_rgb
        self.transform_gray = transform_gray
        self.stack_frames = stack_frames
        self.dual_mode = dual_mode
        
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

        if self.dual_mode:
            # both RGB and grayscale
            frames_rgb = [self.transform_rgb(frame) for frame in video_frames]
            frames_gray = [self.transform_gray(frame) for frame in video_frames]
            frames_rgb = torch.stack(frames_rgb)
            frames_gray = torch.stack(frames_gray)

            return frames_rgb, frames_gray, label

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
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),  # 3 input channels; padding 1 to get full image
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128×128 to 64×64

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 64×64 to 32×32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 64×64 to 32×32
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
        )


        
    def forward(self, x):
            x = self.convolutional(x)
            x = self.classifier(x)
            return x


class LateFusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.rgb_stream = FrameCNN(num_classes)
        self.gray_stream = FrameCNN(num_classes)

    def forward(self, frames_rgb, frames_gray):
        # frames_rgb and frames_gray are [B, N, C, H, W]
        B, N, C, H, W = frames_rgb.shape

        rgb_logits_list = []
        gray_logits_list = []
        for i in range(N):
            logits_rgb_i = self.rgb_stream(frames_rgb[:, i])
            logits_gray_i = self.gray_stream(frames_gray[:, i])
            rgb_logits_list.append(logits_rgb_i.unsqueeze(1))
            gray_logits_list.append(logits_gray_i.unsqueeze(1))

        logits_rgb = torch.cat(rgb_logits_list, dim=1).mean(dim=1) #Average the logits
        logits_gray = torch.cat(gray_logits_list, dim=1).mean(dim=1)
        fused_logits = (logits_rgb + logits_gray) / 2  # simple averaging fusion
        return fused_logits





# TRAIN ONE EPOCH

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0


    for frames_rgb, frames_gray, y in loader:
        frames_rgb, frames_gray, y = frames_rgb.to(device), frames_gray.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(frames_rgb, frames_gray)
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
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for frames_rgb, frames_gray, y in loader:
            frames_rgb, frames_gray, y = frames_rgb.to(device), frames_gray.to(device), y.to(device)
            outputs = model(frames_rgb, frames_gray)
            loss = criterion(outputs, y)

            total_loss += loss.item() * y.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


# MAIN

if __name__ == "__main__":
    ROOT = "/dtu/datasets1/02516/ufc10"
    BATCH_SIZE = 8
    N_FRAMES = 10
    NUM_CLASSES = 10
    NUM_EPOCHS = 10
    SIZE = 128
    DROPOUT = 0.5

    print("Starting...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_rgb = T.Compose([
        T.Resize((SIZE, SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    transform_gray = T.Compose([
        T.Resize((SIZE, SIZE)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FrameVideoDataset(
        ROOT, split='train',
        transform_rgb=transform_rgb,
        transform_gray=transform_gray,
        dual_mode=True
    )
    val_dataset = FrameVideoDataset(
        ROOT, split='val',
        transform_rgb=transform_rgb,
        transform_gray=transform_gray,
        dual_mode=True
    )
    test_dataset = FrameVideoDataset(
        ROOT, split='test',
        transform_rgb=transform_rgb,
        transform_gray=transform_gray,
        dual_mode=True
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LateFusionModel(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    print("Training started")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"[{epoch+1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

# Plot results
plot_training_curves(train_losses, val_losses, train_accs, val_accs)

