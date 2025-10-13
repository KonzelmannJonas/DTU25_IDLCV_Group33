from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T

class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir='/dtu/datasets1/02516/ucf101_noleakage',
    split='train', 
    transform=None
):
        self.root_dir = root_dir
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
       
    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        # use os.path to be platform-independent
        video_name = os.path.basename(os.path.dirname(frame_path))
        video_meta = self._get_meta('video_name', video_name)
        if video_meta.empty:
            raise KeyError(f"no metadata row found for video_name='{video_name}' (frame: {frame_path})")
        # take the first matching label (expect exactly one)
        label = int(video_meta['label'].iloc[0])
        
        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir = '/dtu/datasets1/02516/ucf101_noleakage', 
    split = 'train', 
    transform = None,
    stack_frames = True
):
        self.root_dir = root_dir

        # glob for videos - keep as-is but store root for later path computations
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
        # platform-independent extraction of video name (filename without extension)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_meta = self._get_meta('video_name', video_name)
        if video_meta.empty:
            raise KeyError(f"no metadata row found for video_name='{video_name}' (video: {video_path})")
        label = int(video_meta['label'].iloc[0])

        # compute frames directory relative to the provided root_dir in a robust way
        # e.g. videos/.../v_xxx.avi -> frames/.../v_xxx
        rel = os.path.relpath(video_path, os.path.join(self.root_dir, 'videos'))
        rel_no_ext = os.path.splitext(rel)[0]
        video_frames_dir = os.path.join(self.root_dir, 'frames', rel_no_ext)
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]
        
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)


        return frames, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames




class FrameFlowImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir = '/dtu/datasets1/02516/ucf101_noleakage', 
    split = 'train', 
    transform = None,
    stack_frames = True
):
        self.root_dir = root_dir

        # glob for videos - keep as-is but store root for later path computations
        self.video_paths = sorted(glob(f'{root_dir}/flows_png/{split}/*/*.avi'))
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
        # platform-independent extraction of video name (filename without extension)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_meta = self._get_meta('video_name', video_name)
        if video_meta.empty:
            raise KeyError(f"no metadata row found for video_name='{video_name}' (video: {video_path})")
        label = int(video_meta['label'].iloc[0])

        # compute frames directory relative to the provided root_dir in a robust way
        # e.g. videos/.../v_xxx.avi -> frames/.../v_xxx
        rel = os.path.relpath(video_path, os.path.join(self.root_dir, 'videos'))
        rel_no_ext = os.path.splitext(rel)[0]
        video_frames_dir = os.path.join(self.root_dir, 'frames', rel_no_ext)
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]
        
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)


        return frames, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import torch.optim as optim
    from torchvision import models
    import torch.nn as nn






#     # path to ufc10 folder
#     root_dir = '/dtu/datasets1/02516/ucf101_noleakage'

#     transform = T.Compose([T.Resize((64, 64)),T.ToTensor()]) # T.Compose: does several transforms together
    
#     # define the datasets we want to use
#     frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform) # takes the validationset of the single frames, in each folder are 10 images
#     framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = True) # stacks the frames 
#     framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = False)
#     frameflow_dataset = FrameFlowImageDataset(root_dir=root_dir, split='val', transform=transform)

#     # use DataLoader to split them into batches to train 
#     frameimage_loader = DataLoader(frameimage_dataset,  batch_size=8, shuffle=False)
#     framevideostack_loader = DataLoader(framevideostack_dataset,  batch_size=8, shuffle=False)
#     framevideolist_loader = DataLoader(framevideolist_dataset,  batch_size=8, shuffle=False)

#     # for frames, labels in frameimage_loader:
#     #     print(frames.shape, labels.shape) # [batch, channels, height, width]

#     # for video_frames, labels in framevideolist_loader:
#     #     print(45*'-')
#     #     for frame in video_frames: # loop through number of frames
#     #         print(frame.shape, labels.shape)# [batch, channels, height, width]

#     for video_frames, labels in framevideostack_loader:
#         print(video_frames.shape, labels.shape) # [batch, channels, number of frames, height, width]


# ###########################################################################

#     EPOCHS = 10
    
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     CHECKPOINT_DIR = "./checkpoints"




# maybe change dropout
class DualStreamNetwork(nn.Module):
    def __init__(self, num_classes, n_flow_channels=18):
        super().__init__()

        self.Spatialstream = nn.Sequential(
            # conv1
            nn.Con2d(in_channels=3, out_channels=96, kernel_size=7, stride=2)
            nn.BatchNorm2d(96)
            nn.MaxPool2d(2, 2)
            # conv2
            nn.Con2d(in_channels=96, out_channels=256, kernel_size=5, stride=2)
            nn.BatchNorm2d(256)
            nn.MaxPool2d(2, 2)
            # conv3
            nn.Con2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
            # conv4
            nn.Con2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
            # conv5
            nn.Con2d(in_channels=512, out_channels=512, kernel_size=3, stride=1)
            nn.MaxPool2d(2, 2)
            # full6
            nn.Linear(in_channels=512, out_channels=4096, kernel_size=3, stride=1)
            nn.Dropout(0.2)
            # full7
            nn.Linear(in_channels=4096, out_channels=2048, kernel_size=3, stride=1)
            nn.Dropout(0.2)
            # softmax
            nn.Softmax()
        )

        self.Temporalstream = nn.Sequential(
            # conv1
            nn.Con2d(in_channels=3, out_channels=96, kernel_size=7, stride=2)
            nn.BatchNorm2d(96)
            nn.MaxPool2d(2, 2)
            # conv2
            nn.Con2d(in_channels=96, out_channels=256, kernel_size=5, stride=2)
            nn.BatchNorm2d(256)
            nn.MaxPool2d(2, 2)
            # conv3
            nn.Con2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
            # conv4
            nn.Con2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
            # conv5
            nn.Con2d(in_channels=512, out_channels=512, kernel_size=3, stride=1)
            nn.MaxPool2d(2, 2)
            # full6
            nn.Linear(in_channels=512, out_channels=4096, kernel_size=3, stride=1)
            nn.Dropout(0.2)
            # full7
            nn.Linear(in_channels=4096, out_channels=2048, kernel_size=3, stride=1)
            nn.Dropout(0.2)
            # softmax
            nn.Softmax()
        )
        # Spatial stream (RGB)
        self.spatial_backbone = models.resnet18(pretrained=True)
        self.spatial_backbone.fc = nn.Linear(self.spatial_backbone.fc.in_features, num_classes)

        # Temporal stream (Flow)
        self.temporal_backbone = models.resnet18(pretrained=True)
        # Replace first conv to accept flow channels
        self.temporal_backbone.conv1 = nn.Conv2d(
            n_flow_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.temporal_backbone.fc = nn.Linear(self.temporal_backbone.fc.in_features, num_classes)

        # Final fusion classifier
        self.fusion_fc = nn.Linear(num_classes * 2, num_classes)

    def forward(self, rgb, flow):
        # rgb: [B, N, 3, H, W]
        B, N, C, H, W = rgb.shape
        rgb = rgb.view(B * N, C, H, W)
        rgb_feat = self.spatial_backbone(rgb)  # [B*N, num_classes]
        rgb_feat = rgb_feat.view(B, N, -1).mean(1)  # average+ over frames

        flow_feat = self.temporal_backbone(flow)  # [B, num_classes]

        fused = torch.cat([rgb_feat, flow_feat], dim=1)
        out = self.fusion_fc(fused)
        return out


# =========================
#  TRAIN / EVAL
# =========================

def train_one_epoch(model, rgb_loader, flow_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for (rgb, y_rgb), (flow, y_flow) in zip(rgb_loader, flow_loader):
        assert torch.equal(y_rgb, y_flow), "RGB and Flow labels must match"

        rgb, flow, y = rgb.to(device), flow.to(device), y_rgb.to(device)
        optimizer.zero_grad()
        outputs = model(rgb, flow)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


def evaluate(model, rgb_loader, flow_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for (rgb, y_rgb), (flow, y_flow) in zip(rgb_loader, flow_loader):
            rgb, flow, y = rgb.to(device), flow.to(device), y_rgb.to(device)
            outputs = model(rgb, flow)
            loss = criterion(outputs, y)
            running_loss += loss.item() * y.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
    return running_loss / total, correct / total


# =========================
#  MAIN SCRIPT
# =========================

if __name__ == "__main__":
    ROOT = "/dtu/datasets1/02516/ucf101_noleakage"
    BATCH_SIZE = 8
    N_FRAMES = 10
    N_FLOW = 9
    NUM_CLASSES = 101  # UCF-101

    print("Project_2 startet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spatial_transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    flow_transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    # only take one image 
    train_single_frame_image_ds = FrameImageDataset(ROOT, split='train', n_frames=N_FRAMES, transform=spatial_transform)
    train_multi_frame_optical_flow_ds = FrameFlowImageDataset(ROOT, split='train', n_flow=N_FLOW, transform=flow_transform)

    val_single_frame_image_ds = FrameImageDataset(ROOT, split='val', n_frames=N_FRAMES, transform=spatial_transform)
    val_multi_frame_optical_flow_ds = FrameFlowImageDataset(ROOT, split='val', n_flow=N_FLOW, transform=flow_transform)

    train_single_frame_image_loader = DataLoader(train_single_frame_image_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    train_multi_frame_optical_flow_loader = DataLoader(train_multi_frame_optical_flow_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_single_frame_image_loader = DataLoader(val_single_frame_image_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    val_multi_frame_optical_flow_loader = DataLoader(val_multi_frame_optical_flow_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = DualStreamNetwork(num_classes=NUM_CLASSES, n_flow_channels=2*N_FLOW).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    NUM_EPOCHS = 10
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_single_frame_image_loader, val_single_frame_image_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_single_frame_image_loader, val_multi_frame_optical_flow_loader, criterion, device)

        print(f"[{epoch+1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")