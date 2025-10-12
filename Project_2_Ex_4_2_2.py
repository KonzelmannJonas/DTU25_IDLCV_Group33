from glob import glob
import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T


# Dataset for DualStream Network: combined so that trained/evaluated together
class FrameRGBFlowDataset(torch.utils.data.Dataset):
    """
    Dataset returning (RGB frame, stacked RGB flow images, label)
    """
    def __init__(self, root_dir, split='train', n_flow_frames=10, transform_rgb=None, transform_flow=None):
        self.root_dir = root_dir
        self.split = split
        self.n_flow_frames = n_flow_frames
        self.transform_rgb = transform_rgb
        self.transform_flow = transform_flow

        # Metadata table
        self.df = pd.read_csv(f"{root_dir}/metadata/{split}.csv")
        self.frame_dirs = sorted(glob(f"{root_dir}/frames/{split}/*/*"))

    def __len__(self):
        return len(self.frame_dirs)

    def _load_rgb_frame(self, frame_dir):
        frame_files = sorted(glob(os.path.join(frame_dir, "*.jpg")))
        if len(frame_files) == 0:
            raise FileNotFoundError(f"No RGB frames found in {frame_dir}")
        frame_file = frame_files[len(frame_files)//2]  # mid frame
        frame = Image.open(frame_file).convert("RGB")
        if self.transform_rgb:
            frame = self.transform_rgb(frame)
        else:
            frame = T.ToTensor()(frame)
        return frame

    def _load_flow_stack(self, flow_dir):
        """Load optical flow stack from .npy files."""
        flow_files = sorted(glob(os.path.join(flow_dir, "*.npy")))[:self.n_flow_frames]
        if not flow_files:
            raise FileNotFoundError(f"No .npy flow files found in {flow_dir}")

        flow_frames = []
        for ff in flow_files:
            flow = np.load(ff)  # shape (H, W, 2): u,v
            # # Convert to tensor and reorder to [2, H, W]
            flow_t = torch.from_numpy(flow).permute(0, 1, 2).float()
            # if flow.ndim != 3 or flow.shape[2] != 2:
            #     raise ValueError(f"Invalid flow shape in {ff}: expected (H, W, 2), got {flow.shape}")

   

            # Optional: normalize or resize
            if self.transform_flow:
                flow_t = self.transform_flow(flow_t)

            flow_frames.append(flow_t)

        # Concatenate temporally: [2 * n_flow_frames, H, W]
        flow_tensor = torch.cat(flow_frames, dim=0)
        return flow_tensor

    def __getitem__(self, idx):
        """Return corresponding RGB frame, optical flow stack, and label."""
        rgb_dir = self.frame_dirs[idx]
        flow_dir = rgb_dir.replace('/frames/', '/flows/')  # Adjust folder name

        video_name = os.path.basename(rgb_dir)
        label_row = self.df.loc[self.df['video_name'] == video_name]

        if label_row.empty:
            raise KeyError(f"No label found for video '{video_name}'")

        label = int(label_row['label'].iloc[0])

        rgb = self._load_rgb_frame(rgb_dir)
        flow = self._load_flow_stack(flow_dir)

        return rgb, flow, label



class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir='/dtu/datasets1/02516/ucf101_noleakage',
    split='train', 
    transform=None,
    n_frames: int = None
):
        self.root_dir = root_dir
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        if n_frames is not None:
            self.n_sampled_frames = n_frames
       
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



class StreamConvNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # Input: 224*224
        self.streamConvNet = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output = 96 × 54 × 54

            # conv2
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output = 256 × 13 × 13

            # conv3
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(), # output = 512 × 11 × 11

            # conv4
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(), # output = 512 × 9 × 9

            # conv5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            )
        self.classifier = nn.Sequential(
            nn.Flatten(), # output = 512 × 3 × 3
            # full6
            nn.Linear(in_features=512*3*3, out_features=4096),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            
            # full7
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(2048, num_classes),
            # we dont use a softmax, because cross entropy expects raw digits
            )

    def forward(self, x):
        x = self.streamConvNet(x)
        x = self.classifier(x)
        return x

class DualStreamNetwork(nn.Module):
    def __init__(self, num_classes, n_flow_channels=27):
        super().__init__()
        # do the same ConvNet structure for single frame image and multi-frame optical flow
        self.spatial_stream = StreamConvNet(in_channels=3, num_classes=num_classes)
        self.temporal_stream = StreamConvNet(in_channels=n_flow_channels, num_classes=num_classes)
        self.fusion_fc = nn.Sequential(
            nn.Linear(num_classes * 2, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb, flow):
        # Each stream outputs [B, num_classes]
        rgb_logits = self.spatial_stream(rgb)
        flow_logits = self.temporal_stream(flow)

        # Score fusion
        fused = torch.cat([rgb_logits, flow_logits], dim=1)
        out = self.fusion_fc(fused)
        return out

# Project 4.1
class AggregationPerFrameModel(nn.Module):
    def __init__(self, num_classes, n_flow_channels=27):
        super().__init__()
        # do the same ConvNet structure for single frame image and multi-frame optical flow
        self.spatial_stream = StreamConvNet(in_channels=3, num_classes=num_classes)
        self.fusion_fc = nn.Sequential(
            nn.Linear(num_classes * 2, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, num_classes)
        )



    def forward(self, rgb, flow):
        # Each stream outputs [B, num_classes]
        rgb_logits = self.spatial_stream(rgb)
        flow_logits = self.temporal_stream(flow)

        # Score fusion
        fused = torch.cat([rgb_logits, flow_logits], dim=1)
        out = self.fusion_fc(fused)
        return out

    



# TRAIN ONE EPOCH

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for rgb, flow, y in loader:
        rgb, flow, y = rgb.to(device), flow.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(rgb, flow)
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
        for rgb, flow, y in loader:
            rgb, flow, y = rgb.to(device), flow.to(device), y.to(device)
            outputs = model(rgb, flow)
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
    N_FRAMES = 1
    N_FLOW = 9
    NUM_CLASSES = 10
    NUM_EPOCHS = 20
    DROPOUT = 0.5

    print("Project_2.py startet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spatial_transform = T.Compose([
        T.Resize((224, 224)), 
        T.ToTensor()])
    

    flow_transform = T.Compose([
        T.Resize((224, 224)), 
        T.ToTensor()])

    # # train and validation dataset for single image
    # train_single_frame_image_ds = FrameImageDataset(ROOT, split='train', n_frames=N_FRAMES, transform=spatial_transform)
    # val_single_frame_image_ds = FrameImageDataset(ROOT, split='val', n_frames=N_FRAMES, transform=spatial_transform)
    
    # # train and validation dataset for the flow images
    # train_multi_frame_optical_flow_ds = FrameFlowImageDataset(ROOT, split='train', n_flow=N_FLOW, transform=flow_transform)
    # val_multi_frame_optical_flow_ds = FrameFlowImageDataset(ROOT, split='val', n_flow=N_FLOW, transform=flow_transform)

    # # Dataloader single image
    # train_single_frame_image_loader = DataLoader(train_single_frame_image_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # train_multi_frame_optical_flow_loader = DataLoader(train_multi_frame_optical_flow_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # # Dataloader flow images
    # val_single_frame_image_loader = DataLoader(val_single_frame_image_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # val_multi_frame_optical_flow_loader = DataLoader(val_multi_frame_optical_flow_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)



    '''Creating Train, Val and Testloader'''
    size = 224

    train_dataset_dualstream = FrameRGBFlowDataset(
        root_dir="/dtu/datasets1/02516/ucf101_noleakage",
        split='train',
        n_flow_frames=N_FLOW,
        transform_rgb=T.Compose([T.Resize((size, size)), T.ToTensor()]),
    )

    val_dataset_dualstream = FrameRGBFlowDataset(
        root_dir="/dtu/datasets1/02516/ucf101_noleakage",
        split='val',
        n_flow_frames=N_FLOW,
        transform_rgb=T.Compose([T.Resize((size, size)), T.ToTensor()]),
    )

    test_dataset_dualstream = FrameRGBFlowDataset(
        root_dir="/dtu/datasets1/02516/ucf101_noleakage",
        split='test',
        n_flow_frames=N_FLOW,
        transform_rgb=T.Compose([T.Resize((size, size)), T.ToTensor()]),
    )
    print("Dataset created")
    

    rgb, flow, label = train_dataset_dualstream[0]
    print(rgb.shape)   # [3, 224, 224]
    print(flow.shape)  # [2*n_flow_frames, 224, 224]


    train_loader_dualstream = DataLoader(train_dataset_dualstream, batch_size=8, shuffle=True)
    val_loader_dualstream = DataLoader(val_dataset_dualstream, batch_size=8, shuffle=True)
    test_loader_dualstream = DataLoader(test_dataset_dualstream, batch_size=8, shuffle=True)

    print("Loaders created")

    # MODEL PARAMETERS
    model = DualStreamNetwork(num_classes=NUM_CLASSES, n_flow_channels=2*N_FLOW).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    # TRAINING AND VALIDATION
    print("Training started")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader_dualstream, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader_dualstream, criterion, device)
        print(f"[{epoch+1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
    
    # TESTING TESTSET
    test_loss, test_acc = evaluate(model, test_loader_dualstream, criterion, device)
    print(f"| Test Loss: {test_loss:.4f} | Test: {test_acc:.4f}")