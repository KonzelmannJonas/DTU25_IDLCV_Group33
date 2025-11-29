"""
Generate train/test splits for the pothole dataset.
Creates a splits.json file with 80% train and 20% test images.
"""
import json
import random
import os

def generate_splits(num_images=643, test_ratio=0.2, seed=42):
    """Generate train/test splits for pothole images."""
    random.seed(seed)
    
    # Create list of all image names
    all_images = [f"potholes{i}" for i in range(num_images)]
    
    # Shuffle and split
    random.shuffle(all_images)
    split_idx = int(len(all_images) * (1 - test_ratio))
    
    train_images = sorted(all_images[:split_idx])
    test_images = sorted(all_images[split_idx:])
    
    splits = {
        "train": train_images,
        "test": test_images
    }
    
    # Save to JSON
    output_path = "splits.json"
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Generated splits:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Test: {len(test_images)} images")
    print(f"Saved to: {output_path}")
    
    return splits

if __name__ == "__main__":
    generate_splits()
