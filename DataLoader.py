# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 21:52:29 2025

@author: User
"""

import torch
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


def get_mean_std(loader, device='cuda'):
    mean = torch.zeros(3).to(device)
    std = torch.zeros(3).to(device)
    total_image = 0
    
    for image, _ in loader:
        image = image.to(device)
        batch_size = image.size(0)
        image = image.view(batch_size, image.size(1), -1)
        mean += image.mean(2).sum(0)
        std += image.std(2).sum(0)
        total_image += batch_size
        
    mean /= total_image
    std /= total_image
    
    return mean.cpu(), std.cpu()

data_dir = "/PythonProject/data/GarbageClassification/garbage_classification"
transform_for_stat = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor()])
dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform_for_stat)
batch_size = 32
loader = DataLoader(dataset, batch_size = batch_size, shuffle= True)
mean,std = get_mean_std(loader)
print(f"Mean: {mean}, Std: {std}")

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)])

val_test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)])

total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_indices, val_indices, test_indices = random_split(range(total_size), [train_size, val_size, test_size])

train_set = Subset(ImageFolder(data_dir,transform=train_transform),train_indices)
val_set = Subset(ImageFolder(data_dir,transform=val_test_transform),val_indices)
test_set = Subset(ImageFolder(data_dir,transform=val_test_transform),test_indices)

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle= True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)


print(f"Total: {total_size}, Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

# Check one batch
#images, labels = next(iter(train_loader))
#print(f"Train batch shape: {images.shape}, Labels: {labels}")




