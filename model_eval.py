# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 20:36:05 2025

@author: User
"""


import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from ResNetArchitecture2 import ResNet50
from torch.optim import Adam

device = torch.device('cuda')
net = ResNet50(img_channel=3, num_classes=12).to(device)
net.load_state_dict(torch.load("resnet_weights.pth"))
net.eval()
class_map= {'battery': 0, 'biological': 1, 'brown-glass': 2, 'cardboard': 3, 'clothes': 4, 'green-glass': 5, 'metal': 6, 'paper': 7, 'plastic': 8, 'shoes': 9, 'trash': 10, 'white-glass': 11}
corrected_class_map = {v:k for k,v in class_map.items()}

data_dir = "/PythonProject/data/GarbageClassification/garbage_classification"
transform_for_stat = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.6581,0.6162,0.5856], [0.2117,0.2115,0.2157])])
dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform_for_stat)


import os
import random

# Path to your folder
folder_path = "C:/PythonProject/data/GarbageClassification/test/test1"

# Get all image files (filtering common extensions)
all_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Pick N random images (e.g., 10)
selected_images = random.sample(all_images, 10)

# Full paths for convenience
selected_paths = [os.path.join(folder_path, img) for img in selected_images]

print(selected_paths)


import matplotlib.pyplot as plt
from PIL import Image

# Load imag
for img_path in selected_paths:
    image = Image.open(img_path).convert('RGB')

# Predict
    net.eval()
    with torch.no_grad():
        input_tensor = transform_for_stat(image).unsqueeze(0).to(device)
        outputs = net(input_tensor)
        _, predicted = torch.max(outputs, 1)

# Show image with prediction
    plt.imshow(image)
    plt.title(f"Predicted: {corrected_class_map[predicted.item()]}")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
