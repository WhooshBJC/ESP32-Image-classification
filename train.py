# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 15:50:53 2025

@author: User
"""
if __name__ == '__main__':
    import torch
    from torch.nn import CrossEntropyLoss
    from torch.utils.data import DataLoader, random_split, Subset
    import torchvision
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import transforms
    
    from ResNetArchitecture2 import ResNet101
    from torch.optim import Adam
    
    def get_mean_std(loader, device="cuda"):
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
    
    class_to_idx = train_set.dataset.class_to_idx
    print (class_to_idx)
    
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle= True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    device = torch.device('cuda')
    net = ResNet101(img_channel=3, num_classes=12).to(device)
    
    optimiser = Adam(net.parameters(), lr=0.0001, weight_decay=1e-4)
    loss_fn = CrossEntropyLoss()
    
    
    import torch
    from tqdm import tqdm
    from time import time
    
    # Assume these are defined:
    # model, train_loader, test_loader, criterion, optimizer, device
    LEN_TRAIN = len(train_loader.dataset)
    LEN_TEST = len(test_loader.dataset)
    epochs = 50
    for epoch in range(epochs):
        start = time()
        tr_acc = 0
        test_acc = 0
        running_loss = 0.0
    
        # ---- TRAIN ----
        net.train()
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{epochs}") as tepoch:
            for xtrain, ytrain in tepoch:
                xtrain, ytrain = xtrain.to(device), ytrain.to(device).long()
                
                optimiser.zero_grad()
                outputs = net(xtrain)
                loss = loss_fn(outputs, ytrain)
                loss.backward()
                optimiser.step()
    
                running_loss += loss.item()
                train_pred = outputs.argmax(dim=1)
                tr_acc += (train_pred == ytrain).sum().item()
    
        ep_tr_acc = tr_acc / LEN_TRAIN
        avg_loss = running_loss / len(train_loader)
    
        # ---- EVAL ----
        net.eval()
        with torch.no_grad():
            for xtest, ytest in val_loader:
                xtest, ytest = xtest.to(device), ytest.to(device)
                test_outputs = net(xtest)
                test_pred = test_outputs.argmax(dim=1)
                test_acc += (test_pred == ytest).sum().item()
    
        ep_test_acc = test_acc / LEN_TEST
        duration = (time() - start) / 60
    
        print(f"Epoch: {epoch+1}, Time: {duration:.2f} min, Loss: {avg_loss:.4f}")
        print(f"Train Acc: {ep_tr_acc:.4f}, Test Acc: {ep_test_acc:.4f}")
    
    
    
    # Save only if accuracy meets your threshold
    threshold = 0.85  # Example: 90%
    if ep_test_acc >= threshold:
        torch.save(net.state_dict(), "resnet_weights2.pth")
        print("Model saved successfully!")
    else:
        print("Accuracy below threshold. Model not saved.")

    
    
