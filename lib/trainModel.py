import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms
import torch.nn.functional as F

# See if we can use CUDA (GPU), otherwise use the CUP
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Main function that trains and saves the model
def trainAndSave(path, annotationsFile, epochs, batchSize, test_path, test_batchSize, modelSavePath, output_labels):
    transform = transforms.Compose([
    ])

    train_dataset = CustomImageDataset(os.path.join(path, annotationsFile), path, transform)
    test_dataset = CustomImageDataset(os.path.join(test_path, annotationsFile), test_path, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batchSize, shuffle=True)

    classes = output_labels.split(", ")
    
    model = NeuralNetwork(len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), modelSavePath)
    print("Saved PyTorch Model State to "+modelSavePath)

# Training Function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Testing function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path).float()
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
        
class NeuralNetwork(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*29*29, 120)
        self.fc2 = nn.Linear(120, 75)
        self.fc3 = nn.Linear(75, outputs)

    def forward(self, x):
        # 2D Convolution and pooling stack
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch

        # Linearization stack
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x