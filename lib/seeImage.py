import os
import torch
from lib.trainModel import NeuralNetwork, device, CustomImageDataset
from torchvision.transforms import transforms
import numpy as np

def seeImage(PIL_image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model/model.pth"))

    classes = [
        "No person",
        "Person",
    ]

    model.eval()
    data = transform(PIL_image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model.forward(data)
        return(f"Seeing: {classes[pred[0].argmax(0)]}\n{pred}")

def seeTestingImage(idx, path, annotationsFile):
    transform = transforms.Compose([
    ])

    test_data = CustomImageDataset(os.path.join(path, annotationsFile), path, transform)

    model = NeuralNetwork()
    model.load_state_dict(torch.load("model/model.pth"))

    classes = [
        "No person",
        "Person",
    ]

    model.eval()
    x, y = test_data[idx][0].unsqueeze(0), test_data[idx][1]
    with torch.no_grad():
        pred = model.forward(x)
        # return(f"{pred}")
        return(f"Seeing: {classes[pred[0].argmax(0)]}, Actual: {classes[y]}\n{pred}")
