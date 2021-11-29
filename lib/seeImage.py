import os
import torch
from lib.trainModel import NeuralNetwork, device, CustomImageDataset
from torchvision.transforms import transforms
import numpy as np

def seeImage(PIL_image, model_path):
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    
    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_path))

    classes = [
        "No person",
        "Person",
    ]

    model.eval()
    data = transform(PIL_image).float().to(device).unsqueeze(0)
    # print(data)
    with torch.no_grad():
        pred = model.forward(data)
        return(f"Seeing: {classes[pred[0].argmax(0)]}\n{pred}")

# def seeTestingImage(idx, path, annotationsFile):
#     transform = transforms.Compose([
#     ])

#     test_data = CustomImageDataset(os.path.join(path, annotationsFile), path, transform)

#     model = NeuralNetwork()
#     model.load_state_dict(torch.load("model/model.pth"))

#     classes = [
#         "No person",
#         "Person",
#     ]

#     model.eval()
#     x, y = test_data[idx][0].unsqueeze(0), test_data[idx][1]
#     print(x)
#     with torch.no_grad():
#         pred = model.forward(x)
#         # return(f"{pred}")
#         return(f"Seeing: {classes[pred[0].argmax(0)]}, Actual: {classes[y]}\n{pred}")
