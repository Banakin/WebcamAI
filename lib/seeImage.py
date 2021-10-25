import torch
from lib.trainModel import NeuralNetwork
import torchvision.transforms.functional as transform

def seeImage(PIL_image):
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model/model.pth"))

    classes = [
        "No person",
        "Has person",
    ]

    model.eval()
    x = transform.to_tensor(PIL_image)
    with torch.no_grad():
        pred = model(x)
        predicted = classes[pred[0].argmax(0)]
        print(f'Predicted: "{predicted}"')