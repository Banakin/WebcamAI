import torch
from torch.utils.data import dataloader
from train import NeuralNetwork
from dataloader import test_data

model = NeuralNetwork()
model.load_state_dict(torch.load("model/model.pth"))

classes = [
    "No person",
    "Has person",
]

def main():
    model.eval()
    x, y = test_data[8][0], test_data[8][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

if __name__ == '__main__':
    main()