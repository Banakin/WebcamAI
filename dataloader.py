from torch.utils.data import DataLoader
from dataset import CustomImageDataset
import matplotlib.pyplot as plt

training_data = CustomImageDataset("./data/CustomDataset/raw/images.csv", "./data/CustomDataset/raw")
test_data = CustomImageDataset("./data/CustomDataset/testing/images.csv", "./data/CustomDataset/testing")

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

# Display image and label.
if __name__ == "__main__":
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")