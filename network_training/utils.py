import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torch

class DogNosePrintDataset(Dataset):
    def __init__(self, root_dir, transform=None, folders=None):
        self.root_dir = root_dir
        self.transform = transform
        self.folders = folders if folders else os.listdir(root_dir)
        self.all_images = [(folder, img) for folder in self.folders for img in os.listdir(os.path.join(root_dir, folder))]

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        folder1, img1 = self.all_images[idx]
        img1_path = os.path.join(self.root_dir, folder1, img1)
        img1 = Image.open(img1_path).convert('RGB')
        
        if random.random() > 0.5:
            folder2 = folder1
            img2 = random.choice(os.listdir(os.path.join(self.root_dir, folder2)))
        else:
            folder2 = random.choice(self.folders)
            while folder2 == folder1:
                folder2 = random.choice(self.folders)
            img2 = random.choice(os.listdir(os.path.join(self.root_dir, folder2)))

        img2_path = os.path.join(self.root_dir, folder2, img2)
        img2 = Image.open(img2_path).convert('RGB')

        label = int(folder1 == folder2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

def evaluate_model_with_metrics(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1, img2, labels = img1.cuda(), img2.cuda(), labels.cuda()
            outputs = model(img1, img2)
            outputs = outputs.view(-1)
            loss = criterion(outputs, labels.float())
            running_loss += loss.item() * img1.size(0)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.float()).sum().item()

    test_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return test_loss, accuracy

def plot_metrics(train_losses, test_losses, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))

    # Plot training and test loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r', label='Train Loss')
    plt.plot(epochs, test_losses, 'b', label='Test Loss')
    plt.title('Train and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, 'g', label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the plot to a file
    plt.savefig('training_metrics.png')

    # Display the plot
    plt.show()