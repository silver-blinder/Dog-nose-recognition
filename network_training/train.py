from utils import *
from model import SiameseNetwork, ContrastiveLoss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=25):
    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for img1, img2, labels in train_dataloader:
            img1, img2, labels = img1.cuda(), img2.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(img1, img2)
            outputs = outputs.view(-1)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * img1.size(0)
        
        epoch_loss = running_loss / len(train_dataloader.dataset)
        train_losses.append(epoch_loss)

        test_loss, test_accuracy = evaluate_model_with_metrics(model, test_dataloader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # Plot the results
    plot_metrics(train_losses, test_losses, test_accuracies)

    return model

def create_train_test_split(root_dir, test_size=0.2):
    folders = os.listdir(root_dir)
    train_folders, test_folders = train_test_split(folders, test_size=test_size, random_state=42)

    train_dataset = DogNosePrintDataset(root_dir=root_dir, transform=transform, folders=train_folders)
    test_dataset = DogNosePrintDataset(root_dir=root_dir, transform=transform, folders=test_folders)
    
    return train_dataset, test_dataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

root_dir = 'dir_train'
batch_size = 32
num_epochs = 20
learning_rate = 0.001

# Create train and test datasets
train_dataset, test_dataset = create_train_test_split(root_dir=root_dir)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, criterion, and optimizer
model = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model with visualization
model = train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=num_epochs)

# Save the model's state dictionary
torch.save(model.state_dict(), 'siamese_network.pth')