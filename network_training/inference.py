
from model import SiameseNetwork
import torch
from PIL import Image
import torchvision.transforms as transforms

# Define the same transformations as used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing the image to 224x224 pixels
    transforms.ToTensor(),          # Converting the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizing with ImageNet means and stds
])

model = SiameseNetwork().cuda()
# Load the saved state dictionary
model.load_state_dict(torch.load('siamese_network.pth', weights_only=True))
model.eval()  # Set the model to evaluation mode

def infer(model, img1, img2):
    model.eval()
    with torch.no_grad():
        img1 = img1.cuda()
        img2 = img2.cuda()
        output = model(img1, img2)
        return output

# Load and preprocess the first image
img1_path = r'1.jpg'
img2_path = r'2.jpg'

img1 = Image.open(img1_path).convert('RGB')
img1 = transform(img1)
img1 = img1.unsqueeze(0)  # Add a batch dimension
img2 = Image.open(img2_path).convert('RGB')
img2 = transform(img2)
img2 = img2.unsqueeze(0)  # Add a batch dimension

# Perform inference
output = infer(model, img1.cuda(), img2.cuda())
# print(output)
prediction = (output > 0.5).float()

# Output the result
print(f"Are the two images from the same dog's nose print? {'Yes' if prediction.item() == 1 else 'No'}")



