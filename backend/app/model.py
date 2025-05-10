import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import os
from pathlib import Path

# 获取当前目录
BASE_DIR = Path(__file__).resolve().parent.parent

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward_one(self, x):
        x = self.backbone(x)
        return x
    
    def forward(self, img1, img2):
        output1 = self.forward_one(img1)
        output2 = self.forward_one(img2)
        distance = torch.abs(output1 - output2)
        output = self.fc(distance)
        return output

def load_image(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download image, status code: {response.status_code}")
        
        image = Image.open(BytesIO(response.content))
        image = image.convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return transform(image).unsqueeze(0)
    except Exception as e:
        raise Exception(f"Error loading image: {str(e)}")

def compare_images(image1_url, image2_url):
    try:
        # Log progress to help debug
        print(f"Starting comparison of {image1_url} and {image2_url}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Add explicit model path check
        model_path = os.path.join(BASE_DIR, 'model', 'siamese_network.pth')
        if not os.path.exists(model_path):
            raise Exception(f"Model file not found at {model_path}")
        print(f"Model file exists at: {model_path}")
        
        # 加载模型
        try:
            model = SiameseNetwork().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
        # 处理图片
        img1 = load_image(image1_url).to(device)
        img2 = load_image(image2_url).to(device)
        
        with torch.no_grad():
            output = model(img1, img2)
            prediction = (output > 0.5).float()
            confidence = torch.sigmoid(output).item()
        
        return {
            "is_same_dog": bool(prediction.item()),
            "confidence": round(confidence * 100, 2)
        }
    except Exception as e:
        # More detailed error logging
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error comparing images: {str(e)}")
        print(f"Traceback: {error_trace}")
        raise Exception(f"Error comparing images: {str(e)}")