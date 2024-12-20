import sys
import torch
import json
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import requests
from io import BytesIO

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

def load_image(image_path):
    try:
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            if response.status_code != 200:
                raise Exception(f"Failed to download image, status code: {response.status_code}")
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)
        
        # 转换为RGB
        image = image.convert('RGB')
        
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error in load_image: {str(e)}", file=sys.stderr)  # 错误信息
        raise

def compare_images(image1_path, image2_path):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}", file=sys.stderr)  # 调试信息
        
        # 加载模型
        model = SiameseNetwork().to(device)
        model.load_state_dict(torch.load('model/siamese_network.pth', 
                                       map_location=device,
                                       weights_only=True))  # 添加 weights_only=True
        model.eval()
        
        # 处理图片
        img1 = load_image(image1_path).to(device)
        img2 = load_image(image2_path).to(device)
        
        with torch.no_grad():
            output = model(img1, img2)
            prediction = (output > 0.5).float()
            confidence = torch.sigmoid(output).item()
        
        return {
            "is_same_dog": bool(prediction.item()),
            "confidence": round(confidence * 100, 2)
        }
    except Exception as e:
        print(f"Error in compare_images: {str(e)}", file=sys.stderr)  # 错误信息
        raise

if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            print(json.dumps({"error": "需要两个图片路径参数"}))
            sys.exit(1)
        
        print(f"Comparing images: {sys.argv[1]} and {sys.argv[2]}", file=sys.stderr)  # 调试信息
        result = compare_images(sys.argv[1], sys.argv[2])
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)