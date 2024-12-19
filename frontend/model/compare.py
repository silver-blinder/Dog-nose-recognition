import sys
import torch
import json
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models

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
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def compare_images(image1_path, image2_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load('model/siamese_network.pth', map_location=device))
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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"error": "需要两个图片路径参数"}))
        sys.exit(1)
        
    result = compare_images(sys.argv[1], sys.argv[2])
    print(json.dumps(result))