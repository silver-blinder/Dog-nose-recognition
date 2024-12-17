import torch
import torch.nn as nn
import torchvision.models as models

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # Remove the classification head
        
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

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, label):
        loss = (1 - label) * 0.5 * torch.pow(output, 2) + \
               label * 0.5 * torch.pow(torch.clamp(self.margin - output, min=0.0), 2)
        return loss.mean()