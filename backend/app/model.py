import io
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import requests
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ── 路径 ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── 预处理 ───────────────────────────────────────────────────────────────────
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── 模型定义 ─────────────────────────────────────────────────────────────────

class SiameseNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # 移除分类头，输出 2048 维特征

        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        feat1 = self.forward_one(img1)
        feat2 = self.forward_one(img2)
        diff = torch.abs(feat1 - feat2)
        return self.fc(diff)


# ── 模型单例（懒加载） ───────────────────────────────────────────────────────
_model: Optional[SiameseNetwork] = None
_device: Optional[torch.device] = None


def _get_model() -> Tuple[SiameseNetwork, torch.device]:
    global _model, _device
    if _model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = BASE_DIR / "model" / "siamese_network.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件未找到: {model_path}")

        model = SiameseNetwork().to(device)
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        _model = model
        _device = device
        print(f"[model] 加载完成，设备: {device}，路径: {model_path}")

    return _model, _device  # type: ignore[return-value]


# ── 图片加载 ─────────────────────────────────────────────────────────────────

def _load_from_url(url: str) -> torch.Tensor:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return _TRANSFORM(img).unsqueeze(0)


def _load_from_stream(stream: io.IOBase) -> torch.Tensor:
    img = Image.open(stream).convert("RGB")
    return _TRANSFORM(img).unsqueeze(0)


# ── 推理接口 ─────────────────────────────────────────────────────────────────

def _infer(tensor1: torch.Tensor, tensor2: torch.Tensor) -> dict:
    model, device = _get_model()
    t1 = tensor1.to(device)
    t2 = tensor2.to(device)
    with torch.no_grad():
        output = model(t1, t2)
        score = torch.sigmoid(output).item()   # 归一化到 [0,1]
        prediction = score > 0.5

    return {
        "is_same_dog": bool(prediction),
        "confidence": round(score * 100, 2),
        "raw_score": round(score, 6),
    }


def compare_images(image1_url: str, image2_url: str) -> dict:
    """通过 URL 比对两张犬鼻纹图片"""
    try:
        t1 = _load_from_url(image1_url)
        t2 = _load_from_url(image2_url)
        return _infer(t1, t2)
    except Exception as e:
        raise RuntimeError(f"比对失败（URL 模式）: {e}") from e


def compare_image_files(stream1: io.IOBase, stream2: io.IOBase) -> dict:
    """通过文件流比对两张犬鼻纹图片"""
    try:
        t1 = _load_from_stream(stream1)
        t2 = _load_from_stream(stream2)
        return _infer(t1, t2)
    except Exception as e:
        raise RuntimeError(f"比对失败（文件模式）: {e}") from e
