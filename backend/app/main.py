import os
import io
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model import compare_images, compare_image_files

app = FastAPI(
    title="犬鼻纹识别 API",
    description="基于孪生神经网络（Siamese Network + ResNet-50）的犬鼻纹比对服务",
    version="1.0.0",
)

port = int(os.environ.get("PORT", 8000))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CompareRequest(BaseModel):
    image1_url: str
    image2_url: str


@app.get("/", tags=["health"])
def read_root():
    return {"message": "犬鼻纹识别 API 已运行", "status": "ok", "version": "1.0.0"}


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.post("/compare", tags=["inference"])
async def compare_dog_noses(request: CompareRequest):
    """通过图片 URL 比对两张犬鼻纹图片"""
    try:
        print(f"[/compare] {request.image1_url} vs {request.image2_url}")
        result = compare_images(request.image1_url, request.image2_url)
        print(f"[/compare] result={result}")
        return result
    except Exception as e:
        import traceback
        print(f"[/compare] ERROR: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare-files", tags=["inference"])
async def compare_dog_noses_files(
    image1: UploadFile = File(..., description="第一张犬鼻纹图片"),
    image2: UploadFile = File(..., description="第二张犬鼻纹图片"),
):
    """通过直接上传文件比对两张犬鼻纹图片（适合本地测试）"""
    try:
        data1 = await image1.read()
        data2 = await image2.read()
        result = compare_image_files(io.BytesIO(data1), io.BytesIO(data2))
        return result
    except Exception as e:
        import traceback
        print(f"[/compare-files] ERROR: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
