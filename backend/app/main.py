from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model import compare_images

app = FastAPI(title="狗鼻子识别API")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发时允许所有源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CompareRequest(BaseModel):
    image1_url: str
    image2_url: str

@app.get("/")
def read_root():
    return {"message": "狗鼻子识别API已运行"}

@app.post("/compare")
async def compare_dog_noses(request: CompareRequest):
    try:
        result = compare_images(request.image1_url, request.image2_url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))