import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model import compare_images

app = FastAPI(title="狗鼻子识别API")

# Add this code to handle port binding properly for Render
port = int(os.environ.get("PORT", 8000))

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
        print(f"Received comparison request for: {request.image1_url} and {request.image2_url}")
        result = compare_images(request.image1_url, request.image2_url)
        print(f"Comparison result: {result}")
        return result
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in /compare: {str(e)}")
        print(f"Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail=str(e))