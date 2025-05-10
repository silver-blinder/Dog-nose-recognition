import sys
import os
from pathlib import Path

# 添加目录到系统路径
path = Path(__file__).parent.absolute()
sys.path.insert(0, str(path))

# 导入FastAPI应用
from app.main import app_wsgi as application 