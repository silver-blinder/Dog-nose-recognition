#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# 犬鼻纹识别系统 - 本地一键启动脚本
# ══════════════════════════════════════════════════════════════════════════════
#
# 使用方法：
#   chmod +x start.sh
#   ./start.sh          # 同时启动前后端
#   ./start.sh backend  # 仅启动后端
#   ./start.sh frontend # 仅启动前端
#
# 依赖：
#   Backend  : Python 3.8+, pip install -r backend/requirements.txt
#   Frontend : Node.js 18+, npm install (在 frontend/ 目录)
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

log() { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()  { echo -e "${GREEN}[OK]${NC}  $*"; }
warn(){ echo -e "${YELLOW}[WARN]${NC} $*"; }
err() { echo -e "${RED}[ERR]${NC}  $*" >&2; }

# ── 检查依赖 ──────────────────────────────────────────────────────────────────
check_python() {
  if ! command -v python3 &>/dev/null; then
    err "未找到 python3，请先安装 Python 3.8+"
    exit 1
  fi
  ok "Python: $(python3 --version)"
}

check_node() {
  if ! command -v node &>/dev/null; then
    err "未找到 Node.js，请先安装 Node.js 18+"
    exit 1
  fi
  ok "Node: $(node --version)"
}

check_npm_deps() {
  if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    warn "前端依赖未安装，正在运行 npm install..."
    npm install --prefix "$FRONTEND_DIR"
  fi
}

check_python_deps() {
  if ! python3 -c "import fastapi" &>/dev/null; then
    warn "Python 依赖未安装，正在安装..."
    pip3 install -r "$BACKEND_DIR/requirements.txt" --quiet
  fi
}

# ── 启动后端 ──────────────────────────────────────────────────────────────────
start_backend() {
  check_python
  check_python_deps
  log "启动后端 FastAPI 服务 → http://localhost:8000"
  log "API 文档 → http://localhost:8000/docs"
  cd "$BACKEND_DIR"
  python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
}

# ── 启动前端 ──────────────────────────────────────────────────────────────────
start_frontend() {
  check_node
  check_npm_deps
  log "启动前端 Next.js 服务 → http://localhost:3000"
  cd "$FRONTEND_DIR"
  npm run dev
}

# ── 同时启动 ──────────────────────────────────────────────────────────────────
start_all() {
  check_python
  check_node
  check_npm_deps

  echo ""
  echo "══════════════════════════════════════════════"
  echo "    犬鼻纹识别系统 - 本地开发环境"
  echo "══════════════════════════════════════════════"
  echo "  前端地址: http://localhost:3000"
  echo "  后端地址: http://localhost:8000"
  echo "  API 文档: http://localhost:8000/docs"
  echo "══════════════════════════════════════════════"
  echo ""

  # 后台启动后端
  log "启动后端..."
  (cd "$BACKEND_DIR" && python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 2>&1 | sed 's/^/[backend] /') &
  BACKEND_PID=$!

  sleep 2  # 等待后端启动

  # 前台启动前端（阻塞）
  log "启动前端..."
  (cd "$FRONTEND_DIR" && npm run dev 2>&1 | sed 's/^/[frontend] /') &
  FRONTEND_PID=$!

  # 等待任意子进程退出
  wait $BACKEND_PID $FRONTEND_PID
}

# ── 主入口 ────────────────────────────────────────────────────────────────────
MODE="${1:-all}"
case "$MODE" in
  backend)  start_backend ;;
  frontend) start_frontend ;;
  all)      start_all ;;
  *)
    echo "用法: $0 [backend|frontend|all]"
    exit 1
    ;;
esac
