#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# 论文构建脚本 - 一键生成 论文.docx
# 步骤：
#   0. 生成架构/流程图 PNG（黑白学术风格）
#   1. 生成 reference.docx（格式模板）
#   2. 预处理 论文.md → 论文_pandoc.md
#   3. Pandoc 转换 → 论文.docx（初版）
#   4. post_process.py 后处理（格式修复、页码等）
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
log() { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()  { echo -e "${GREEN}[OK]${NC}  $*"; }

log "步骤 0/5：生成架构/流程图 PNG"
python3 make_diagrams.py

log "步骤 1/5：生成 reference.docx"
python3 make_reference_docx.py

log "步骤 2/5：预处理 论文.md"
python3 preprocess_thesis.py

log "步骤 3/5：Pandoc 转换"
pandoc 论文_pandoc.md \
  --from markdown \
  --to docx \
  --reference-doc reference.docx \
  --output 论文.docx \
  --mathml \
  2>&1 | grep -v "WARNING\|translations\|no translation" || true

log "步骤 4/5：后处理格式修复"
python3 post_process.py

ok "完成！输出文件：$SCRIPT_DIR/论文.docx"
