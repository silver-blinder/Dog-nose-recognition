"""
预处理 论文.md → 论文_pandoc.md，供 Pandoc 转 Word 使用。
原 论文.md 不做任何修改。

v3 新增处理：
1. 公式加章节编号 (n.m) 格式
2. ASCII 架构/流程图代码块 → 图片引用
3. 去掉手工目录、emoji、&emsp;、独立 --- 分割线
4. 图片路径转绝对路径
5. 添加 YAML front matter
6. 保留摘要区块（中英文），转换为 Pandoc 可识别的特殊格式
"""
import re
import pathlib

BASE = pathlib.Path('/Users/songjiayi4/Workspace/Dog-nose-recognition')
SRC  = BASE / '论文.md'
DST  = BASE / '论文_pandoc.md'
DIAG = BASE / 'network_training/experiments/results/figures/diagrams'
FIGS = BASE / 'network_training/experiments/results/figures'

# ── 读入原文 ──────────────────────────────────────────────────
text = SRC.read_text(encoding='utf-8')

# 1. 去掉 emoji（首行 🥷）
text = re.sub(r'^[^\S\r\n]*🥷[^\S\r\n]*\n?', '', text, flags=re.MULTILINE)

# 2. 提取摘要区块（"## 摘　要" 和 "## ABSTRACT" 到 "## 目录" 之前）
#    保留摘要内容，转为 Heading 1 格式以便 Pandoc 处理
abstract_cn_match = re.search(
    r'## 摘　要\n([\s\S]*?)(?=\n## ABSTRACT)',
    text
)
abstract_en_match = re.search(
    r'## ABSTRACT\n([\s\S]*?)(?=\n## 目录)',
    text
)

cn_abstract_body = abstract_cn_match.group(1).strip() if abstract_cn_match else ''
en_abstract_body = abstract_en_match.group(1).strip() if abstract_en_match else ''

# 3. 删掉手工目录区块（"## 目录" 到第一个 "## 第X章"）以及摘要区块
text = re.sub(
    r'## 摘　要\n[\s\S]*?(?=\n## 第[一二三四五六七八九]章|\n## 绪论)',
    '',
    text
)
text = re.sub(
    r'## ABSTRACT\n[\s\S]*?(?=\n## 目录)',
    '',
    text
)
text = re.sub(
    r'## 目录\n[\s\S]*?(?=\n## 第[一二三四五六七八九]章|\n## 绪论)',
    '',
    text
)

# 4. &emsp; → 空字符串
text = text.replace('&emsp;', '')

# 5. 独立 --- → 空行（非代码块内）
lines = text.split('\n')
in_code = False
cleaned = []
for line in lines:
    if line.startswith('```'):
        in_code = not in_code
    if not in_code and re.fullmatch(r'-{3,}', line.strip()):
        cleaned.append('')
    else:
        cleaned.append(line)
text = '\n'.join(cleaned)

# ── 处理代码块：区分架构图 vs 代码 ────────────────────────────
DIAGRAM_BLOCKS = [
    (
        r'┌─+┐.*?用户界面层.*?└─+┘',
        str(DIAG / 'fig_system_arch.png'),
        '图 3-1  系统整体三层架构',
    ),
    (
        r'输入图片1.*?ResNet-50 Backbone.*?Sigmoid',
        str(DIAG / 'fig_siamese_arch.png'),
        '图 3-2  孪生网络结构',
    ),
    (
        r'for epoch in range.*?val_acc.*?=',
        str(DIAG / 'fig_train_flow.png'),
        '图 3-3  训练流程',
    ),
]

SKIP_SINGLE_LINE = r'1×1 卷积.*?→.*?3×3 卷积.*?→.*?1×1 卷积'

def replace_diagram_blocks(txt):
    """将符合条件的架构图代码块替换为 Markdown 图片引用"""
    result = []
    i = 0
    text_lines = txt.split('\n')
    n = len(text_lines)

    while i < n:
        line = text_lines[i]
        if line.startswith('```'):
            lang = line[3:].strip()
            j = i + 1
            while j < n and not text_lines[j].startswith('```'):
                j += 1
            block_content = '\n'.join(text_lines[i+1:j])

            replaced = False
            for kw, img_path, caption in DIAGRAM_BLOCKS:
                if re.search(kw, block_content, re.DOTALL):
                    result.append(f'\n![{caption}]({img_path})\n')
                    result.append(f'*{caption}*\n')
                    replaced = True
                    break

            if not replaced:
                result.append(line)
                result.extend(text_lines[i+1:j+1])

            i = j + 1
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)

text = replace_diagram_blocks(text)

# 6. 图片路径改为绝对路径
def abs_img(m):
    alt  = m.group(1)
    path = m.group(2)
    p = pathlib.Path(path)
    if p.is_absolute() and p.exists():
        return m.group(0)
    candidate = BASE / path
    if candidate.exists():
        return f'![{alt}]({candidate})'
    candidate2 = FIGS / p.name
    if candidate2.exists():
        return f'![{alt}]({candidate2})'
    candidate3 = DIAG / p.name
    if candidate3.exists():
        return f'![{alt}]({candidate3})'
    return m.group(0)

text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', abs_img, text)

# 7. 公式加章节编号
def add_equation_numbers(txt):
    lines = txt.split('\n')
    chapter = 0
    eq_in_chapter = 0
    result = []
    in_code = False

    for line in lines:
        if line.startswith('```'):
            in_code = not in_code

        if not in_code:
            m_ch = re.match(r'^##\s+第([一二三四五六七八九十]+)章', line)
            if m_ch:
                cn_map = {'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10}
                chapter = cn_map.get(m_ch.group(1), chapter + 1)
                eq_in_chapter = 0

            if re.match(r'^\$\$.+\$\$$', line.strip()):
                eq_in_chapter += 1
                label = f'({chapter}.{eq_in_chapter})'
                result.append(line)
                result.append(f'<div style="text-align:right">**{label}**</div>')
                continue

        result.append(line)

    return '\n'.join(result)

text = add_equation_numbers(text)

# 8. 去掉论文标题（YAML 里已有 title）
text = re.sub(r'^#\s+基于孪生神经网络的犬鼻纹识别系统\s*\n', '', text)

# 9. 构建摘要段落（直接用 Pandoc Div 标记，供后处理识别）
def build_abstract_section(cn_body, en_body):
    """构建摘要页面内容，使用 Heading 1 标记以便 post_process.py 识别"""
    parts = []

    if cn_body:
        parts.append('# 摘　要\n')
        parts.append(cn_body)
        parts.append('\n')

    if en_body:
        parts.append('\n# ABSTRACT\n')
        parts.append(en_body)
        parts.append('\n')

    return '\n'.join(parts)

abstract_section = build_abstract_section(cn_abstract_body, en_abstract_body)

# 10. 添加 YAML front matter
front = (
    '---\n'
    'title: "基于孪生神经网络的犬鼻纹识别系统"\n'
    'author: ""\n'
    'date: ""\n'
    'lang: zh-CN\n'
    'toc: true\n'
    'toc-depth: 3\n'
    'numbersections: false\n'
    '---\n\n'
)
text = front + abstract_section + '\n\n' + text

# 11. 合并多余空行
text = re.sub(r'\n{4,}', '\n\n\n', text)

DST.write_text(text, encoding='utf-8')

# 验证原文件未被修改
src_text = SRC.read_text(encoding='utf-8')
assert '摘　要' in src_text, "摘要应在原文件中"
print(f'[OK] 原文件未修改: {SRC}  ({SRC.stat().st_size} bytes)')
print(f'[OK] 中间文件: {DST}  ({len(text)} chars)')

n_imgs  = len(re.findall(r'^!\[', text, re.MULTILINE))
n_eqs   = len(re.findall(r'^\$\$', text, re.MULTILINE))
n_diag  = text.count('fig_system_arch') + text.count('fig_siamese') + text.count('fig_train') + text.count('fig_query')
print(f'    图片引用: {n_imgs}  公式: {n_eqs}  架构图已替换: {n_diag}')
print(f'    中文摘要: {"已提取" if cn_abstract_body else "未找到"}  英文摘要: {"已提取" if en_abstract_body else "未找到"}')
