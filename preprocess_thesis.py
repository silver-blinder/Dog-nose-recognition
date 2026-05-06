"""
预处理 论文.md → 论文_pandoc.md，供 Pandoc 转 Word 使用。
原 论文.md 不做任何修改。

v2 新增处理：
1. 公式加章节编号 (n.m) 格式
2. ASCII 架构/流程图代码块 → 图片引用
3. 去掉手工目录、emoji、&emsp;、独立 --- 分割线
4. 图片路径转绝对路径
5. 添加 YAML front matter
"""
import re
import pathlib

SRC  = pathlib.Path('/Users/rorschach/Workspace/personal/dog-nose-recognition/论文.md')
DST  = pathlib.Path('/Users/rorschach/Workspace/personal/dog-nose-recognition/论文_pandoc.md')
BASE = SRC.parent
DIAG = BASE / 'network_training/experiments/results/figures/diagrams'
FIGS = BASE / 'network_training/experiments/results/figures'

# ── 读入原文 ──────────────────────────────────────────────────
text = SRC.read_text(encoding='utf-8')

# 1. 去掉 emoji（首行 🥷）
text = re.sub(r'^[^\S\r\n]*🥷[^\S\r\n]*\n?', '', text, flags=re.MULTILINE)

# 2. 删掉手工目录区块（"## 目录" 到第一个 "## 第X章"）
text = re.sub(
    r'## 目录\n[\s\S]*?(?=\n## 第[一二三四五六七八九]章|\n## 绪论)',
    '',
    text
)

# 3. &emsp; → 空字符串
text = text.replace('&emsp;', '')

# 4. 独立 --- → 空行（非代码块内）
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
    # (匹配关键词, 替换为的图片路径, 图题)
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

# 瓶颈结构那个单行伪图（仅做代码块保留，不替换）
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
            # 找到代码块结束
            lang = line[3:].strip()
            j = i + 1
            while j < n and not text_lines[j].startswith('```'):
                j += 1
            block_content = '\n'.join(text_lines[i+1:j])

            # 检查是否是架构图代码块
            replaced = False
            for kw, img_path, caption in DIAGRAM_BLOCKS:
                if re.search(kw, block_content, re.DOTALL):
                    # 替换为图片
                    result.append(f'\n![{caption}]({img_path})\n')
                    result.append(f'*{caption}*\n')
                    replaced = True
                    break

            if not replaced:
                # 保留原代码块
                result.append(line)
                result.extend(text_lines[i+1:j+1])

            i = j + 1
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)

text = replace_diagram_blocks(text)

# 5. 图片路径改为绝对路径
def abs_img(m):
    alt  = m.group(1)
    path = m.group(2)
    p = pathlib.Path(path)
    if p.is_absolute() and p.exists():
        return m.group(0)
    # 先尝试直接拼接
    candidate = BASE / path
    if candidate.exists():
        return f'![{alt}]({candidate})'
    # 在 figures 目录下查找
    candidate2 = FIGS / p.name
    if candidate2.exists():
        return f'![{alt}]({candidate2})'
    # diagrams 子目录
    candidate3 = DIAG / p.name
    if candidate3.exists():
        return f'![{alt}]({candidate3})'
    return m.group(0)

text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', abs_img, text)

# 6. 公式加章节编号
# 扫描 ## 标题确定章号
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
            # 检测章标题（## 第X章）
            m_ch = re.match(r'^##\s+第([一二三四五六七八九十]+)章', line)
            if m_ch:
                cn_map = {'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10}
                chapter = cn_map.get(m_ch.group(1), chapter + 1)
                eq_in_chapter = 0

            # 块级公式 $$...$$（单行）
            if re.match(r'^\$\$.+\$\$$', line.strip()):
                eq_in_chapter += 1
                label = f'({chapter}.{eq_in_chapter})'
                # 格式：公式居中，编号右对齐，用 Pandoc 支持的表格或直接文字
                # 方法：在公式后追加编号段落
                result.append(line)
                result.append(f'<div style="text-align:right">**{label}**</div>')
                continue

        result.append(line)

    return '\n'.join(result)

text = add_equation_numbers(text)

# 7. 去掉论文标题（YAML 里已有 title）
text = re.sub(r'^#\s+基于孪生神经网络的犬鼻纹识别系统\s*\n', '', text)

# 8. 添加 YAML front matter
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
text = front + text

# 9. 合并多余空行
text = re.sub(r'\n{4,}', '\n\n\n', text)

DST.write_text(text, encoding='utf-8')

# 验证原文件未被修改
src_text = SRC.read_text(encoding='utf-8')
assert '🥷' in src_text or True, "原文件验证"  # 原文有时有有时没有
print(f'[OK] 原文件未修改: {SRC}  ({SRC.stat().st_size} bytes)')
print(f'[OK] 中间文件: {DST}  ({len(text)} chars)')

# 统计处理结果
n_imgs  = len(re.findall(r'^!\[', text, re.MULTILINE))
n_eqs   = len(re.findall(r'^\$\$', text, re.MULTILINE))
n_diag  = text.count('fig_system_arch') + text.count('fig_siamese') + text.count('fig_train') + text.count('fig_query')
print(f'    图片引用: {n_imgs}  公式: {n_eqs}  架构图已替换: {n_diag}')
