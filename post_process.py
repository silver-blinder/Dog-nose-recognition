"""
post_process.py — 对 Pandoc 生成的 论文.docx 进行后处理：
1. 删除所有 Heading 段落里 Pandoc 强加的 w:numPr（去掉标题前的点）
2. 修复所有表格的列宽（均分页面宽度）
3. 将公式编号段落合并到公式段落同行（右对齐 tab）
"""
import zipfile, shutil, re, copy, os
from pathlib import Path
from lxml import etree

SRC = Path('/Users/rorschach/Workspace/personal/dog-nose-recognition/论文.docx')
DST = Path('/Users/rorschach/Workspace/personal/dog-nose-recognition/论文.docx')
TMP = Path('/Users/rorschach/Workspace/personal/dog-nose-recognition/论文_tmp.docx')

shutil.copy2(SRC, TMP)

W  = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
ns = {'w': W}

def qn(tag):
    return f'{{{W}}}{tag}'


# ── 解压 docx ──────────────────────────────────────────────────
work = Path('/tmp/docx_work')
if work.exists():
    shutil.rmtree(work)
with zipfile.ZipFile(TMP, 'r') as z:
    z.extractall(str(work))

doc_path = Path('/tmp/docx_work/word/document.xml')
nb_path  = Path('/tmp/docx_work/word/numbering.xml')
tree = etree.parse(str(doc_path))
root = tree.getroot()

# ══════════════════════════════════════════════════════════════
# 问题1-A：修复 numbering.xml —— 把 abstractNum 990 (Pandoc heading
# bullet list, lvlText=" " Wingdings 点) 的 numFmt 全部改为 none
# ══════════════════════════════════════════════════════════════
if nb_path.exists():
    nb_tree = etree.parse(str(nb_path))
    nb_root = nb_tree.getroot()
    fixed_nb = 0
    for ab in nb_root.findall('w:abstractNum', ns):
        aid = ab.get(qn('abstractNumId'))
        if aid == '990':   # Pandoc 的 heading bullet list
            for lvl in ab.findall('w:lvl', ns):
                numFmt = lvl.find('w:numFmt', ns)
                if numFmt is not None:
                    numFmt.set(qn('val'), 'none')
                lvlText = lvl.find('w:lvlText', ns)
                if lvlText is not None:
                    lvlText.set(qn('val'), '')
                fixed_nb += 1
    nb_tree.write(str(nb_path), xml_declaration=True,
                  encoding='UTF-8', standalone=True)
    print(f'[1A] numbering.xml: abstractNum 990 lvl 已改为 none: {fixed_nb} 处')

# ══════════════════════════════════════════════════════════════
# 问题1：删除 Heading 段落里的 w:numPr（去掉标题前的 bullet 点）
# Pandoc 在 document.xml 每个 Heading 段落的 pPr 里写了 numPr，
# 但样式定义里没有，Word 看到后会套用 numbering.xml 的 bullet。
# ══════════════════════════════════════════════════════════════
heading_styles = {'Heading1', 'Heading2', 'Heading3', 'Heading4',
                  'Heading5', 'Heading6', 'Heading7', 'Heading8', 'Heading9'}
removed_num = 0

for p in root.findall('.//w:p', ns):
    pPr = p.find('w:pPr', ns)
    if pPr is None:
        continue
    pStyle = pPr.find('w:pStyle', ns)
    if pStyle is None:
        continue
    sval = pStyle.get(qn('val'), '')
    if sval in heading_styles:
        numPr = pPr.find('w:numPr', ns)
        if numPr is not None:
            pPr.remove(numPr)
            removed_num += 1
        # 同时确保 ind 里没有 hanging 缩进（去掉点留下的缩进）
        ind = pPr.find('w:ind', ns)
        if ind is not None:
            for attr in ('w:hanging', 'w:left'):
                qattr = qn(attr.split(':')[1])
                if qattr in ind.attrib:
                    ind.attrib.pop(qattr)
            ind.set(qn('firstLine'), '0')
            ind.set(qn('left'), '0')

print(f'[1B] 已删除 Heading numPr (body): {removed_num} 处')

# ══════════════════════════════════════════════════════════════
# 问题2：公式编号合并到同一段落（右对齐 tab）
# 预处理时，公式后有一个 <div style="text-align:right">**(n.m)**</div>
# Pandoc 把它转成了单独段落。找到连续的"公式段 + 编号段"，合并。
# 公式段：段落中含 w:oMath 或纯 $$ 转换的 OMML
# 编号段：段落文字匹配 ^\(\d+\.\d+\)$
# ══════════════════════════════════════════════════════════════
body = root.find('.//w:body', ns)
paragraphs = list(body)

# 页面宽度 = 12240 twips（A4-margin后约6inch），用于 right tab
PAGE_WIDTH_TWIPS = 8480  # 实际内容区宽度（A4 21cm - 左3cm - 右2.5cm ≈ 15.5cm = 8789 twips）

merged_eq = 0
i = 0
while i < len(paragraphs) - 1:
    p_cur  = paragraphs[i]
    p_next = paragraphs[i + 1]

    # 判断 p_next 是否是编号段（文字形如 (2.1) 或 **(2.1)**）
    if p_next.tag != qn('p'):
        i += 1
        continue

    next_text = ''.join(r.text or '' for r in p_next.findall('.//w:t', ns)).strip()
    # 匹配形如 (2.1) 或 (2.1) 加粗
    eq_num_match = re.match(r'^\((\d+)\.(\d+)\)$', next_text)
    if not eq_num_match:
        i += 1
        continue

    # 当前段落是否含数学公式（OMML oMath）
    oMath_ns = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
    has_math = bool(p_cur.findall(f'.//{{{oMath_ns}}}oMath'))

    if not has_math:
        i += 1
        continue

    # 在当前段落末尾追加：右对齐 tab + 编号文本
    # 方法：在 pPr 添加右对齐 tab stop，然后追加 \t + 编号 run

    # 1) 添加 tab stop 到 pPr
    pPr = p_cur.find('w:pPr', ns)
    if pPr is None:
        pPr = etree.SubElement(p_cur, qn('pPr'))
        p_cur.insert(0, pPr)

    tabs_el = pPr.find('w:tabs', ns)
    if tabs_el is None:
        tabs_el = etree.SubElement(pPr, qn('tabs'))

    tab_stop = etree.SubElement(tabs_el, qn('tab'))
    tab_stop.set(qn('val'), 'right')
    tab_stop.set(qn('pos'), str(PAGE_WIDTH_TWIPS))

    # 2) 追加 tab run
    tab_run = etree.SubElement(p_cur, qn('r'))
    tab_t   = etree.SubElement(tab_run, qn('tab'))

    # 3) 追加编号 run
    num_run = etree.SubElement(p_cur, qn('r'))
    rPr_num = etree.SubElement(num_run, qn('rPr'))
    # 与公式字号一致，加粗
    b_el = etree.SubElement(rPr_num, qn('b'))
    sz_el = etree.SubElement(rPr_num, qn('sz'))
    sz_el.set(qn('val'), '24')  # 12pt
    t_el  = etree.SubElement(num_run, qn('t'))
    t_el.text = f'({eq_num_match.group(1)}.{eq_num_match.group(2)})'

    # 4) 删除 p_next（编号段）
    body.remove(p_next)
    paragraphs = list(body)  # 刷新列表

    merged_eq += 1
    # i 不递增，继续检查当前位置（可能有连续公式）

print(f'[2] 已合并公式编号: {merged_eq} 处')

# ══════════════════════════════════════════════════════════════
# 问题3：修复表格列宽（均分）
# ══════════════════════════════════════════════════════════════
# A4内容区宽度约 8789 twips（15.5cm）
CONTENT_WIDTH = 8789
fixed_tables = 0

for tbl in root.findall('.//w:tbl', ns):
    # 获取列数（从第一行单元格数）
    first_row = tbl.find('w:tr', ns)
    if first_row is None:
        continue
    cells = first_row.findall('w:tc', ns)
    if not cells:
        continue
    ncols = len(cells)
    col_w = CONTENT_WIDTH // ncols

    # 设置 tblPr 中的 tblW（表格总宽）
    tblPr = tbl.find('w:tblPr', ns)
    if tblPr is None:
        tblPr = etree.SubElement(tbl, qn('tblPr'))
        tbl.insert(0, tblPr)

    tblW = tblPr.find('w:tblW', ns)
    if tblW is None:
        tblW = etree.SubElement(tblPr, qn('tblW'))
    tblW.set(qn('w'),    str(CONTENT_WIDTH))
    tblW.set(qn('type'), 'dxa')

    # 设置 tblGrid（列宽定义）
    tblGrid = tbl.find('w:tblGrid', ns)
    if tblGrid is None:
        tblGrid = etree.Element(qn('tblGrid'))
        tbl.insert(1, tblGrid)
    else:
        for gc in list(tblGrid):
            tblGrid.remove(gc)
    for _ in range(ncols):
        gc = etree.SubElement(tblGrid, qn('gridCol'))
        gc.set(qn('w'), str(col_w))

    # 为每个单元格设置 tcW
    for row in tbl.findall('w:tr', ns):
        row_cells = row.findall('w:tc', ns)
        for tc in row_cells:
            tcPr = tc.find('w:tcPr', ns)
            if tcPr is None:
                tcPr = etree.SubElement(tc, qn('tcPr'))
                tc.insert(0, tcPr)
            tcW = tcPr.find('w:tcW', ns)
            if tcW is None:
                tcW = etree.SubElement(tcPr, qn('tcW'))
            tcW.set(qn('w'),    str(col_w))
            tcW.set(qn('type'), 'dxa')

    # 设置表格边框（确保显示边框）
    tblBorders = tblPr.find('w:tblBorders', ns)
    if tblBorders is None:
        tblBorders = etree.SubElement(tblPr, qn('tblBorders'))
    for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = tblBorders.find(f'w:{side}', ns)
        if el is None:
            el = etree.SubElement(tblBorders, qn(side))
        el.set(qn('val'),   'single')
        el.set(qn('sz'),    '4')
        el.set(qn('space'), '0')
        el.set(qn('color'), '000000')

    # 设置单元格内文字垂直居中 + 正常字号
    for row in tbl.findall('w:tr', ns):
        for tc in row.findall('w:tc', ns):
            tcPr = tc.find('w:tcPr', ns)
            vAlign = tcPr.find('w:vAlign', ns) if tcPr is not None else None
            if tcPr is not None and vAlign is None:
                vAlign = etree.SubElement(tcPr, qn('vAlign'))
                vAlign.set(qn('val'), 'center')

    fixed_tables += 1

print(f'[3] 已修复表格列宽: {fixed_tables} 张')

# ── 保存 ───────────────────────────────────────────────────────
tree.write(str(doc_path), xml_declaration=True,
           encoding='UTF-8', standalone=True)

# 重新打包为 docx
DST.unlink(missing_ok=True)
with zipfile.ZipFile(str(DST), 'w', compression=zipfile.ZIP_DEFLATED) as zout:
    work = Path('/tmp/docx_work')
    for f in work.rglob('*'):
        if f.is_file():
            zout.write(f, f.relative_to(work))

TMP.unlink(missing_ok=True)
print(f'[OK] 输出: {DST}  ({DST.stat().st_size // 1024} KB)')
