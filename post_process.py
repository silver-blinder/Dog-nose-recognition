"""
post_process.py — 对 Pandoc 生成的 论文.docx 进行后处理。

styles.xml 中的样式 ID（压缩名）与 document.xml 中 pStyle.val 一致：
  '1'   = Heading 1（章标题：居中、小三号15pt、黑体加粗）
  '21'  = Heading 2（节标题：顶格、四号14pt、黑体加粗）
  '31'  = Heading 3（小节：顶格、小四号12pt、黑体加粗，非斜体）
  '4'   = Heading 4（四级：顶格、小四号12pt、黑体加粗）
  'af'  = Body Text（正文段落）
  'TOC' = TOC Heading（目录标题）
  'TOC1'= toc 1，'TOC2'= toc 2，'TOC3'= toc 3

修复项：
1. Heading 1/2/3/4：直接覆盖 pPr + rPr，强制消除斜体/颜色/字号继承
2. 公式：所有含 oMath 的段落居中 + 清零缩进；有独立编号段的合并编号
3. 表格：宋体五号，垂直居中，全边框
4. 图题（ImageCaption）：宋体五号，居中
5. 摘要标题 + 关键词行
6. TOC 样式：目录标题"目　录"三号黑体居中，TOC2四号宋体，TOC3小四宋体
7. 页码：修改现有 footer1.xml（空）和 footer2.xml（PAGE域），
         删除 body 中已有的中间分节符，在第一章前重新插入正确的分节符
"""
import zipfile, shutil, re
from pathlib import Path
from lxml import etree

BASE = Path('/Users/songjiayi4/Workspace/Dog-nose-recognition')
SRC = BASE / '论文.docx'
DST = BASE / '论文.docx'
TMP = BASE / '论文_tmp.docx'

shutil.copy2(SRC, TMP)

W   = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
ns  = {'w': W}

def qn(tag):
    return f'{{{W}}}{tag}'

def rqn(tag):
    return f'{{{R_NS}}}{tag}'

# ── 通用 pPr 操作 ──────────────────────────────────────────────
def ensure_pPr(p):
    pPr = p.find('w:pPr', ns)
    if pPr is None:
        pPr = etree.Element(qn('pPr'))
        p.insert(0, pPr)
    return pPr

def set_pPr_align(pPr, val):
    for old in pPr.findall('w:jc', ns):
        pPr.remove(old)
    jc = etree.SubElement(pPr, qn('jc'))
    jc.set(qn('val'), val)

def set_pPr_ind(pPr, first_line=0, left=0):
    for old in pPr.findall('w:ind', ns):
        pPr.remove(old)
    ind = etree.SubElement(pPr, qn('ind'))
    ind.set(qn('firstLine'), str(first_line))
    ind.set(qn('left'), str(left))

def set_pPr_spacing(pPr, before=0, after=0, line=300, rule='auto'):
    for old in pPr.findall('w:spacing', ns):
        pPr.remove(old)
    spc = etree.SubElement(pPr, qn('spacing'))
    spc.set(qn('before'),   str(before))
    spc.set(qn('after'),    str(after))
    spc.set(qn('line'),     str(line))
    spc.set(qn('lineRule'), rule)

def remove_page_break_before(pPr):
    for el in pPr.findall('w:pageBreakBefore', ns):
        pPr.remove(el)

def add_page_break_before(pPr):
    if not pPr.findall('w:pageBreakBefore', ns):
        etree.SubElement(pPr, qn('pageBreakBefore'))

# ── 通用 rPr 操作 ──────────────────────────────────────────────
def ensure_rPr(run):
    rPr = run.find('w:rPr', ns)
    if rPr is None:
        rPr = etree.Element(qn('rPr'))
        run.insert(0, rPr)
    return rPr

def set_run_font(rPr, cn='宋体', en='Times New Roman'):
    for old in rPr.findall('w:rFonts', ns):
        rPr.remove(old)
    rf = etree.Element(qn('rFonts'))
    rf.set(qn('ascii'),    en)
    rf.set(qn('hAnsi'),    en)
    rf.set(qn('eastAsia'), cn)
    rf.set(qn('cs'),       en)
    rPr.insert(0, rf)

def set_run_sz(rPr, pt):
    half = str(int(pt * 2))
    for tag in ('w:sz', 'w:szCs'):
        for old in rPr.findall(tag, ns):
            rPr.remove(old)
        el = etree.SubElement(rPr, qn(tag.split(':')[1]))
        el.set(qn('val'), half)

def set_run_bold(rPr, bold=True):
    for tag in ('w:b', 'w:bCs'):
        for old in rPr.findall(tag, ns):
            rPr.remove(old)
    if bold:
        etree.SubElement(rPr, qn('b'))
        etree.SubElement(rPr, qn('bCs'))

def set_run_italic(rPr, italic=False):
    """强制设置或消除斜体（w:i / w:iCs）"""
    for tag in ('w:i', 'w:iCs'):
        for old in rPr.findall(tag, ns):
            rPr.remove(old)
    if italic:
        etree.SubElement(rPr, qn('i'))
        etree.SubElement(rPr, qn('iCs'))
    else:
        # 写 <w:i w:val="false"/> 强制关闭继承的斜体
        i_el = etree.SubElement(rPr, qn('i'))
        i_el.set(qn('val'), 'false')
        ics_el = etree.SubElement(rPr, qn('iCs'))
        ics_el.set(qn('val'), 'false')

def set_run_color_black(rPr):
    for old in rPr.findall('w:color', ns):
        rPr.remove(old)
    c = etree.SubElement(rPr, qn('color'))
    c.set(qn('val'), '000000')

def clear_run_theme_color(rPr):
    """删除主题色标记，避免继承蓝色等主题颜色"""
    for tag in ('w:rStyle', ):
        for old in rPr.findall(f'w:{tag.split(":")[-1]}', ns):
            rPr.remove(old)


# ── 解压 docx ──────────────────────────────────────────────────
work = Path('/tmp/docx_work')
if work.exists():
    shutil.rmtree(work)
with zipfile.ZipFile(TMP, 'r') as z:
    z.extractall(str(work))

doc_path = work / 'word' / 'document.xml'
nb_path  = work / 'word' / 'numbering.xml'
tree = etree.parse(str(doc_path))
root = tree.getroot()
body = root.find('.//w:body', ns)


# ══════════════════════════════════════════════════════════════
# Step 0：修复 numbering.xml（防止标题带自动编号）
# ══════════════════════════════════════════════════════════════
if nb_path.exists():
    nb_tree = etree.parse(str(nb_path))
    nb_root = nb_tree.getroot()
    fixed_nb = 0
    for ab in nb_root.findall('w:abstractNum', ns):
        aid = ab.get(qn('abstractNumId'))
        if aid == '990':
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
    print(f'[0] numbering.xml 修复: {fixed_nb} 处')


# ══════════════════════════════════════════════════════════════
# Step 1：强制修复所有 Heading 段落样式
#
# 注意：document.xml 中 pStyle val 使用压缩名（与 styles.xml 的 styleId 一致）
#   '1'  = Heading 1  （章标题）
#   '21' = Heading 2  （节标题）
#   '31' = Heading 3  （小节，需要非斜体！）
#   '4'  = Heading 4  （四级标题）
#
# 对每个 run 必须显式写 <w:i w:val="false"/> 覆盖 styles.xml 的斜体继承
# ══════════════════════════════════════════════════════════════
HEADING_CFG = {
    'Heading1': dict(align='center', pt=15, cn='黑体', en='Times New Roman', bold=True,
                     before=480, after=120, line=300, page_break=True,  first_line=0, left=0),
    'Heading2': dict(align='left',   pt=14, cn='黑体', en='Times New Roman', bold=True,
                     before=240, after=120, line=300, page_break=False, first_line=0, left=0),
    'Heading3': dict(align='left',   pt=12, cn='黑体', en='Times New Roman', bold=True,
                     before=120, after=60,  line=300, page_break=False, first_line=0, left=0),
    'Heading4': dict(align='left',   pt=12, cn='黑体', en='Times New Roman', bold=True,
                     before=60,  after=60,  line=300, page_break=False, first_line=0, left=0),
}

fixed_headings = 0
for p in body.findall('.//w:p', ns):
    pPr = p.find('w:pPr', ns)
    if pPr is None:
        continue
    pStyle = pPr.find('w:pStyle', ns)
    if pStyle is None:
        continue
    sval = pStyle.get(qn('val'), '')
    cfg = HEADING_CFG.get(sval)
    if cfg is None:
        continue

    # ── pPr：删除 numPr，设置对齐/缩进/间距/分页 ──
    for numPr in pPr.findall('w:numPr', ns):
        pPr.remove(numPr)
    set_pPr_align(pPr, cfg['align'])
    set_pPr_ind(pPr, first_line=cfg['first_line'], left=cfg['left'])
    set_pPr_spacing(pPr, before=cfg['before'], after=cfg['after'], line=cfg['line'])
    remove_page_break_before(pPr)
    if cfg['page_break']:
        add_page_break_before(pPr)

    # ── rPr：强制覆盖字体/字号/粗体/斜体/颜色 ──
    # 同时处理段落级 rPr（pPr 下的 rPr，控制段落默认字符格式）
    p_rPr = pPr.find('w:rPr', ns)
    if p_rPr is None:
        p_rPr = etree.SubElement(pPr, qn('rPr'))
    set_run_font(p_rPr, cn=cfg['cn'], en=cfg['en'])
    set_run_sz(p_rPr, cfg['pt'])
    set_run_bold(p_rPr, cfg['bold'])
    set_run_italic(p_rPr, italic=False)
    set_run_color_black(p_rPr)

    # 每个 run 的字符格式也强制覆盖
    for r in p.findall('w:r', ns):
        rPr = ensure_rPr(r)
        set_run_font(rPr, cn=cfg['cn'], en=cfg['en'])
        set_run_sz(rPr, cfg['pt'])
        set_run_bold(rPr, cfg['bold'])
        set_run_italic(rPr, italic=False)
        set_run_color_black(rPr)

    fixed_headings += 1

print(f'[1] Heading 样式强制修复: {fixed_headings} 处')


# ══════════════════════════════════════════════════════════════
# Step 2：公式处理
#   2a. 所有含 oMath 的段落：居中 + 清零缩进（firstLine=0, left=0）
#   2b. 有独立编号段（下一段为 "(n.m)" 格式）的公式：合并编号到右侧
# ══════════════════════════════════════════════════════════════
oMath_ns_uri = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
PAGE_WIDTH_TWIPS = 8789  # A4 页面内容区宽度（twips）

# 2a：先对所有含 oMath 的段落设置居中和零缩进
centered_eq = 0
for p in body.iter(qn('p')):
    if not p.findall(f'.//{{{oMath_ns_uri}}}oMath'):
        continue
    pPr = ensure_pPr(p)
    set_pPr_align(pPr, 'center')
    set_pPr_ind(pPr, first_line=0, left=0)
    centered_eq += 1

print(f'[2a] 公式段落居中: {centered_eq} 处')

# 2b：合并独立编号段
paragraphs = list(body)
merged_eq = 0
i = 0
while i < len(paragraphs) - 1:
    p_cur  = paragraphs[i]
    p_next = paragraphs[i + 1]

    if p_next.tag != qn('p'):
        i += 1
        continue

    next_text = ''.join(r.text or '' for r in p_next.findall('.//w:t', ns)).strip()
    eq_num_match = re.match(r'^\((\d+)\.(\d+)\)$', next_text)
    if not eq_num_match:
        i += 1
        continue

    has_math = bool(p_cur.findall(f'.//{{{oMath_ns_uri}}}oMath'))
    if not has_math:
        i += 1
        continue

    # 添加右对齐 tab stop
    pPr = ensure_pPr(p_cur)
    tabs_el = pPr.find('w:tabs', ns)
    if tabs_el is None:
        tabs_el = etree.SubElement(pPr, qn('tabs'))
    for old_tab in list(tabs_el):
        tabs_el.remove(old_tab)
    tab_stop = etree.SubElement(tabs_el, qn('tab'))
    tab_stop.set(qn('val'), 'right')
    tab_stop.set(qn('pos'), str(PAGE_WIDTH_TWIPS))

    # 追加 tab run + 编号 run
    tab_run = etree.SubElement(p_cur, qn('r'))
    etree.SubElement(tab_run, qn('tab'))

    num_run = etree.SubElement(p_cur, qn('r'))
    rPr_num = etree.SubElement(num_run, qn('rPr'))
    etree.SubElement(rPr_num, qn('b'))
    etree.SubElement(rPr_num, qn('bCs'))
    for sz_tag in ('sz', 'szCs'):
        sz_el = etree.SubElement(rPr_num, qn(sz_tag))
        sz_el.set(qn('val'), '24')
    t_el = etree.SubElement(num_run, qn('t'))
    t_el.text = f'({eq_num_match.group(1)}.{eq_num_match.group(2)})'

    body.remove(p_next)
    paragraphs = list(body)
    merged_eq += 1

print(f'[2b] 公式编号合并: {merged_eq} 处')


# ══════════════════════════════════════════════════════════════
# Step 3：表格样式（宋体五号，垂直居中，全边框）
# ══════════════════════════════════════════════════════════════
CONTENT_WIDTH = 8789
fixed_tables = 0

for tbl in root.findall('.//w:tbl', ns):
    first_row = tbl.find('w:tr', ns)
    if first_row is None:
        continue
    cells = first_row.findall('w:tc', ns)
    if not cells:
        continue
    ncols  = len(cells)
    col_w  = CONTENT_WIDTH // ncols

    tblPr = tbl.find('w:tblPr', ns)
    if tblPr is None:
        tblPr = etree.Element(qn('tblPr'))
        tbl.insert(0, tblPr)

    tblW = tblPr.find('w:tblW', ns)
    if tblW is None:
        tblW = etree.SubElement(tblPr, qn('tblW'))
    tblW.set(qn('w'),    str(CONTENT_WIDTH))
    tblW.set(qn('type'), 'dxa')

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

    for row in tbl.findall('w:tr', ns):
        for tc in row.findall('w:tc', ns):
            tcPr = tc.find('w:tcPr', ns)
            if tcPr is None:
                tcPr = etree.Element(qn('tcPr'))
                tc.insert(0, tcPr)
            tcW = tcPr.find('w:tcW', ns)
            if tcW is None:
                tcW = etree.SubElement(tcPr, qn('tcW'))
            tcW.set(qn('w'),    str(col_w))
            tcW.set(qn('type'), 'dxa')
            vAlign = tcPr.find('w:vAlign', ns)
            if vAlign is None:
                vAlign = etree.SubElement(tcPr, qn('vAlign'))
            vAlign.set(qn('val'), 'center')

            for p in tc.findall('w:p', ns):
                p_pPr = ensure_pPr(p)
                set_pPr_align(p_pPr, 'center')
                set_pPr_ind(p_pPr, 0, 0)
                set_pPr_spacing(p_pPr, 0, 0, 300)
                for r in p.findall('w:r', ns):
                    rPr = ensure_rPr(r)
                    set_run_font(rPr, cn='宋体', en='Times New Roman')
                    set_run_sz(rPr, 10.5)

    fixed_tables += 1

print(f'[3] 表格样式修复: {fixed_tables} 张')


# ══════════════════════════════════════════════════════════════
# Step 4：图题样式（宋体五号居中，段后空一行）
# ══════════════════════════════════════════════════════════════
fixed_captions = 0
for p in body.findall('.//w:p', ns):
    pPr = p.find('w:pPr', ns)
    if pPr is None:
        continue
    pStyle = pPr.find('w:pStyle', ns)
    if pStyle is None:
        continue
    if pStyle.get(qn('val'), '') not in ('Caption', 'ImageCaption',
                                          'FigureCaption', 'TableCaption'):
        continue
    set_pPr_align(pPr, 'center')
    set_pPr_ind(pPr, 0, 0)
    set_pPr_spacing(pPr, before=0, after=240, line=300)
    for r in p.findall('w:r', ns):
        rPr = ensure_rPr(r)
        set_run_font(rPr, cn='宋体', en='Times New Roman')
        set_run_sz(rPr, 10.5)
        set_run_bold(rPr, False)
        set_run_italic(rPr, italic=False)
        set_run_color_black(rPr)
    fixed_captions += 1

print(f'[4] 图题样式修复: {fixed_captions} 处')


# ══════════════════════════════════════════════════════════════
# Step 4.5：代码块（SourceCode）美化
#   - 等宽字体 Consolas，9pt，黑色
#   - 灰色背景（shd fill=F5F5F5）
#   - 左侧粗竖线边框（pBdr left sz=18）+ 上下细线
#   - 首行缩进归零，左缩进 560 twips（约 1cm）
# ══════════════════════════════════════════════════════════════
fixed_code = 0
all_paras = list(body.iter(qn('p')))

for p in all_paras:
    pPr = p.find('w:pPr', ns)
    if pPr is None:
        continue
    pStyle = pPr.find('w:pStyle', ns)
    if pStyle is None:
        continue
    if pStyle.get(qn('val'), '') != 'SourceCode':
        continue

    # ── 段落格式 ──
    set_pPr_ind(pPr, first_line=0, left=560)
    set_pPr_spacing(pPr, before=0, after=0, line=240, rule='auto')

    # ── 灰色背景 ──
    for old_shd in pPr.findall('w:shd', ns):
        pPr.remove(old_shd)
    shd = etree.SubElement(pPr, qn('shd'))
    shd.set(qn('val'),   'clear')
    shd.set(qn('color'), 'auto')
    shd.set(qn('fill'),  'F5F5F5')

    # ── 边框（左侧粗竖线 + 上下细线） ──
    for old_bdr in pPr.findall('w:pBdr', ns):
        pPr.remove(old_bdr)
    pBdr = etree.SubElement(pPr, qn('pBdr'))
    for side, sz, space, color in [
        ('top',    4,  1, 'E0E0E0'),
        ('left',  18,  4, '888888'),
        ('bottom', 4,  1, 'E0E0E0'),
    ]:
        el = etree.SubElement(pBdr, qn(side))
        el.set(qn('val'),   'single')
        el.set(qn('sz'),    str(sz))
        el.set(qn('space'), str(space))
        el.set(qn('color'), color)

    # ── run 字体/字号/颜色 ──
    for r in p.findall('w:r', ns):
        rPr = ensure_rPr(r)
        set_run_font(rPr, cn='Consolas', en='Consolas')
        set_run_sz(rPr, 9)
        set_run_bold(rPr, False)
        set_run_italic(rPr, italic=False)
        set_run_color_black(rPr)

    fixed_code += 1

print(f'[4.5] 代码块美化: {fixed_code} 处')


# ══════════════════════════════════════════════════════════════
# Step 5：摘要标题（style='1' 且文字为 '摘　要'/'ABSTRACT'）
# 三号(16pt)，黑体/Times加粗居中，不强制分页
# ══════════════════════════════════════════════════════════════
fixed_abstract = 0
for p in body.findall('.//w:p', ns):
    pPr = p.find('w:pPr', ns)
    if pPr is None:
        continue
    pStyle = pPr.find('w:pStyle', ns)
    if pStyle is None or pStyle.get(qn('val'), '') != 'Heading1':
        continue
    p_text = ''.join(r.text or '' for r in p.findall('.//w:t', ns)).strip()
    if p_text not in ('摘\u3000要', '摘要', 'ABSTRACT'):
        continue

    is_en = (p_text == 'ABSTRACT')
    remove_page_break_before(pPr)
    set_pPr_align(pPr, 'center')
    set_pPr_ind(pPr, 0, 0)
    set_pPr_spacing(pPr, before=0, after=240, line=300)

    for r in p.findall('w:r', ns):
        rPr = ensure_rPr(r)
        cn_font = 'Times New Roman' if is_en else '黑体'
        set_run_font(rPr, cn=cn_font, en='Times New Roman')
        set_run_sz(rPr, 16)
        set_run_bold(rPr, True)
        set_run_italic(rPr, italic=False)
        set_run_color_black(rPr)
    fixed_abstract += 1

print(f'[5] 摘要标题修复: {fixed_abstract} 处')


# ══════════════════════════════════════════════════════════════
# Step 6：关键词行（小四号黑体顶格）
# ══════════════════════════════════════════════════════════════
fixed_kw = 0
for p in body.findall('.//w:p', ns):
    p_text = ''.join(r.text or '' for r in p.findall('.//w:t', ns)).strip()
    is_cn = '关键词' in p_text[:6]
    is_en = p_text.startswith('Key words') or p_text.startswith('Key Words')
    if not (is_cn or is_en):
        continue
    pPr = ensure_pPr(p)
    set_pPr_align(pPr, 'left')
    set_pPr_ind(pPr, 0, 0)
    for r in p.findall('w:r', ns):
        rPr = ensure_rPr(r)
        r_text = ''.join(t.text or '' for t in r.findall('w:t', ns))
        is_label = ('关键词' in r_text or 'Key words' in r_text
                    or 'Key Words' in r_text)
        cn_font = 'Times New Roman' if is_en else ('黑体' if is_label else '宋体')
        set_run_font(rPr, cn=cn_font, en='Times New Roman')
        set_run_sz(rPr, 12)
        set_run_bold(rPr, is_label)
        set_run_italic(rPr, italic=False)
        set_run_color_black(rPr)
    fixed_kw += 1

print(f'[6] 关键词行修复: {fixed_kw} 处')


# ══════════════════════════════════════════════════════════════
# Step 7：目录样式修复
#   TOC Heading（styleId='TOC'）：改文字为"目　录"，三号黑体居中
#   TOC1（styleId='TOC1'）：四号宋体（章标题级别）
#   TOC2（styleId='TOC2'）：四号宋体，缩进约720
#   TOC3（styleId='TOC3'）：小四号宋体，缩进约1440
# ══════════════════════════════════════════════════════════════

# 7a：修改 styles.xml 中 TOC 样式定义
styles_path = work / 'word' / 'styles.xml'
stree = etree.parse(str(styles_path))
sroot = stree.getroot()

TOC_STYLE_CFG = {
    # styles.xml styleId → 格式配置
    'TOCHeading': dict(pt=16, cn='黑体', en='Times New Roman', bold=True,
                       align='center', before=0, after=240, line=300,
                       first_line=0, left=0),
    'TOC1':       dict(pt=14, cn='宋体', en='Times New Roman', bold=False,
                       align='left', before=60, after=0, line=300,
                       first_line=0, left=0),
    'TOC2':       dict(pt=14, cn='宋体', en='Times New Roman', bold=False,
                       align='left', before=60, after=0, line=300,
                       first_line=0, left=720),
    'TOC3':       dict(pt=12, cn='宋体', en='Times New Roman', bold=False,
                       align='left', before=0, after=0, line=300,
                       first_line=0, left=1440),
}

for style in sroot.findall('.//w:style', ns):
    sid = style.get(qn('styleId'), '')
    cfg = TOC_STYLE_CFG.get(sid)
    if cfg is None:
        continue
    # pPr
    pPr_s = style.find('w:pPr', ns)
    if pPr_s is None:
        pPr_s = etree.SubElement(style, qn('pPr'))
    set_pPr_align(pPr_s, cfg['align'])
    set_pPr_ind(pPr_s, cfg['first_line'], cfg['left'])
    set_pPr_spacing(pPr_s, cfg['before'], cfg['after'], cfg['line'])
    # rPr
    rPr_s = style.find('w:rPr', ns)
    if rPr_s is None:
        rPr_s = etree.SubElement(style, qn('rPr'))
    set_run_font(rPr_s, cn=cfg['cn'], en=cfg['en'])
    set_run_sz(rPr_s, cfg['pt'])
    set_run_bold(rPr_s, cfg['bold'])
    set_run_italic(rPr_s, italic=False)
    set_run_color_black(rPr_s)

stree.write(str(styles_path), xml_declaration=True,
            encoding='UTF-8', standalone=True)
print(f'[7] TOC 样式定义已更新到 styles.xml')

# 7b：修改 document.xml 中 TOC Heading 段落的文字为"目　录"
toc_heading_fixed = 0
for p in body.findall('.//w:p', ns):
    pPr = p.find('w:pPr', ns)
    if pPr is None:
        continue
    pStyle = pPr.find('w:pStyle', ns)
    if pStyle is None or pStyle.get(qn('val'), '') != 'TOCHeading':
        continue
    # 修改文字
    for r in p.findall('w:r', ns):
        for t in r.findall('w:t', ns):
            if t.text and ('Contents' in t.text or 'Table' in t.text
                           or 'toc' in t.text.lower()):
                t.text = '目\u3000录'
        rPr = ensure_rPr(r)
        set_run_font(rPr, cn='黑体', en='Times New Roman')
        set_run_sz(rPr, 16)
        set_run_bold(rPr, True)
        set_run_italic(rPr, italic=False)
        set_run_color_black(rPr)
    # 也修正 pPr
    set_pPr_align(pPr, 'center')
    set_pPr_ind(pPr, 0, 0)
    set_pPr_spacing(pPr, 0, 240, 300)
    toc_heading_fixed += 1

# 7c：对 document.xml 中 TOC1/TOC2/TOC3 段落直接覆盖格式
toc_entry_fixed = 0
for p in body.findall('.//w:p', ns):
    pPr = p.find('w:pPr', ns)
    if pPr is None:
        continue
    pStyle = pPr.find('w:pStyle', ns)
    if pStyle is None:
        continue
    sval = pStyle.get(qn('val'), '')
    cfg = TOC_STYLE_CFG.get(sval)
    if cfg is None or sval == 'TOCHeading':
        continue
    set_pPr_align(pPr, cfg['align'])
    set_pPr_ind(pPr, cfg['first_line'], cfg['left'])
    set_pPr_spacing(pPr, cfg['before'], cfg['after'], cfg['line'])
    for r in p.findall('w:r', ns):
        rPr = ensure_rPr(r)
        set_run_font(rPr, cn=cfg['cn'], en=cfg['en'])
        set_run_sz(rPr, cfg['pt'])
        set_run_bold(rPr, cfg['bold'])
        set_run_italic(rPr, italic=False)
        set_run_color_black(rPr)
    toc_entry_fixed += 1

print(f'[7] TOC heading 段落修复: {toc_heading_fixed} 处，目录条目修复: {toc_entry_fixed} 处')

# ── 第一次保存 document.xml ───────────────────────────────────
tree.write(str(doc_path), xml_declaration=True,
           encoding='UTF-8', standalone=True)


# ══════════════════════════════════════════════════════════════
# Step 8：页码修复
#
# 诊断发现：
#   - 文档中已有一个"中间分节符"在 body[16]，引用 rId7→footer1.xml（空）
#   - 第一章在 body[17]
#   - 末尾 sectPr 引用 rId19→footer2.xml（有 PAGE 域）
#   - 目前 footer2.xml 已有"- PAGE -"格式，footer1.xml 为空
#   - 但中间分节符的 pgNumType 没有 fmt 设置，且该节（摘要/目录）
#     的页码计数不正确
#
# 修复方案：
#   1. 直接修改 footer1.xml → 清空（确保摘要/目录节无页码）
#   2. 直接修改 footer2.xml → 确保正文页码格式 "- 页码 -"
#   3. 重新解析 document.xml，删除现有的中间分节符（body[16]），
#      在第一章前重新插入，确保 pgNumType 正确（start=1）
#   4. 末尾 sectPr 确保引用 footer2.xml + pgNumType start=1
# ══════════════════════════════════════════════════════════════

# ── 8a：修改 footer1.xml（空，无页码） ──
footer1_path = work / 'word' / 'footer1.xml'
FOOTER_BLANK_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:p><w:pPr><w:jc w:val="center"/></w:pPr></w:p>
</w:ftr>'''
footer1_path.write_text(FOOTER_BLANK_XML, encoding='utf-8')

# ── 8b：修改 footer2.xml（正文页码 "- N -"） ──
footer2_path = work / 'word' / 'footer2.xml'
FOOTER_MAIN_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:p>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"
                  w:eastAsia="宋体" w:cs="Times New Roman"/>
        <w:sz w:val="24"/><w:szCs w:val="24"/>
      </w:rPr>
    </w:pPr>
    <w:r>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"
                  w:eastAsia="宋体" w:cs="Times New Roman"/>
        <w:sz w:val="24"/><w:szCs w:val="24"/>
      </w:rPr>
      <w:t xml:space="preserve">- </w:t>
    </w:r>
    <w:r>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"
                  w:eastAsia="宋体" w:cs="Times New Roman"/>
        <w:sz w:val="24"/><w:szCs w:val="24"/>
      </w:rPr>
      <w:fldChar w:fldCharType="begin"/>
    </w:r>
    <w:r>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"
                  w:eastAsia="宋体" w:cs="Times New Roman"/>
        <w:sz w:val="24"/><w:szCs w:val="24"/>
      </w:rPr>
      <w:instrText xml:space="preserve"> PAGE </w:instrText>
    </w:r>
    <w:r>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"
                  w:eastAsia="宋体" w:cs="Times New Roman"/>
        <w:sz w:val="24"/><w:szCs w:val="24"/>
      </w:rPr>
      <w:fldChar w:fldCharType="separate"/>
    </w:r>
    <w:r>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"
                  w:eastAsia="宋体" w:cs="Times New Roman"/>
        <w:sz w:val="24"/><w:szCs w:val="24"/>
      </w:rPr>
      <w:t>1</w:t>
    </w:r>
    <w:r>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"
                  w:eastAsia="宋体" w:cs="Times New Roman"/>
        <w:sz w:val="24"/><w:szCs w:val="24"/>
      </w:rPr>
      <w:fldChar w:fldCharType="end"/>
    </w:r>
    <w:r>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"
                  w:eastAsia="宋体" w:cs="Times New Roman"/>
        <w:sz w:val="24"/><w:szCs w:val="24"/>
      </w:rPr>
      <w:t xml:space="preserve"> -</w:t>
    </w:r>
  </w:p>
</w:ftr>'''
footer2_path.write_text(FOOTER_MAIN_XML, encoding='utf-8')
print('[8] footer1.xml（无页码）/ footer2.xml（PAGE域）已更新')

# ── 8c：重新解析 document.xml，修复分节符 ──
tree8 = etree.parse(str(doc_path))
root8 = tree8.getroot()
body8 = root8.find('.//w:body', ns)

# 找到并删除现有的"中间分节符"（pPr内含 sectPr 的空段落）
removed_sect = 0
for p in list(body8):
    if p.tag != qn('p'):
        continue
    pPr8 = p.find('w:pPr', ns)
    if pPr8 is None:
        continue
    inner_sect = pPr8.find('w:sectPr', ns)
    if inner_sect is None:
        continue
    # 是空段落（无实质文字）才删除
    p_text = ''.join(r.text or '' for r in p.findall('.//w:t', ns)).strip()
    if p_text == '':
        body8.remove(p)
        removed_sect += 1
print(f'[8] 删除旧中间分节符: {removed_sect} 个')

# 找第一章（style='21' 且含"第一章"或"绪论"）
paras8 = list(body8)
first_chapter_idx = None
for idx, p in enumerate(paras8):
    if p.tag != qn('p'):
        continue
    pPr8 = p.find('w:pPr', ns)
    if pPr8 is None:
        continue
    pStyle8 = pPr8.find('w:pStyle', ns)
    if pStyle8 is None:
        continue
    if pStyle8.get(qn('val'), '') != 'Heading2':
        continue
    p_text = ''.join(r.text or '' for r in p.findall('.//w:t', ns))
    if '第一章' in p_text or '绪论' in p_text:
        first_chapter_idx = idx
        break

if first_chapter_idx is not None:
    # rId7 = footer1.xml（无页码），rId19 = footer2.xml（有页码）
    BLANK_RID = 'rId7'
    MAIN_RID  = 'rId19'

    # 插入分节段落（前节：摘要+目录，无页码）
    sect_p = etree.Element(qn('p'))
    sect_pPr = etree.SubElement(sect_p, qn('pPr'))
    inner_sectPr = etree.SubElement(sect_pPr, qn('sectPr'))

    fr_blank = etree.SubElement(inner_sectPr, qn('footerReference'))
    fr_blank.set(qn('type'), 'default')
    fr_blank.set(rqn('id'), BLANK_RID)

    # 前节：页码不显示，格式 decimal，无 start（让 Word 跳过页码计数）
    pgNumType_front = etree.SubElement(inner_sectPr, qn('pgNumType'))
    pgNumType_front.set(qn('fmt'), 'decimal')

    # 分节类型 nextPage
    sectType_el = etree.SubElement(inner_sectPr, qn('type'))
    sectType_el.set(qn('val'), 'nextPage')

    body8.insert(first_chapter_idx, sect_p)
    print(f'[8] 分节符已插入（第一章前，索引 {first_chapter_idx}）')

    # 修改末尾 sectPr（正文节）
    final_sectPr = body8.find('w:sectPr', ns)
    if final_sectPr is not None:
        for fr in final_sectPr.findall('w:footerReference', ns):
            final_sectPr.remove(fr)
        fr_main = etree.SubElement(final_sectPr, qn('footerReference'))
        fr_main.set(qn('type'), 'default')
        fr_main.set(rqn('id'), MAIN_RID)
        # 正文页码从 1 开始
        pgNum = final_sectPr.find('w:pgNumType', ns)
        if pgNum is None:
            pgNum = etree.SubElement(final_sectPr, qn('pgNumType'))
        pgNum.set(qn('fmt'),   'decimal')
        pgNum.set(qn('start'), '1')
        print('[8] 末尾 sectPr（正文节）页码设置: start=1')
else:
    print('[8] 未找到第一章，分节符未插入')

tree8.write(str(doc_path), xml_declaration=True,
            encoding='UTF-8', standalone=True)
print('[8] 页码修复完成')


# ── 重新打包 ──────────────────────────────────────────────────
DST.unlink(missing_ok=True)
with zipfile.ZipFile(str(DST), 'w', compression=zipfile.ZIP_DEFLATED) as zout:
    for f in work.rglob('*'):
        if f.is_file():
            zout.write(f, f.relative_to(work))

TMP.unlink(missing_ok=True)
print(f'[OK] 输出: {DST}  ({DST.stat().st_size // 1024} KB)')
