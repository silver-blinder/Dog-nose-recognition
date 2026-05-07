"""
生成 Pandoc reference.docx（同济论文格式）
格式要求：
  Heading 1 - 章标题：居中、小三号(15pt)、黑体、西文 Times New Roman 加粗
  Heading 2 - 节标题(1.1)：顶格、四号(14pt)、黑体、西文 Times New Roman 加粗
  Heading 3 - 小节(1.1.1)：顶格、四号(14pt)、黑体、西文 Times New Roman 加粗
  Normal    - 正文：小四号(12pt)、宋体、首行缩进2字符、1.25倍行距
  目录标题   - 三号(16pt)、黑体、居中
  TOC 1     - 章条目：四号(14pt)、宋体
  TOC 2     - 节条目：小四号(12pt)、宋体、左缩进
  Caption   - 图表题：五号(10.5pt)、宋体、居中（图题在图下，表题在表上）
"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

BASE = '/Users/songjiayi4/Workspace/Dog-nose-recognition'

doc = Document()

# ── 页面设置：A4 ──────────────────────────────────────────────
section = doc.sections[0]
section.page_height = Cm(29.7)
section.page_width  = Cm(21.0)
section.top_margin    = Cm(3.0)
section.bottom_margin = Cm(2.5)
section.left_margin   = Cm(3.0)
section.right_margin  = Cm(2.5)


def force_cn_font(style_obj, cn_font, en_font='Times New Roman'):
    """直接操作 XML 强制设置中英文字体。"""
    s_elem = style_obj.element
    rPr = s_elem.find(qn('w:rPr'))
    if rPr is None:
        rPr = OxmlElement('w:rPr')
        s_elem.append(rPr)

    old = rPr.find(qn('w:rFonts'))
    if old is not None:
        rPr.remove(old)

    rFonts = OxmlElement('w:rFonts')
    rFonts.set(qn('w:ascii'),     en_font)
    rFonts.set(qn('w:hAnsi'),     en_font)
    rFonts.set(qn('w:eastAsia'),  cn_font)
    rFonts.set(qn('w:cs'),        en_font)
    rPr.insert(0, rFonts)


def set_sz(style_obj, pt):
    """设置字号（半磅单位）"""
    s_elem = style_obj.element
    rPr = s_elem.find(qn('w:rPr'))
    if rPr is None:
        rPr = OxmlElement('w:rPr')
        s_elem.append(rPr)
    for tag in ('w:sz', 'w:szCs'):
        old = rPr.find(qn(tag))
        if old is not None:
            rPr.remove(old)
        sz = OxmlElement(tag)
        sz.set(qn('w:val'), str(int(pt * 2)))
        rPr.append(sz)


def set_bold(style_obj, bold=True):
    s_elem = style_obj.element
    rPr = s_elem.find(qn('w:rPr'))
    if rPr is None:
        rPr = OxmlElement('w:rPr')
        s_elem.append(rPr)
    old = rPr.find(qn('w:b'))
    if old is not None:
        rPr.remove(old)
    if bold:
        b = OxmlElement('w:b')
        rPr.append(b)
        bCs = OxmlElement('w:bCs')
        rPr.append(bCs)


def set_color_black(style_obj):
    s_elem = style_obj.element
    rPr = s_elem.find(qn('w:rPr'))
    if rPr is None:
        rPr = OxmlElement('w:rPr')
        s_elem.append(rPr)
    old = rPr.find(qn('w:color'))
    if old is not None:
        rPr.remove(old)
    color = OxmlElement('w:color')
    color.set(qn('w:val'), '000000')
    rPr.append(color)


def set_para_fmt(style_obj,
                 align=None,
                 first_line_cm=None,
                 left_indent_cm=None,
                 space_before_pt=0,
                 space_after_pt=0,
                 line_spacing_lines=1.25,
                 page_break_before=False):
    """设置段落格式（操作 pPr XML）"""
    s_elem = style_obj.element
    pPr = s_elem.find(qn('w:pPr'))
    if pPr is None:
        pPr = OxmlElement('w:pPr')
        s_elem.insert(0, pPr)

    if align is not None:
        jc = pPr.find(qn('w:jc'))
        if jc is not None:
            pPr.remove(jc)
        jc = OxmlElement('w:jc')
        jc.set(qn('w:val'), align)
        pPr.append(jc)

    ind = pPr.find(qn('w:ind'))
    if ind is None:
        ind = OxmlElement('w:ind')
        pPr.append(ind)
    else:
        for a in list(ind.attrib):
            del ind.attrib[a]
    if first_line_cm is not None:
        val = int(first_line_cm * 567)
        ind.set(qn('w:firstLine'), str(val))
    else:
        ind.set(qn('w:firstLine'), '0')
    if left_indent_cm is not None:
        ind.set(qn('w:left'), str(int(left_indent_cm * 567)))
    else:
        ind.set(qn('w:left'), '0')

    spc = pPr.find(qn('w:spacing'))
    if spc is None:
        spc = OxmlElement('w:spacing')
        pPr.append(spc)
    else:
        for a in list(spc.attrib):
            del spc.attrib[a]
    spc.set(qn('w:before'), str(int(space_before_pt * 20)))
    spc.set(qn('w:after'),  str(int(space_after_pt  * 20)))
    spc.set(qn('w:line'),      str(int(line_spacing_lines * 240)))
    spc.set(qn('w:lineRule'), 'auto')

    if page_break_before:
        pbk = pPr.find(qn('w:pageBreakBefore'))
        if pbk is None:
            pbk = OxmlElement('w:pageBreakBefore')
            pPr.append(pbk)

    rPr2 = pPr.find(qn('w:rPr'))
    if rPr2 is None:
        rPr2 = OxmlElement('w:rPr')
        pPr.append(rPr2)


# ════════════════════════════════════════════════════════════════
# Normal 正文：宋体小四，首行缩进2字符，1.25倍行距
# ════════════════════════════════════════════════════════════════
normal = doc.styles['Normal']
force_cn_font(normal, '宋体', 'Times New Roman')
set_sz(normal, 12)   # 小四 = 12pt
set_bold(normal, False)
set_color_black(normal)
set_para_fmt(normal,
             align='both',
             first_line_cm=0.85,
             left_indent_cm=0,
             space_before_pt=0,
             space_after_pt=0,
             line_spacing_lines=1.25)

# ════════════════════════════════════════════════════════════════
# Heading 1 - 章标题：居中、小三(15pt)、黑体加粗、章前分页
# ════════════════════════════════════════════════════════════════
h1 = doc.styles['Heading 1']
force_cn_font(h1, '黑体', 'Times New Roman')
set_sz(h1, 15)
set_bold(h1, True)
set_color_black(h1)
set_para_fmt(h1,
             align='center',
             first_line_cm=None,
             left_indent_cm=0,
             space_before_pt=24,
             space_after_pt=6,
             line_spacing_lines=1.25,
             page_break_before=True)

# ════════════════════════════════════════════════════════════════
# Heading 2 - 节标题(1.1)：顶格、四号(14pt)、黑体加粗
# ════════════════════════════════════════════════════════════════
h2 = doc.styles['Heading 2']
force_cn_font(h2, '黑体', 'Times New Roman')
set_sz(h2, 14)
set_bold(h2, True)
set_color_black(h2)
set_para_fmt(h2,
             align='left',
             first_line_cm=None,
             left_indent_cm=0,
             space_before_pt=12,
             space_after_pt=6,
             line_spacing_lines=1.25)

# ════════════════════════════════════════════════════════════════
# Heading 3 - 小节(1.1.1)：顶格、四号(14pt)、黑体加粗
# ════════════════════════════════════════════════════════════════
h3 = doc.styles['Heading 3']
force_cn_font(h3, '黑体', 'Times New Roman')
set_sz(h3, 14)
set_bold(h3, True)
set_color_black(h3)
set_para_fmt(h3,
             align='left',
             first_line_cm=None,
             left_indent_cm=0,
             space_before_pt=6,
             space_after_pt=3,
             line_spacing_lines=1.25)

# ════════════════════════════════════════════════════════════════
# Heading 4 - 四级(1.1.1.1)：顶格、小四(12pt)、宋体加粗
# ════════════════════════════════════════════════════════════════
h4 = doc.styles['Heading 4']
force_cn_font(h4, '宋体', 'Times New Roman')
set_sz(h4, 12)
set_bold(h4, True)
set_color_black(h4)
set_para_fmt(h4,
             align='left',
             first_line_cm=None,
             left_indent_cm=0,
             space_before_pt=3,
             space_after_pt=3,
             line_spacing_lines=1.25)

# ════════════════════════════════════════════════════════════════
# Caption 图表题：居中、五号(10.5pt)、宋体
# 图题位于图下方（段后空一行），表题位于表上方（段前空一行）
# ════════════════════════════════════════════════════════════════
try:
    cap = doc.styles['Caption']
except:
    cap = doc.styles.add_style('Caption', WD_STYLE_TYPE.PARAGRAPH)
force_cn_font(cap, '宋体', 'Times New Roman')
set_sz(cap, 10.5)
set_bold(cap, False)
set_color_black(cap)
set_para_fmt(cap,
             align='center',
             first_line_cm=None,
             left_indent_cm=0,
             space_before_pt=0,
             space_after_pt=12,   # 图与下文空一行（约12pt）
             line_spacing_lines=1.25)

# ════════════════════════════════════════════════════════════════
# 目录标题（TOC Heading）：三号(16pt)、黑体、居中、与内容空一行
# ════════════════════════════════════════════════════════════════
try:
    toc_heading = doc.styles['TOC Heading']
except:
    try:
        toc_heading = doc.styles.add_style('TOC Heading', WD_STYLE_TYPE.PARAGRAPH)
    except:
        toc_heading = None

if toc_heading:
    force_cn_font(toc_heading, '黑体', 'Times New Roman')
    set_sz(toc_heading, 16)   # 三号 = 16pt
    set_bold(toc_heading, True)
    set_color_black(toc_heading)
    set_para_fmt(toc_heading,
                 align='center',
                 first_line_cm=None,
                 left_indent_cm=0,
                 space_before_pt=0,
                 space_after_pt=12,   # 与内容空一行
                 line_spacing_lines=1.25)

# ════════════════════════════════════════════════════════════════
# TOC 1 - 目录章条目：四号(14pt)、宋体
# ════════════════════════════════════════════════════════════════
try:
    toc1 = doc.styles['TOC 1']
except:
    try:
        toc1 = doc.styles.add_style('TOC 1', WD_STYLE_TYPE.PARAGRAPH)
    except:
        toc1 = None

if toc1:
    force_cn_font(toc1, '宋体', 'Times New Roman')
    set_sz(toc1, 14)    # 四号 = 14pt
    set_bold(toc1, False)
    set_color_black(toc1)
    set_para_fmt(toc1,
                 align='left',
                 first_line_cm=None,
                 left_indent_cm=0,
                 space_before_pt=6,
                 space_after_pt=0,
                 line_spacing_lines=1.25)

# ════════════════════════════════════════════════════════════════
# TOC 2 - 目录节条目：小四(12pt)、宋体、左缩进（空一格约0.43cm）
# ════════════════════════════════════════════════════════════════
try:
    toc2 = doc.styles['TOC 2']
except:
    try:
        toc2 = doc.styles.add_style('TOC 2', WD_STYLE_TYPE.PARAGRAPH)
    except:
        toc2 = None

if toc2:
    force_cn_font(toc2, '宋体', 'Times New Roman')
    set_sz(toc2, 12)    # 小四 = 12pt
    set_bold(toc2, False)
    set_color_black(toc2)
    set_para_fmt(toc2,
                 align='left',
                 first_line_cm=None,
                 left_indent_cm=0.43,   # 空一格
                 space_before_pt=0,
                 space_after_pt=0,
                 line_spacing_lines=1.25)

# ════════════════════════════════════════════════════════════════
# TOC 3 - 目录三级条目：小四(12pt)、宋体、左缩进（空两格约0.85cm）
# ════════════════════════════════════════════════════════════════
try:
    toc3 = doc.styles['TOC 3']
except:
    try:
        toc3 = doc.styles.add_style('TOC 3', WD_STYLE_TYPE.PARAGRAPH)
    except:
        toc3 = None

if toc3:
    force_cn_font(toc3, '宋体', 'Times New Roman')
    set_sz(toc3, 12)
    set_bold(toc3, False)
    set_color_black(toc3)
    set_para_fmt(toc3,
                 align='left',
                 first_line_cm=None,
                 left_indent_cm=0.85,   # 空两格
                 space_before_pt=0,
                 space_after_pt=0,
                 line_spacing_lines=1.25)

# ════════════════════════════════════════════════════════════════
# 代码块样式
# ════════════════════════════════════════════════════════════════
try:
    code_s = doc.styles['Verbatim Char']
except:
    try:
        code_s = doc.styles.add_style('Verbatim Char', WD_STYLE_TYPE.CHARACTER)
    except:
        code_s = None

try:
    sc = doc.styles['Source Code']
except:
    try:
        sc = doc.styles.add_style('Source Code', WD_STYLE_TYPE.PARAGRAPH)
    except:
        sc = None

if sc:
    force_cn_font(sc, 'Consolas', 'Consolas')
    set_sz(sc, 9)
    set_bold(sc, False)
    set_color_black(sc)
    set_para_fmt(sc,
                 align='left',
                 first_line_cm=None,
                 left_indent_cm=0.4,
                 space_before_pt=0,
                 space_after_pt=0,
                 line_spacing_lines=1.2)
    # 灰色背景
    s_elem = sc.element
    pPr = s_elem.find(qn('w:pPr'))
    if pPr is None:
        pPr = OxmlElement('w:pPr')
        s_elem.insert(0, pPr)
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  'F5F5F5')
    pPr.append(shd)
    # 左侧粗竖线 + 上下细线（仅左边框，连续段自然合并）
    pBdr = OxmlElement('w:pBdr')
    for side, sz, space, color in [
        ('top',    4,  1, 'E0E0E0'),
        ('left',  18,  4, '888888'),   # 粗竖线，3pt
        ('bottom', 4,  1, 'E0E0E0'),
    ]:
        el = OxmlElement(f'w:{side}')
        el.set(qn('w:val'),   'single')
        el.set(qn('w:sz'),    str(sz))
        el.set(qn('w:space'), str(space))
        el.set(qn('w:color'), color)
        pBdr.append(el)
    pPr.append(pBdr)

out = f'{BASE}/reference.docx'
doc.save(out)
print(f'[OK] reference.docx -> {out}')
