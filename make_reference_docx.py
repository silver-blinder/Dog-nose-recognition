"""
生成 Pandoc reference.docx（同济论文格式，严格按图一）
格式要求：
  Heading 1 - 章标题：居中、小三号(15pt)、黑体、西文 Times New Roman 加粗
  Heading 2 - 节标题(1.1)：顶格、四号(14pt)、黑体、西文 Times New Roman 加粗
  Heading 3 - 小节(1.1.1)：顶格、四号(14pt)、黑体、西文 Times New Roman 加粗
  Normal    - 正文：小四号(12pt)、宋体、首行缩进2字符、1.25倍行距
"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

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
    """
    直接操作 XML 强制设置中英文字体。
    style_obj 是 docx Style 对象。
    """
    # 找到或创建 rPr 节点
    s_elem = style_obj.element
    # 找 rPr
    rPr = s_elem.find(qn('w:rPr'))
    if rPr is None:
        rPr = OxmlElement('w:rPr')
        s_elem.append(rPr)

    # 删掉旧 rFonts
    old = rPr.find(qn('w:rFonts'))
    if old is not None:
        rPr.remove(old)

    # 写新 rFonts，四个属性全部设置
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

    # 对齐
    if align is not None:
        jc = pPr.find(qn('w:jc'))
        if jc is not None:
            pPr.remove(jc)
        jc = OxmlElement('w:jc')
        jc.set(qn('w:val'), align)   # 'center' / 'left' / 'both'
        pPr.append(jc)

    # 缩进
    ind = pPr.find(qn('w:ind'))
    if ind is None:
        ind = OxmlElement('w:ind')
        pPr.append(ind)
    else:
        for a in list(ind.attrib):
            del ind.attrib[a]
    if first_line_cm is not None:
        # 1 cm = 567 twips
        val = int(first_line_cm * 567)
        ind.set(qn('w:firstLine'), str(val))
    else:
        ind.set(qn('w:firstLine'), '0')
    if left_indent_cm is not None:
        ind.set(qn('w:left'), str(int(left_indent_cm * 567)))
    else:
        ind.set(qn('w:left'), '0')

    # 段前段后
    spc = pPr.find(qn('w:spacing'))
    if spc is None:
        spc = OxmlElement('w:spacing')
        pPr.append(spc)
    else:
        for a in list(spc.attrib):
            del spc.attrib[a]
    # 段前段后单位 twips (1pt=20twips)
    spc.set(qn('w:before'), str(int(space_before_pt * 20)))
    spc.set(qn('w:after'),  str(int(space_after_pt  * 20)))
    # 行距：multiple = lineSpacing/240
    spc.set(qn('w:line'),      str(int(line_spacing_lines * 240)))
    spc.set(qn('w:lineRule'), 'auto')

    # 章标题分页
    if page_break_before:
        pbk = pPr.find(qn('w:pageBreakBefore'))
        if pbk is None:
            pbk = OxmlElement('w:pageBreakBefore')
            pPr.append(pbk)

    # 关掉自动标题编号样式继承颜色
    rPr2 = pPr.find(qn('w:rPr'))
    if rPr2 is None:
        rPr2 = OxmlElement('w:rPr')
        pPr.append(rPr2)


# ════════════════════════════════════════════════════════════════
# Normal 正文：宋体小四，首行缩进2字符（约0.85cm），1.25倍行距
# ════════════════════════════════════════════════════════════════
normal = doc.styles['Normal']
force_cn_font(normal, '宋体', 'Times New Roman')
set_sz(normal, 12)   # 小四 = 12pt
set_bold(normal, False)
set_color_black(normal)
set_para_fmt(normal,
             align='both',
             first_line_cm=0.85,   # 两个字符约 0.85cm
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
             space_before_pt=3,
             space_after_pt=3,
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

# Source Code Para
try:
    sc = doc.styles['Source Code']
except:
    try:
        sc = doc.styles.add_style('Source Code', WD_STYLE_TYPE.PARAGRAPH)
    except:
        sc = None

if sc:
    force_cn_font(sc, 'Courier New', 'Courier New')
    set_sz(sc, 9)
    set_bold(sc, False)
    set_para_fmt(sc,
                 align='left',
                 first_line_cm=None,
                 left_indent_cm=0.5,
                 space_before_pt=4,
                 space_after_pt=4,
                 line_spacing_lines=1.0)

out = '/Users/rorschach/Workspace/personal/dog-nose-recognition/reference.docx'
doc.save(out)
print(f'[OK] reference.docx -> {out}')
