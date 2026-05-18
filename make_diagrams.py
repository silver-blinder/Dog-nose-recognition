"""
生成论文中4张架构/流程图的 PNG 文件（黑白学术线条风格）
输出到 network_training/experiments/results/figures/diagrams/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from pathlib import Path

OUT = Path('/Users/songjiayi4/Workspace/Dog-nose-recognition/network_training/experiments/results/figures/diagrams')
OUT.mkdir(parents=True, exist_ok=True)

# ── 字体设置 ──────────────────────────────────────────────────────────────────
import matplotlib.font_manager as _fm
_zh_candidates = ['PingFang SC', 'Heiti TC', 'STHeiti', 'Arial Unicode MS',
                  'Hiragino Sans GB', 'SimHei', 'Microsoft YaHei']
_zh_available  = {f.name for f in _fm.fontManager.ttflist}
_zh_font = next((f for f in _zh_candidates if f in _zh_available), None)
if _zh_font:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [_zh_font] + list(plt.rcParams.get('font.sans-serif', []))
else:
    plt.rcParams['font.family'] = ['STHeiti', 'Arial Unicode MS', 'Hiragino Sans GB', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 黑白学术配色 ──────────────────────────────────────────────────────────────
WHITE     = '#FFFFFF'
BLACK     = '#000000'
LIGHT_GRAY = '#E8E8E8'
MID_GRAY  = '#AAAAAA'
DARK      = '#000000'


def rect_box(ax, x, y, w, h, label, sublabel='', fontsize=10, bold=False):
    """绘制矩形方框（黑色边框，白色或浅灰填充）"""
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="square,pad=0",
                          facecolor=LIGHT_GRAY, edgecolor=BLACK, linewidth=1.2)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    if sublabel:
        ax.text(x, y + h * 0.15, label, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, color=BLACK)
        ax.text(x, y - h * 0.2, sublabel, ha='center', va='center',
                fontsize=fontsize - 1.5, color='#444444')
    else:
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, color=BLACK)


def diamond_box(ax, x, y, w, h, label, fontsize=9.5):
    """绘制菱形（判断框）"""
    pts = np.array([
        [x,         y + h/2],
        [x + w/2,   y],
        [x,         y - h/2],
        [x - w/2,   y],
    ])
    diamond = plt.Polygon(pts, closed=True, facecolor=WHITE, edgecolor=BLACK, linewidth=1.2)
    ax.add_patch(diamond)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fontsize, color=BLACK)


def arrow(ax, x1, y1, x2, y2, label='', label_offset=(0.08, 0)):
    """绘制箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=BLACK, lw=1.2))
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=8.5, color=BLACK, va='center')


def horiz_line(ax, x1, y, x2):
    ax.plot([x1, x2], [y, y], color=BLACK, lw=1.2)


def vert_line(ax, x, y1, y2):
    ax.plot([x, x], [y1, y2], color=BLACK, lw=1.2)


# ══════════════════════════════════════════════════════════════════════════════
# 图1：三层系统架构图（彩色卡片风格）
# ══════════════════════════════════════════════════════════════════════════════
def make_system_arch():
    # 彩色配色
    C_BLUE   = '#5B9BD5'   # 用户界面层边框
    C_GREEN  = '#70AD47'   # 业务逻辑层边框
    C_ORANGE = '#ED7D31'   # 模型层边框
    C_PURPLE = '#9B59B6'   # 数据存储层边框
    BG_BLUE   = '#DEEAF1'
    BG_GREEN  = '#E2EFDA'
    BG_ORANGE = '#FCE4D6'
    BG_PURPLE = '#F0E6F6'

    # 行间距（更紧凑）
    LINE_GAP = 0.18

    # 坐标系高度压缩，裁去空白
    fig, ax = plt.subplots(figsize=(5.5, 3.9))
    ax.set_xlim(0, 5.5); ax.set_ylim(2.78, 6.85)
    ax.axis('off')
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)

    CX = 2.75   # 图水平中心

    # ── 用户界面层 ──────────────────────────────────────────────────────────
    bx1_cy, bx1_w, bx1_h = 6.35, 4.6, 0.88
    rect1 = FancyBboxPatch((CX - bx1_w/2, bx1_cy - bx1_h/2), bx1_w, bx1_h,
                           boxstyle="round,pad=0.04",
                           facecolor=BG_BLUE, edgecolor=C_BLUE, linewidth=2)
    ax.add_patch(rect1)
    ty = bx1_cy + 0.22
    ax.text(CX, ty, '用户界面层（Frontend）', ha='center', va='center',
            fontsize=11, fontweight='bold', color=C_BLUE)
    ty -= LINE_GAP
    ax.text(CX, ty, 'Next.js 15 + Tailwind CSS + TypeScript', ha='center',
            va='center', fontsize=9.5, color='#333333')
    ty -= LINE_GAP
    ax.text(CX, ty, '首页 / 查询页面 / 登记页面 / API Routes', ha='center',
            va='center', fontsize=9, color='#555555')

    # ── 箭头（Frontend → Backend）────────────────────────────────────────────
    arr1_top = bx1_cy - bx1_h/2
    arr1_bot = arr1_top - 0.36
    ax.annotate('', xy=(CX, arr1_bot), xytext=(CX, arr1_top),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.4))
    ax.text(CX + 0.18, (arr1_top + arr1_bot)/2, 'HTTP / REST',
            fontsize=8.5, color='#555555', va='center')

    # ── 业务逻辑层 ───────────────────────────────────────────────────────────
    bx2_cy = arr1_bot - 0.44
    bx2_w, bx2_h = 4.6, 0.82
    rect2 = FancyBboxPatch((CX - bx2_w/2, bx2_cy - bx2_h/2), bx2_w, bx2_h,
                           boxstyle="round,pad=0.04",
                           facecolor=BG_GREEN, edgecolor=C_GREEN, linewidth=2)
    ax.add_patch(rect2)
    ty = bx2_cy + 0.19
    ax.text(CX, ty, '业务逻辑层（Backend）', ha='center', va='center',
            fontsize=11, fontweight='bold', color=C_GREEN)
    ty -= LINE_GAP
    ax.text(CX, ty, 'FastAPI + PyTorch + Uvicorn', ha='center',
            va='center', fontsize=9.5, color='#333333')
    ty -= LINE_GAP
    ax.text(CX, ty, '/compare   /compare-files   /health', ha='center',
            va='center', fontsize=9, color='#555555')

    # ── 分叉箭头（Backend → 两个子层）────────────────────────────────────────
    bx2_bot = bx2_cy - bx2_h/2
    fork_y  = bx2_bot - 0.22
    bx3_cx  = 1.38
    bx4_cx  = 4.12
    bx3_h34 = 0.76
    bx3_cy  = bx4_cy = fork_y - 0.22 - bx3_h34/2

    ax.plot([CX, CX],         [bx2_bot, fork_y], color='#555555', lw=1.4)
    ax.plot([bx3_cx, bx4_cx], [fork_y,  fork_y], color='#555555', lw=1.4)
    bx34_top = bx3_cy + bx3_h34/2
    ax.annotate('', xy=(bx3_cx, bx34_top), xytext=(bx3_cx, fork_y),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.4))
    ax.annotate('', xy=(bx4_cx, bx34_top), xytext=(bx4_cx, fork_y),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.4))

    # ── 模型层 ───────────────────────────────────────────────────────────────
    bx3_w = 2.36
    rect3 = FancyBboxPatch((bx3_cx - bx3_w/2, bx3_cy - bx3_h34/2), bx3_w, bx3_h34,
                           boxstyle="round,pad=0.04",
                           facecolor=BG_ORANGE, edgecolor=C_ORANGE, linewidth=2)
    ax.add_patch(rect3)
    ty = bx3_cy + 0.18
    ax.text(bx3_cx, ty, '模型层（ML Model）', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=C_ORANGE)
    ty -= LINE_GAP
    ax.text(bx3_cx, ty, 'Siamese-ResNet50', ha='center',
            va='center', fontsize=8.5, color='#333333')
    ty -= LINE_GAP
    ax.text(bx3_cx, ty, 'siamese_network.pth', ha='center',
            va='center', fontsize=8, color='#555555')

    # ── 数据存储层 ───────────────────────────────────────────────────────────
    bx4_w = 2.36
    rect4 = FancyBboxPatch((bx4_cx - bx4_w/2, bx4_cy - bx3_h34/2), bx4_w, bx3_h34,
                           boxstyle="round,pad=0.04",
                           facecolor=BG_PURPLE, edgecolor=C_PURPLE, linewidth=2)
    ax.add_patch(rect4)
    ty = bx4_cy + 0.18
    ax.text(bx4_cx, ty, '数据存储层（Supabase）', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=C_PURPLE)
    ty -= LINE_GAP
    ax.text(bx4_cx, ty, 'PostgreSQL（犬只档案）', ha='center',
            va='center', fontsize=8.5, color='#333333')
    ty -= LINE_GAP
    ax.text(bx4_cx, ty, 'Storage Bucket（鼻纹图片）', ha='center',
            va='center', fontsize=8, color='#555555')

    fig.tight_layout(pad=0.3)
    p = OUT / 'fig_system_arch.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close(fig)
    print(f'[OK] {p}')


# ══════════════════════════════════════════════════════════════════════════════
# 图2：孪生网络结构图（彩色紧凑版）
# ══════════════════════════════════════════════════════════════════════════════
def make_siamese_arch():
    # 配色
    C_BLUE   = '#5B9BD5'; BG_BLUE   = '#DEEAF1'
    C_GREEN  = '#70AD47'; BG_GREEN  = '#E2EFDA'
    C_ORANGE = '#ED7D31'; BG_ORANGE = '#FCE4D6'
    C_PURPLE = '#9B59B6'; BG_PURPLE = '#F0E6F6'
    C_GOLD   = '#C9A800'; BG_GOLD   = '#FFF9CC'
    ARROW_C  = '#555555'

    GAP = 0.17   # 行间距

    fig, ax = plt.subplots(figsize=(8, 6.6))
    ax.set_xlim(0, 8); ax.set_ylim(1.35, 8.2)
    ax.axis('off')
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)

    CX = 4.0   # 全图水平中心

    def box(cx, cy, w, h, fc, ec, lw=1.8):
        p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0.04",
                           facecolor=fc, edgecolor=ec, linewidth=lw)
        ax.add_patch(p)

    def arr(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=ARROW_C, lw=1.4))

    # ── 输入两框 ─────────────────────────────────────────────────────────────
    IN_W, IN_H = 2.8, 0.52
    IN_CY = 7.85
    in_cx1, in_cx2 = 1.8, 6.2
    for cx, lbl in [(in_cx1, '输入图片1（224×224×3）'),
                    (in_cx2, '输入图片2（224×224×3）')]:
        box(cx, IN_CY, IN_W, IN_H, BG_BLUE, C_BLUE)
        ax.text(cx, IN_CY, lbl, ha='center', va='center',
                fontsize=10, fontweight='bold', color=C_BLUE)

    # ── 箭头向下到骨干框顶 ───────────────────────────────────────────────────
    BB_H = 0.58
    BB_CY = 6.72
    bb_top = BB_CY + BB_H/2
    arr(in_cx1, IN_CY - IN_H/2, in_cx1, bb_top)
    arr(in_cx2, IN_CY - IN_H/2, in_cx2, bb_top)

    # ── 共享骨干（宽框）─────────────────────────────────────────────────────
    BB_W = 7.2
    box(CX, BB_CY, BB_W, BB_H, BG_GREEN, C_GREEN, lw=2)
    ty = BB_CY + GAP / 2 + 0.04   # 两行垂直居中：从中心偏上半个行距
    ax.text(CX, ty, 'ResNet-50 Backbone（共享权重，孪生结构）',
            ha='center', va='center', fontsize=10.5, fontweight='bold', color=C_GREEN)
    ty -= GAP + 0.04
    ax.text(CX, ty, 'conv1 → pool1 → conv2_x → conv3_x → conv4_x → conv5_x → AvgPool → 2048维特征',
            ha='center', va='center', fontsize=8.5, color='#333333')

    # ── 骨干底部 → 两个特征向量框 ───────────────────────────────────────────
    bb_bot = BB_CY - BB_H/2
    FV_H = 0.46
    FV_CY = bb_bot - 0.38 - FV_H/2
    fv_cx1, fv_cx2 = 1.9, 6.1
    FV_W = 2.6
    for cx, lbl in [(fv_cx1, '特征向量1（2048-d）'),
                    (fv_cx2, '特征向量2（2048-d）')]:
        arr(cx, bb_bot, cx, FV_CY + FV_H/2)
        box(cx, FV_CY, FV_W, FV_H, BG_BLUE, C_BLUE)
        ax.text(cx, FV_CY, lbl, ha='center', va='center',
                fontsize=10, fontweight='bold', color=C_BLUE)

    # ── 两特征向量汇聚到元素级差框 ─────────────────────────────────────────
    DIFF_H = 0.50
    DIFF_CY = FV_CY - FV_H/2 - 0.42 - DIFF_H/2
    DIFF_W = 5.0
    arr(fv_cx1, FV_CY - FV_H/2, CX - DIFF_W*0.3, DIFF_CY + DIFF_H/2)
    arr(fv_cx2, FV_CY - FV_H/2, CX + DIFF_W*0.3, DIFF_CY + DIFF_H/2)
    box(CX, DIFF_CY, DIFF_W, DIFF_H, BG_ORANGE, C_ORANGE)
    ax.text(CX, DIFF_CY, '元素级绝对值差  |feat1 − feat2|  （2048-d）',
            ha='center', va='center', fontsize=10, fontweight='bold', color=C_ORANGE)

    # ── 元素级差 → FC Head ──────────────────────────────────────────────────
    FC_H = 1.00
    FC_CY = DIFF_CY - DIFF_H/2 - 0.38 - FC_H/2
    FC_W = 4.6
    arr(CX, DIFF_CY - DIFF_H/2, CX, FC_CY + FC_H/2)
    box(CX, FC_CY, FC_W, FC_H, BG_PURPLE, C_PURPLE)
    ty = FC_CY + 0.32
    ax.text(CX, ty, '全连接分类头（FC Head）',
            ha='center', va='center', fontsize=10.5, fontweight='bold', color=C_PURPLE)
    for line in ['Linear(2048 → 256)  +  ReLU',
                 'Linear(256 → 128)  +  ReLU',
                 'Linear(128 → 1)']:
        ty -= GAP + 0.04
        ax.text(CX, ty, line, ha='center', va='center',
                fontsize=9.5, color='#333333')

    # ── FC Head → 输出 ───────────────────────────────────────────────────────
    OUT_H = 0.46
    OUT_CY = FC_CY - FC_H/2 - 0.36 - OUT_H/2
    OUT_W = 3.8
    arr(CX, FC_CY - FC_H/2, CX, OUT_CY + OUT_H/2)
    box(CX, OUT_CY, OUT_W, OUT_H, BG_GOLD, C_GOLD)
    ax.text(CX, OUT_CY, 'Sigmoid  →  相似度概率  [0, 1]',
            ha='center', va='center', fontsize=10, fontweight='bold', color=C_GOLD)

    fig.tight_layout(pad=0.2)
    p = OUT / 'fig_siamese_arch.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close(fig)
    print(f'[OK] {p}')


# ══════════════════════════════════════════════════════════════════════════════
# 图3：训练流程图（竖向主干，中间四步横向一行）
# ══════════════════════════════════════════════════════════════════════════════
def make_train_flow():
    # 画布：宽13（容纳横向四步）× 高8.4
    fig, ax = plt.subplots(figsize=(13, 8.4))
    ax.set_xlim(0, 13); ax.set_ylim(0.8, 9)
    ax.axis('off')
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)

    CX = 6.5          # 竖向主轴 x
    FS = 13           # 主字号（放大）
    FS_SUB = 11       # 副文字字号（放大）

    def vbox(cx, cy, w, h, line1, line2=''):
        """带可选副标题的矩形框"""
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                              boxstyle="square,pad=0",
                              facecolor=LIGHT_GRAY, edgecolor=BLACK, linewidth=1.4)
        ax.add_patch(rect)
        if line2:
            ax.text(cx, cy + h * 0.20, line1, ha='center', va='center',
                    fontsize=FS, fontweight='bold', color=BLACK)
            ax.text(cx, cy - h * 0.16, line2, ha='center', va='center',
                    fontsize=FS_SUB, color='#444444')
        else:
            ax.text(cx, cy, line1, ha='center', va='center',
                    fontsize=FS, fontweight='bold', color=BLACK)

    def varr(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=BLACK, lw=1.4))

    # ━━━ ① 开始新 Epoch ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    E_W, E_H = 3.6, 0.62
    E_CY = 8.34
    vbox(CX, E_CY, E_W, E_H, '开始新 Epoch')
    varr(CX, E_CY - E_H/2, CX, 7.50)

    # ━━━ ② 加载样本对批次 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    L_W, L_H = 4.4, 0.80
    L_CY = 7.12
    vbox(CX, L_CY, L_W, L_H, '加载样本对批次', '(img1, img2, label)')
    load_bot_y = L_CY - L_H/2

    # ━━━ ③ 中间四步横向一行 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ROW_CY  = 5.68         # 四步中心 y（收紧行间距）
    ROW_H   = 1.00         # 框高
    ROW_W   = 2.56         # 单框宽
    ROW_GAP = 0.26         # 框间距
    xs = [CX - 1.5*(ROW_W + ROW_GAP),
          CX - 0.5*(ROW_W + ROW_GAP),
          CX + 0.5*(ROW_W + ROW_GAP),
          CX + 1.5*(ROW_W + ROW_GAP)]

    row_labels = [
        ('前向传播',      '孪生网络输出\n相似度得分'),
        ('计算对比损失',  'ContrastiveLoss\n(score, label)'),
        ('反向传播',      'loss.backward()'),
        ('参数更新',      'optimizer.step()'),
    ]

    # 折线：底中心竖向下 → 横向左 → 箭头从上入"前向传播"框顶
    row_top_y = ROW_CY + ROW_H/2
    ax.plot([CX, CX],    [load_bot_y, row_top_y + 0.10],       color=BLACK, lw=1.4)
    ax.plot([CX, xs[0]], [row_top_y + 0.10, row_top_y + 0.10], color=BLACK, lw=1.4)
    varr(xs[0], row_top_y + 0.10, xs[0], row_top_y)

    for i, (cx, (l1, l2)) in enumerate(zip(xs, row_labels)):
        vbox(cx, ROW_CY, ROW_W, ROW_H, l1, l2)
        if i < len(xs) - 1:
            ax.annotate('', xy=(xs[i+1] - ROW_W/2, ROW_CY),
                        xytext=(cx + ROW_W/2, ROW_CY),
                        arrowprops=dict(arrowstyle='->', color=BLACK, lw=1.4))

    # 最后一个横框底部 → 折回竖轴
    last_cx = xs[-1]
    fold_y  = ROW_CY - ROW_H/2 - 0.18
    ax.plot([last_cx, last_cx], [ROW_CY - ROW_H/2, fold_y], color=BLACK, lw=1.4)
    ax.plot([last_cx, CX],      [fold_y, fold_y],            color=BLACK, lw=1.4)
    varr(CX, fold_y, CX, 4.76)

    # ━━━ ④ 判断菱形：批次用完？ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    DM_CY = 4.32
    DM_W, DM_H = 3.4, 0.90
    diamond_box(ax, CX, DM_CY, DM_W, DM_H, '批次用完？', fontsize=FS)

    # "否" → 右侧绕回加载批次框右侧
    no_x   = CX + DM_W/2
    loop_x = 12.4
    ax.plot([no_x, loop_x],   [DM_CY, DM_CY],  color=BLACK, lw=1.4)
    ax.plot([loop_x, loop_x], [DM_CY, L_CY],   color=BLACK, lw=1.4)
    ax.annotate('', xy=(CX + L_W/2, L_CY), xytext=(loop_x, L_CY),
                arrowprops=dict(arrowstyle='->', color=BLACK, lw=1.4))
    ax.text(loop_x + 0.12, (DM_CY + L_CY)/2, '否',
            fontsize=FS, color=BLACK, ha='left', va='center')

    # "是" → 向下
    varr(CX, DM_CY - DM_H/2, CX, 3.30)
    ax.text(CX + 0.18, DM_CY - DM_H/2 - 0.16, '是', fontsize=FS, color=BLACK)

    # ━━━ ⑤ 验证阶段 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    V_W, V_H = 4.4, 0.72
    V_CY = 2.92
    vbox(CX, V_CY, V_W, V_H, '验证阶段', '计算 val_loss / val_acc')
    varr(CX, V_CY - V_H/2, CX, 2.02)

    # ━━━ ⑥ 保存最优模型 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    S_W, S_H = 4.4, 0.62
    S_CY = 1.64
    vbox(CX, S_CY, S_W, S_H, '保存最优模型 checkpoint')

    fig.tight_layout(pad=0.3)
    p = OUT / 'fig_train_flow.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close(fig)
    print(f'[OK] {p}')


# ══════════════════════════════════════════════════════════════════════════════
# 图4：查询/注册数据流图（两列并排）
# ══════════════════════════════════════════════════════════════════════════════
def make_query_flow():
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    fig.patch.set_facecolor(WHITE)

    BOX_H  = 0.75
    BOX_W  = 4.2
    CX     = 2.5          # 列中心 x
    Y_START = 9.5
    Y_STEP  = 1.35

    query_steps = [
        '用户上传查询鼻纹图片',
        'Next.js 临时上传至\nSupabase Storage → 获取 URL',
        '携带查询URL + 全库档案URL\n批量调用 /compare 接口',
        'FastAPI + 孪生网络\n逐一计算相似度',
        '返回置信度最高的匹配结果',
        '删除临时文件',
    ]
    register_steps_before = [
        '用户填写犬只信息\n并上传鼻纹图片',
        '图片上传至\nSupabase Storage',
        '与全库已有档案比对\n计算最高相似度',
    ]
    # 第4步是菱形判断
    register_steps_after = [
        '写入 PostgreSQL\n新档案',
    ]

    for ax, (title, steps_q) in zip(axes,
                                     [('查询流程', query_steps),
                                      ('注册流程', None)]):
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 11)
        ax.axis('off')
        ax.set_facecolor(WHITE)

        if title == '查询流程':
            ax.text(CX, 10.6, title, ha='center', fontsize=12,
                    fontweight='bold', color=BLACK)
            for i, lbl in enumerate(steps_q):
                y = Y_START - i * Y_STEP
                rect_box(ax, CX, y, BOX_W, BOX_H, lbl, fontsize=9.5)
                if i < len(steps_q) - 1:
                    arrow(ax, CX, y - BOX_H/2, CX, y - Y_STEP + BOX_H/2)
        else:
            # 注册流程
            ax.text(CX, 10.6, title, ha='center', fontsize=12,
                    fontweight='bold', color=BLACK)
            n_before = len(register_steps_before)
            for i, lbl in enumerate(register_steps_before):
                y = Y_START - i * Y_STEP
                rect_box(ax, CX, y, BOX_W, BOX_H, lbl, fontsize=9.5)
                if i < n_before - 1:
                    arrow(ax, CX, y - BOX_H/2, CX, y - Y_STEP + BOX_H/2)

            # 箭头到菱形
            y_last_before = Y_START - (n_before - 1) * Y_STEP
            dm_y = y_last_before - Y_STEP
            arrow(ax, CX, y_last_before - BOX_H/2, CX, dm_y + 0.45)

            # 菱形
            diamond_box(ax, CX, dm_y, 3.2, 0.9,
                        '相似度 > 50%？\n（重复检测）', fontsize=9)

            # "是" → 右侧文字
            ax.text(4.05, dm_y, '是→\n返回已有\n档案', fontsize=8,
                    color=BLACK, va='center', ha='left')
            horiz_line(ax, CX + 3.2/2, dm_y, 4.0)

            # "否" → 向下
            ax.text(CX + 0.12, dm_y - 0.55, '否', fontsize=9, color=BLACK)
            after_y = dm_y - Y_STEP
            arrow(ax, CX, dm_y - 0.45, CX, after_y + BOX_H/2)

            rect_box(ax, CX, after_y, BOX_W, BOX_H,
                     '写入 PostgreSQL\n新档案', fontsize=9.5)

    fig.tight_layout()
    p = OUT / 'fig_query_flow.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close(fig)
    print(f'[OK] {p}')


if __name__ == '__main__':
    make_system_arch()
    make_siamese_arch()
    make_train_flow()
    make_query_flow()
    print('All diagrams generated.')
