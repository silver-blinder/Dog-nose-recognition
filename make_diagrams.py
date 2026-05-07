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
# 图1：三层系统架构图
# ══════════════════════════════════════════════════════════════════════════════
def make_system_arch():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)

    ax.text(5, 6.65, '系统整体架构', ha='center', va='center',
            fontsize=13, fontweight='bold', color=BLACK)

    # ── 用户界面层 ──
    rect1 = FancyBboxPatch((0.5, 4.6), 9, 1.5, boxstyle="square,pad=0",
                            facecolor=LIGHT_GRAY, edgecolor=BLACK, linewidth=1.5)
    ax.add_patch(rect1)
    ax.text(5, 5.65, '用户界面层（Frontend）', ha='center', fontsize=11,
            fontweight='bold', color=BLACK)
    ax.text(5, 5.25, 'Next.js 15 + Tailwind CSS + TypeScript', ha='center',
            fontsize=9.5, color=BLACK)
    ax.text(5, 4.88, '首页 / 查询页面 / 登记页面 / API Routes', ha='center',
            fontsize=9, color='#444444')

    # ── 箭头 + 标签 ──
    arrow(ax, 5, 4.6, 5, 3.9)
    ax.text(5.15, 4.25, 'HTTP / REST', fontsize=8.5, color=BLACK)

    # ── 业务逻辑层 ──
    rect2 = FancyBboxPatch((0.5, 2.8), 9, 1.4, boxstyle="square,pad=0",
                            facecolor=LIGHT_GRAY, edgecolor=BLACK, linewidth=1.5)
    ax.add_patch(rect2)
    ax.text(5, 3.75, '业务逻辑层（Backend）', ha='center', fontsize=11,
            fontweight='bold', color=BLACK)
    ax.text(5, 3.35, 'FastAPI + PyTorch + Uvicorn', ha='center',
            fontsize=9.5, color=BLACK)
    ax.text(5, 2.98, '/compare   /compare-files   /health', ha='center',
            fontsize=9, color='#444444')

    # ── 分叉箭头 ──
    arrow(ax, 2.5, 2.8, 2.5, 2.15)
    arrow(ax, 7.5, 2.8, 7.5, 2.15)

    # ── 模型层 ──
    rect3 = FancyBboxPatch((0.3, 0.9), 4, 1.5, boxstyle="square,pad=0",
                            facecolor=LIGHT_GRAY, edgecolor=BLACK, linewidth=1.5)
    ax.add_patch(rect3)
    ax.text(2.3, 1.9, '模型层（ML Model）', ha='center', fontsize=10,
            fontweight='bold', color=BLACK)
    ax.text(2.3, 1.5, 'Siamese-ResNet50', ha='center', fontsize=9.5, color=BLACK)
    ax.text(2.3, 1.15, 'siamese_network.pth', ha='center', fontsize=9, color='#444444')

    # ── 数据存储层 ──
    rect4 = FancyBboxPatch((5.3, 0.9), 4.2, 1.5, boxstyle="square,pad=0",
                            facecolor=LIGHT_GRAY, edgecolor=BLACK, linewidth=1.5)
    ax.add_patch(rect4)
    ax.text(7.4, 1.9, '数据存储层（Supabase）', ha='center', fontsize=10,
            fontweight='bold', color=BLACK)
    ax.text(7.4, 1.5, 'PostgreSQL（犬只档案）', ha='center', fontsize=9.5, color=BLACK)
    ax.text(7.4, 1.15, 'Storage Bucket（鼻纹图片）', ha='center', fontsize=9, color='#444444')

    fig.tight_layout()
    p = OUT / 'fig_system_arch.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close(fig)
    print(f'[OK] {p}')


# ══════════════════════════════════════════════════════════════════════════════
# 图2：孪生网络结构图
# ══════════════════════════════════════════════════════════════════════════════
def make_siamese_arch():
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 11); ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)

    ax.text(5.5, 7.7, '孪生网络结构', ha='center', fontsize=13,
            fontweight='bold', color=BLACK)

    # ── 输入 ──
    rect_box(ax, 2.2, 7.1, 3.0, 0.5, '输入图片1  (224×224×3)', fontsize=9.5)
    rect_box(ax, 8.8, 7.1, 3.0, 0.5, '输入图片2  (224×224×3)', fontsize=9.5)

    arrow(ax, 2.2, 6.85, 2.2, 6.52)
    arrow(ax, 8.8, 6.85, 8.8, 6.52)

    # ── 共享骨干 ──
    rect_bb = FancyBboxPatch((0.8, 5.1), 9.4, 1.3, boxstyle="square,pad=0",
                              facecolor=LIGHT_GRAY, edgecolor=BLACK, linewidth=1.5)
    ax.add_patch(rect_bb)
    ax.text(5.5, 5.95, 'ResNet-50 Backbone（共享权重，孪生结构）', ha='center',
            fontsize=11, fontweight='bold', color=BLACK)
    ax.text(5.5, 5.53, 'conv1→pool1→conv2_x→conv3_x→conv4_x→conv5_x→AvgPool  →  2048维特征',
            ha='center', fontsize=9, color='#444444')

    # ── 特征向量 ──
    rect_box(ax, 2.8, 4.78, 3.0, 0.5, '特征向量1  (2048-d)', fontsize=9.5)
    rect_box(ax, 8.2, 4.78, 3.0, 0.5, '特征向量2  (2048-d)', fontsize=9.5)

    arrow(ax, 2.5, 5.1, 2.8, 5.03)
    arrow(ax, 8.5, 5.1, 8.2, 5.03)

    # ── 汇聚箭头 → 差值 ──
    arrow(ax, 2.8, 4.53, 5.5, 4.05)
    arrow(ax, 8.2, 4.53, 5.5, 4.05)

    # ── 元素级差 ──
    rect_box(ax, 5.5, 3.78, 4.2, 0.48,
             '元素级绝对值差  |feat1 − feat2|  (2048-d)', fontsize=9.5)

    arrow(ax, 5.5, 3.54, 5.5, 3.1)

    # ── FC Head ──
    rect_fc = FancyBboxPatch((3.0, 1.7), 5.0, 1.6, boxstyle="square,pad=0",
                              facecolor=LIGHT_GRAY, edgecolor=BLACK, linewidth=1.5)
    ax.add_patch(rect_fc)
    ax.text(5.5, 3.06, '全连接分类头（FC Head）', ha='center', fontsize=10.5,
            fontweight='bold', color=BLACK)
    for y_fc, txt_fc in [(2.65, 'Linear(2048 → 256) + ReLU'),
                         (2.2,  'Linear(256 → 128) + ReLU'),
                         (1.83, 'Linear(128 → 1)')]:
        ax.text(5.5, y_fc, txt_fc, ha='center', fontsize=9.5, color=BLACK)

    arrow(ax, 5.5, 1.7, 5.5, 1.2)

    # ── 输出 ──
    rect_box(ax, 5.5, 0.95, 3.8, 0.45,
             'Sigmoid → 相似度概率  [0, 1]', fontsize=10)

    fig.tight_layout()
    p = OUT / 'fig_siamese_arch.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close(fig)
    print(f'[OK] {p}')


# ══════════════════════════════════════════════════════════════════════════════
# 图3：训练流程图
# ══════════════════════════════════════════════════════════════════════════════
def make_train_flow():
    fig, ax = plt.subplots(figsize=(7, 11))
    ax.set_xlim(0, 7); ax.set_ylim(0, 11)
    ax.axis('off')
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)

    ax.text(3.5, 10.7, '训练流程', ha='center', fontsize=13,
            fontweight='bold', color=BLACK)

    # ── 顺序步骤 ──
    step_labels = [
        '开始新 Epoch',
        '加载样本对批次\n(img1, img2, label)',
        '前向传播\n孪生网络输出相似度得分',
        '计算对比损失\nContrastiveLoss(score, label)',
        '反向传播\nloss.backward()',
        '参数更新\noptimizer.step()',
    ]
    step_ys = [10.1, 9.0, 7.9, 6.8, 5.7, 4.6]
    BOX_H = 0.75

    for label, y in zip(step_labels, step_ys):
        rect_box(ax, 3.5, y, 5.0, BOX_H, label, fontsize=9.5)

    for y_top, y_bot in zip(step_ys[:-1], step_ys[1:]):
        arrow(ax, 3.5, y_top - BOX_H/2, 3.5, y_bot + BOX_H/2)

    # ── 判断菱形 ──
    dm_y = 3.55
    dm_w, dm_h = 3.6, 0.9
    diamond_box(ax, 3.5, dm_y, dm_w, dm_h, '批次用完？', fontsize=9.5)
    arrow(ax, 3.5, step_ys[-1] - BOX_H/2, 3.5, dm_y + dm_h/2)

    # "否" → 左绕回
    ax.text(1.3, (dm_y + step_ys[1]) / 2, '否', fontsize=9, color=BLACK, ha='center')
    vert_line(ax, 0.9, dm_y, step_ys[1])
    horiz_line(ax, 0.9, dm_y, 3.5 - dm_w/2)
    horiz_line(ax, 0.9, step_ys[1], 3.5 - 5.0/2)
    ax.annotate('', xy=(3.5 - 5.0/2, step_ys[1]), xytext=(0.9, step_ys[1]),
                arrowprops=dict(arrowstyle='->', color=BLACK, lw=1.2))

    # "是" → 下
    ax.text(3.7, dm_y - dm_h/2 - 0.15, '是', fontsize=9, color=BLACK)
    arrow(ax, 3.5, dm_y - dm_h/2, 3.5, 2.55)

    # ── 验证阶段 ──
    rect_box(ax, 3.5, 2.2, 5.0, BOX_H,
             '验证阶段\n计算 val_loss / val_acc', fontsize=9.5)
    arrow(ax, 3.5, 2.2 - BOX_H/2, 3.5, 1.3)

    # ── 保存模型 ──
    rect_box(ax, 3.5, 0.95, 5.0, 0.6, '保存最优模型 checkpoint', fontsize=9.5)

    fig.tight_layout()
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
