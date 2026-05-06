"""
生成论文中4张架构/流程图的 PNG 文件（替换 ASCII 代码块）
输出到 network_training/experiments/results/figures/diagrams/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUT = Path('/Users/rorschach/Workspace/personal/dog-nose-recognition/network_training/experiments/results/figures/diagrams')
OUT.mkdir(parents=True, exist_ok=True)

# ── 字体设置 ─────────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = ['STHeiti', 'Arial Unicode MS', 'Hiragino Sans GB', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

GRAY_BG   = '#F5F5F5'
BLUE      = '#1565C0'
BLUE_LITE = '#BBDEFB'
GREEN     = '#2E7D32'
GREEN_LT  = '#C8E6C9'
ORANGE    = '#E65100'
ORANGE_LT = '#FFE0B2'
PURPLE    = '#6A1B9A'
PURPLE_LT = '#E1BEE7'
WHITE     = '#FFFFFF'
DARK      = '#212121'
ARROW_C   = '#455A64'


def box(ax, x, y, w, h, label, sublabel='', facecolor=BLUE_LITE, edgecolor=BLUE, fontsize=11):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.02",
                          facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + h*0.12, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=DARK)
        ax.text(x, y - h*0.2, sublabel, ha='center', va='center',
                fontsize=fontsize - 2, color='#424242')
    else:
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=DARK)


def arrow(ax, x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=ARROW_C, lw=1.8))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.02, my, label, fontsize=9, color=ARROW_C, va='center')


# ══════════════════════════════════════════════════════════════════════════════
# 图1：三层系统架构图
# ══════════════════════════════════════════════════════════════════════════════
def make_system_arch():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)

    ax.text(5, 6.6, '系统整体架构', ha='center', va='center',
            fontsize=14, fontweight='bold', color=DARK)

    # 用户界面层
    rect1 = FancyBboxPatch((0.5, 4.5), 9, 1.5, boxstyle="round,pad=0.05",
                            facecolor=BLUE_LITE, edgecolor=BLUE, linewidth=2)
    ax.add_patch(rect1)
    ax.text(5, 5.55, '用户界面层（Frontend）', ha='center', fontsize=12, fontweight='bold', color=BLUE)
    ax.text(5, 5.15, 'Next.js 15 + Tailwind CSS + TypeScript', ha='center', fontsize=10, color=DARK)
    ax.text(5, 4.78, '首页 / 查询页面 / 登记页面 / API Routes', ha='center', fontsize=9.5, color='#424242')

    # HTTP/REST 箭头
    arrow(ax, 5, 4.5, 5, 3.8, '')
    ax.text(5.15, 4.15, 'HTTP / REST', fontsize=9, color=ARROW_C)

    # 业务逻辑层
    rect2 = FancyBboxPatch((0.5, 2.8), 9, 1.4, boxstyle="round,pad=0.05",
                            facecolor=GREEN_LT, edgecolor=GREEN, linewidth=2)
    ax.add_patch(rect2)
    ax.text(5, 3.75, '业务逻辑层（Backend）', ha='center', fontsize=12, fontweight='bold', color=GREEN)
    ax.text(5, 3.35, 'FastAPI + PyTorch + Uvicorn', ha='center', fontsize=10, color=DARK)
    ax.text(5, 2.98, '/compare   /compare-files   /health', ha='center', fontsize=9.5, color='#424242')

    # 下层箭头：分两路
    arrow(ax, 2.5, 2.8, 2.5, 2.1)
    arrow(ax, 7.5, 2.8, 7.5, 2.1)

    # 模型层
    rect3 = FancyBboxPatch((0.3, 0.9), 4, 1.5, boxstyle="round,pad=0.05",
                            facecolor=ORANGE_LT, edgecolor=ORANGE, linewidth=2)
    ax.add_patch(rect3)
    ax.text(2.3, 1.9, '模型层（ML Model）', ha='center', fontsize=11, fontweight='bold', color=ORANGE)
    ax.text(2.3, 1.5, 'Siamese-ResNet50', ha='center', fontsize=9.5, color=DARK)
    ax.text(2.3, 1.15, 'siamese_network.pth', ha='center', fontsize=9, color='#424242')

    # 数据存储层
    rect4 = FancyBboxPatch((5.3, 0.9), 4.2, 1.5, boxstyle="round,pad=0.05",
                            facecolor=PURPLE_LT, edgecolor=PURPLE, linewidth=2)
    ax.add_patch(rect4)
    ax.text(7.4, 1.9, '数据存储层（Supabase）', ha='center', fontsize=11, fontweight='bold', color=PURPLE)
    ax.text(7.4, 1.5, 'PostgreSQL（犬只档案）', ha='center', fontsize=9.5, color=DARK)
    ax.text(7.4, 1.15, 'Storage Bucket（鼻纹图片）', ha='center', fontsize=9, color='#424242')

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

    ax.text(5.5, 7.65, '孪生网络结构', ha='center', fontsize=14, fontweight='bold', color=DARK)

    # 输入
    box(ax, 2.2, 7.0, 2.8, 0.55, '输入图片1  (224×224×3)', facecolor='#E3F2FD', edgecolor=BLUE, fontsize=10)
    box(ax, 8.8, 7.0, 2.8, 0.55, '输入图片2  (224×224×3)', facecolor='#E3F2FD', edgecolor=BLUE, fontsize=10)

    # ResNet-50 骨干（共享）
    rect = FancyBboxPatch((1.0, 5.2), 9, 1.3, boxstyle="round,pad=0.05",
                           facecolor=BLUE_LITE, edgecolor=BLUE, linewidth=2)
    ax.add_patch(rect)
    ax.text(5.5, 6.05, 'ResNet-50 Backbone（共享权重，孪生结构）', ha='center', fontsize=12, fontweight='bold', color=BLUE)
    ax.text(5.5, 5.62, 'conv1→pool1→conv2_x→conv3_x→conv4_x→conv5_x→AvgPool  →  2048维特征', ha='center', fontsize=9.5, color='#424242')

    arrow(ax, 2.2, 6.72, 2.2, 6.52)
    arrow(ax, 8.8, 6.72, 8.8, 6.52)

    # 特征向量
    box(ax, 2.8, 4.85, 2.8, 0.5, '特征向量1  (2048-d)', facecolor='#E8F5E9', edgecolor=GREEN, fontsize=10)
    box(ax, 8.2, 4.85, 2.8, 0.5, '特征向量2  (2048-d)', facecolor='#E8F5E9', edgecolor=GREEN, fontsize=10)

    arrow(ax, 2.5, 5.2, 2.8, 5.1)
    arrow(ax, 8.5, 5.2, 8.2, 5.1)

    # 合并箭头到差值
    arrow(ax, 2.8, 4.6, 5.5, 4.1)
    arrow(ax, 8.2, 4.6, 5.5, 4.1)

    # 元素级差
    box(ax, 5.5, 3.85, 3.8, 0.45, '元素级绝对值差  |feat1 − feat2|  (2048-d)',
        facecolor=ORANGE_LT, edgecolor=ORANGE, fontsize=10)

    arrow(ax, 5.5, 3.62, 5.5, 3.2)

    # FC Head
    fc_items = [
        (3.0, 'Linear(2048 → 256) + ReLU'),
        (2.55, 'Linear(256 → 128) + ReLU'),
        (2.1, 'Linear(128 → 1)'),
    ]
    rect_fc = FancyBboxPatch((3.2, 1.85), 4.6, 1.55, boxstyle="round,pad=0.04",
                              facecolor=PURPLE_LT, edgecolor=PURPLE, linewidth=2)
    ax.add_patch(rect_fc)
    ax.text(5.5, 3.15, '全连接分类头（FC Head）', ha='center', fontsize=11, fontweight='bold', color=PURPLE)
    for y, txt in fc_items:
        ax.text(5.5, y, txt, ha='center', fontsize=9.5, color=DARK)

    arrow(ax, 5.5, 1.85, 5.5, 1.4)

    # 输出
    box(ax, 5.5, 1.15, 3.5, 0.45, 'Sigmoid → 相似度概率  [0, 1]',
        facecolor='#FFF9C4', edgecolor='#F9A825', fontsize=10.5)

    fig.tight_layout()
    p = OUT / 'fig_siamese_arch.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close(fig)
    print(f'[OK] {p}')


# ══════════════════════════════════════════════════════════════════════════════
# 图3：训练流程图
# ══════════════════════════════════════════════════════════════════════════════
def make_train_flow():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 8); ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)

    ax.text(4, 9.65, '训练流程', ha='center', fontsize=14, fontweight='bold', color=DARK)

    steps = [
        (9.1, '开始新 Epoch', BLUE_LITE, BLUE),
        (8.3, '加载样本对批次\n(img1, img2, label)', '#E8F5E9', GREEN),
        (7.3, '前向传播\n孪生网络输出相似度得分', BLUE_LITE, BLUE),
        (6.3, '计算对比损失\nContrastiveLoss(score, label)', ORANGE_LT, ORANGE),
        (5.3, '反向传播\nloss.backward()', ORANGE_LT, ORANGE),
        (4.3, '参数更新\noptimizer.step()', GREEN_LT, GREEN),
    ]

    for y, label, fc, ec in steps:
        box(ax, 4, y, 5.5, 0.7, label, facecolor=fc, edgecolor=ec, fontsize=10)
        if y > 4.3:
            arrow(ax, 4, y - 0.35, 4, y - 0.65)

    # 循环判断菱形
    diamond_y = 3.45
    d_pts = np.array([[4, diamond_y+0.45], [5.5, diamond_y], [4, diamond_y-0.45], [2.5, diamond_y]])
    diamond = plt.Polygon(d_pts, closed=True, facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=2)
    ax.add_patch(diamond)
    ax.text(4, diamond_y, '批次用完？', ha='center', va='center', fontsize=10, fontweight='bold', color=DARK)
    arrow(ax, 4, 4.3-0.35, 4, diamond_y+0.45)  # 进入菱形

    # 否 → 循环回去（左侧弯路）
    ax.annotate('', xy=(1.2, 8.3), xytext=(1.2, diamond_y),
                arrowprops=dict(arrowstyle='->', color=ARROW_C, lw=1.8))
    ax.plot([2.5, 1.2], [diamond_y, diamond_y], color=ARROW_C, lw=1.8)
    ax.annotate('', xy=(1.2, 8.3), xytext=(1.2, 8.3),
                arrowprops=dict(arrowstyle='->', color=ARROW_C, lw=1.8))
    ax.plot([1.2, 4-2.75], [8.3, 8.3], color=ARROW_C, lw=1.8)
    ax.text(1.0, (diamond_y+8.3)/2, '否', fontsize=9, color=ARROW_C, ha='center')

    # 是 → 验证阶段
    arrow(ax, 4, diamond_y-0.45, 4, 2.55)
    ax.text(4.15, diamond_y-0.6, '是', fontsize=9, color=ARROW_C)

    box(ax, 4, 2.2, 5.5, 0.65, '验证阶段\n计算 val_loss / val_acc', '#E3F2FD', BLUE, fontsize=10)
    arrow(ax, 4, 1.87, 4, 1.35)

    box(ax, 4, 1.05, 5.5, 0.55, '保存最优模型 checkpoint', GREEN_LT, GREEN, fontsize=10)

    fig.tight_layout()
    p = OUT / 'fig_train_flow.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close(fig)
    print(f'[OK] {p}')


# ══════════════════════════════════════════════════════════════════════════════
# 图4：查询/注册数据流图
# ══════════════════════════════════════════════════════════════════════════════
def make_query_flow():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor(WHITE)

    titles = ['查询流程', '注册流程']
    flows = [
        [   # 查询流程
            ('用户上传查询鼻纹图片', BLUE_LITE, BLUE),
            ('Next.js 临时上传至\nSupabase Storage → 获取 URL', '#E8F5E9', GREEN),
            ('携带查询URL + 全库档案URL\n批量调用 /compare 接口', ORANGE_LT, ORANGE),
            ('FastAPI + 孪生网络\n逐一计算相似度', BLUE_LITE, BLUE),
            ('返回置信度最高的\n匹配结果', GREEN_LT, GREEN),
            ('删除临时文件', '#FFF9C4', '#F9A825'),
        ],
        [   # 注册流程
            ('用户填写犬只信息\n并上传鼻纹图片', BLUE_LITE, BLUE),
            ('图片上传至\nSupabase Storage', '#E8F5E9', GREEN),
            ('与全库已有档案比对\n计算最高相似度', ORANGE_LT, ORANGE),
            ('相似度 > 50%？\n（重复注册检测）', '#FFF9C4', '#F9A825'),
            ('写入 PostgreSQL\n新档案', GREEN_LT, GREEN),
        ],
    ]
    branch_labels = [None, None, None, ('是→返回已有档案', '否↓')]

    for col, (ax, title, flow) in enumerate(zip(axes, titles, flows)):
        ax.set_xlim(0, 5); ax.set_ylim(0, len(flow)*1.3 + 0.5)
        ax.axis('off')
        ax.set_facecolor(WHITE)
        ax.text(2.5, len(flow)*1.3 + 0.2, title,
                ha='center', fontsize=13, fontweight='bold', color=DARK)

        for i, (label, fc, ec) in enumerate(flow):
            y = (len(flow) - 1 - i) * 1.3 + 0.5
            box(ax, 2.5, y, 4.2, 0.8, label, facecolor=fc, edgecolor=ec, fontsize=9.5)
            if i < len(flow) - 1:
                arrow(ax, 2.5, y - 0.4, 2.5, y - 0.9)

            # 注册流程菱形分支标注
            if col == 1 and i == 3:
                ax.text(4.0, y, '是→返回\n已有档案', fontsize=8, color=ORANGE,
                        ha='left', va='center')

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
