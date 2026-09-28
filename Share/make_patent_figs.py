# -*- coding: utf-8 -*-
"""
为专利交底书 4.2【技术侧】生成补充流程图 / 时序图（彩色轻装饰版 v2）。

风格：柔和配色 + 轻阴影 + 彩色分区标签；重点节点放大加粗、占据视觉中心。
输出（Share/ 目录，覆盖 v1）：
  patent_fig_stage1_dct_cache.png       阶段1  Lab+Alpha 四通道 DCT 特征提取与单图缓存校验流程图
  patent_fig_stage23_rf_kmeans.png      阶段2~4 随机森林降维、(MiniBatch)KMeans 聚类与 BallTree 构建（含增量更新）
  patent_fig_stage5a_query_sequence.png 阶段5a 单图查询时序图（两层索引）
  patent_fig_stage5b_scan_components.png 阶段5b 全量扫描与连通分量分组流程图

依赖：Pillow。脚本自带"文字溢出检测"，运行结束打印 WARN 列表。
"""
import math
import os

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMHEI = r'C:\Windows\Fonts\simhei.ttf'


def F(sz):
    return ImageFont.truetype(SIMHEI, sz)


F_MAIN = F(34)    # 普通框文字
F_SMALL = F(29)   # 消息/注释
F_TINY = F(26)    # 小标签
F_NODE = F(40)    # 示意图节点字母
F_TITLE = F(33)   # 分区标题
F_BIG = F(38)     # 重点框文字（放大）

INK = (30, 41, 59)          # #1E293B 文字
ARROW = (51, 65, 85)        # #334155 箭头
DASHC = (148, 163, 184)     # #94A3B8 虚线
SHADOW = (203, 213, 225)    # #CBD5E1 阴影
FRAME = (226, 232, 240)     # #E2E8F0 外框


def hx(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


PAL = {k: (hx(b), hx(f)) for k, (b, f) in {
    'blue':   ('#2563EB', '#DBEAFE'),
    'slate':  ('#475569', '#F1F5F9'),
    'amber':  ('#B45309', '#FEF3C7'),
    'green':  ('#047857', '#D1FAE5'),
    'violet': ('#6D28D9', '#EDE9FE'),
    'rose':   ('#BE123C', '#FFE4E6'),
    'teal':   ('#0F766E', '#CCFBF1'),
}.items()}

WARN = []


def text_lines(draw, lines, font):
    ws = [draw.textlength(ln, font=font) for ln in lines]
    lh = int(font.size * 1.5)
    return max(ws), lh * len(lines), lh


# ---------------------------------------------------------------- 基础图元
def dashed_line(draw, p1, p2, color=DASHC, dash=16, gap=11, width=3):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    ux, uy = dx / dist, dy / dist
    t = 0.0
    while t < dist:
        t2 = min(t + dash, dist)
        draw.line([p1[0] + ux * t, p1[1] + uy * t,
                   p1[0] + ux * t2, p1[1] + uy * t2], fill=color, width=width)
        t = t2 + gap


def arrow_head(draw, p_from, p_to, color=ARROW, size=24):
    dx, dy = p_to[0] - p_from[0], p_to[1] - p_from[1]
    d = math.hypot(dx, dy)
    if d == 0:
        return
    ux, uy = dx / d, dy / d
    px, py = -uy, ux
    draw.polygon([p_to,
                  (p_to[0] - ux * size + px * size * 0.55, p_to[1] - uy * size + py * size * 0.55),
                  (p_to[0] - ux * size - px * size * 0.55, p_to[1] - uy * size - py * size * 0.55)],
                 fill=color)


def arrow(draw, p1, p2, color=ARROW, dash=False, width=4, head=True):
    if dash:
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        d = math.hypot(dx, dy) or 1
        ex, ey = p2[0] - dx / d * 12, p2[1] - dy / d * 12
        dashed_line(draw, p1, (ex, ey), color=color, width=width)
    else:
        draw.line([p1, p2], fill=color, width=width)
    if head:
        arrow_head(draw, p1, p2, color=color)


def polyline(draw, pts, color=ARROW, dash=False, width=4, head=True):
    for i in range(len(pts) - 2):
        if dash:
            dashed_line(draw, pts[i], pts[i + 1], color=color, width=width)
        else:
            draw.line([pts[i], pts[i + 1]], fill=color, width=width)
    arrow(draw, pts[-2], pts[-1], color=color, dash=dash, width=width, head=head)


def node_box(draw, cx, cy, w, text, style='slate', shape='rect', emph=False, shadow=True):
    """流程框：柔和填充 + 彩色描边 + 轻阴影；emph=True 时字体放大加粗、边框加粗。"""
    bd, fl = PAL[style]
    f = F_BIG if emph else F_MAIN
    lines = text.split('\n')
    _, _, lh = text_lines(draw, lines, f)
    h = lh * len(lines) + (64 if emph else 56)
    # 溢出检测
    for ln in lines:
        tw = draw.textlength(ln, font=f)
        if tw > w - 26:
            WARN.append(f'OVERFLOW box[{text[:10]}...] line "{ln}" {tw:.0f} > {w - 26}')
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    bw = 6 if emph else 4
    if shape == 'diamond':
        x0, x1 = x0 - 30, x1 + 30
        y0, y1 = y0 - 10, y1 + 10
        if shadow:
            draw.polygon([(cx + 7, y0 + 9), (x1 + 7, cy + 9), (cx + 7, y1 + 9), (x0 + 7, cy + 9)],
                         fill=SHADOW)
        draw.polygon([(cx, y0), (x1, cy), (cx, y1), (x0, cy)], fill=fl, outline=bd, width=bw)
    else:
        r = h / 2 if shape == 'stadium' else 16
        if shadow:
            draw.rounded_rectangle([x0 + 7, y0 + 9, x1 + 7, y1 + 9], radius=r, fill=SHADOW)
        draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=fl, outline=bd, width=bw)
    sw = 2 if emph else 0
    y = cy - lh * len(lines) / 2
    for ln in lines:
        tw = draw.textlength(ln, font=f)
        draw.text((cx - tw / 2, y), ln, font=f, fill=INK, stroke_width=sw, stroke_fill=INK)
        y += lh
    return {'top': (cx, y0), 'bottom': (cx, y1), 'left': (x0, cy), 'right': (x1, cy)}


def chip(draw, cx, cy, text, style, font=F_TITLE):
    """实心彩色胶囊标签（白字）。"""
    bd, _ = PAL[style]
    w = draw.textlength(text, font=font) + 56
    h = font.size + 34
    draw.rounded_rectangle([cx - w / 2 + 4, cy - h / 2 + 4, cx + w / 2 + 4, cy + h / 2 + 4],
                           radius=h / 2, fill=SHADOW)
    draw.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                           radius=h / 2, fill=bd)
    draw.text((cx, cy), text, font=font, fill='white', anchor='mm')


def yn_pill(draw, cx, cy, txt):
    style = 'green' if txt == '是' else 'rose'
    bd, fl = PAL[style]
    f = F_TINY
    w = draw.textlength(txt, font=f) + 30
    h = f.size + 20
    draw.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                           radius=h / 2, fill=fl, outline=bd, width=3)
    draw.text((cx, cy), txt, font=f, fill=bd, anchor='mm')


def note_pill(draw, cx, cy, text, style='slate', font=F_TINY):
    """浅色注释胶囊。"""
    bd, fl = PAL[style]
    w = draw.textlength(text, font=font) + 36
    h = font.size + 24
    draw.rounded_rectangle([cx - w / 2 + 3, cy - h / 2 + 3, cx + w / 2 + 3, cy + h / 2 + 3],
                           radius=h / 2, fill=SHADOW)
    draw.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                           radius=h / 2, fill=fl, outline=bd, width=2)
    draw.text((cx, cy), text, font=font, fill=bd, anchor='mm')


def dashed_rounded_rect(draw, box, radius, color, width=3, dash=18, gap=12):
    x0, y0, x1, y1 = box

    def seg(p1, p2):
        dashed_line(draw, p1, p2, color=color, width=width, dash=dash, gap=gap)

    seg((x0 + radius, y0), (x1 - radius, y0))
    seg((x1 - radius, y1), (x0 + radius, y1))
    seg((x1, y0 + radius), (x1, y1 - radius))
    seg((x0, y1 - radius), (x0, y0 + radius))

    def arc(cx, cy, a0, a1):
        n = 12
        pts = [(cx + radius * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
                cy + radius * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]
        for i in range(n):
            seg(pts[i], pts[i + 1])

    arc(x1 - radius, y0 + radius, 270, 360)
    arc(x1 - radius, y1 - radius, 0, 90)
    arc(x0 + radius, y1 - radius, 90, 180)
    arc(x0 + radius, y0 + radius, 180, 270)


def panel(draw, box, fill, border, radius=26):
    draw.rounded_rectangle(box, radius=radius, fill=fill)
    dashed_rounded_rect(draw, box, radius, border)


def frame(draw, w, h):
    draw.rounded_rectangle([16, 16, w - 16, h - 16], radius=28, outline=FRAME, width=3)


def circle_node(draw, x, y, ch, style):
    bd, fl = PAL[style]
    r = 48
    draw.ellipse([x - r + 6, y - r + 8, x + r + 6, y + r + 8], fill=SHADOW)
    draw.ellipse([x - r, y - r, x + r, y + r], fill=fl, outline=bd, width=5)
    draw.text((x, y - 2), ch, font=F_NODE, fill=INK, anchor='mm')


# ---------------------------------------------------------------- 图12：阶段1
def fig_stage1():
    W, H = 2060, 2260
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)
    frame(d, W, H)
    LX, RX, BUS = 540, 1600, 1250

    chip(d, LX, 108, '缓存快速路径 · 毫秒级', 'green')
    chip(d, RX, 108, '未命中缓存 · 重新提取', 'blue')

    n_in = node_box(d, LX, 240, 560, '输入：图片文件', 'blue', 'stadium')
    dia1 = node_box(d, LX, 480, 760, '缓存文件存在且\nDCT 参数一致?', 'amber', 'diamond')
    dia2 = node_box(d, LX, 790, 760, 'mtime 一致或\nCRC32 校验通过?', 'amber', 'diamond')
    n_load = node_box(d, LX, 1070, 620, '加载缓存 DCT 特征\n(含文件哈希等元数据)', 'green')

    r1 = node_box(d, RX, 450, 700, '读取图片并统一缩放至 256×256', 'blue')
    r2 = node_box(d, RX, 700, 700, 'RGB+Alpha 转换为\nLab+Alpha 四通道', 'blue')
    r3 = node_box(d, RX, 950, 700, 'L / a / b / A 四通道\n分别做 2D-DCT', 'blue')
    r4 = node_box(d, RX, 1200, 700, '取每通道左上角 32×32\n低频系数 (默认 N=32)', 'blue')
    r5 = node_box(d, RX, 1450, 700, '四通道展平拼接\n4096 维特征向量', 'violet', emph=True)
    r6 = node_box(d, RX, 1700, 700, '写入单图特征缓存\n(按文件路径 MD5 存 .pkl)', 'teal')

    n_out = node_box(d, 1070, 2130, 880, '输出：4096 维 DCT 特征向量', 'blue', 'stadium', emph=True)

    # 左栏
    arrow(d, n_in['bottom'], dia1['top'])
    arrow(d, dia1['bottom'], dia2['top'])
    yn_pill(d, LX + 62, (dia1['bottom'][1] + dia2['top'][1]) / 2, '是')
    arrow(d, dia2['bottom'], n_load['top'])
    yn_pill(d, LX + 62, (dia2['bottom'][1] + n_load['top'][1]) / 2, '是')
    polyline(d, [n_load['bottom'], (LX, 2130), (624, 2130)])

    # 汇流总线 → 重新提取
    d.line([dia1['right'], (BUS, 480)], fill=ARROW, width=4)
    d.line([dia2['right'], (BUS, 790)], fill=ARROW, width=4)
    d.line([(BUS, 790), (BUS, 340)], fill=ARROW, width=4)
    d.line([(BUS, 340), (RX, 340)], fill=ARROW, width=4)
    arrow(d, (RX, 340), r1['top'])
    yn_pill(d, 1085, 448, '否')
    yn_pill(d, 1085, 758, '否')

    for a, b in [(r1, r2), (r2, r3), (r3, r4), (r4, r5), (r5, r6)]:
        arrow(d, a['bottom'], b['top'])
    polyline(d, [r6['bottom'], (RX, 2130), (1516, 2130)])

    # 注释
    note_pill(d, 890, 1560, '信息保留率约为 8×8 哈希的 64 倍', 'violet')
    dashed_line(d, (1128, 1560), (1242, 1560), color=PAL['violet'][0])

    img.save(os.path.join(OUT_DIR, 'patent_fig_stage1_dct_cache.png'))
    print('saved fig_stage1')


# ---------------------------------------------------------------- 图13：阶段2~4
def fig_stage23():
    W, H = 2340, 2680
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)
    frame(d, W, H)
    CX = 800

    chip(d, 205, 740, '阶段2 · 特征降维', 'violet')
    chip(d, 205, 1560, '阶段3 · 图片聚类', 'blue')
    chip(d, 205, 2110, '阶段4 · 空间索引', 'teal')

    n1 = node_box(d, CX, 150, 760, '全量图片的 4096 维 DCT 特征', 'blue', 'stadium')
    n2 = node_box(d, CX, 380, 820, 'MiniBatchKMeans 粗聚类\n生成伪标签(约 50 簇)', 'violet')
    n3 = node_box(d, CX, 610, 820, '以伪标签训练随机森林分类器\n(100 棵决策树)', 'violet')
    n4 = node_box(d, CX, 840, 820, '按特征重要度排序\n保留 Top-500 个原始维度', 'violet', emph=True)
    n5 = node_box(d, CX, 1070, 820, '得到 500 维特征\n特征选择模型持久化保存', 'violet', emph=True)
    dia = node_box(d, CX, 1350, 580, '样本量 > 10000 ?', 'amber', 'diamond')
    n6 = node_box(d, CX, 1640, 820, 'MiniBatchKMeans 聚类\n(小批量 1000 张/批)', 'blue')
    nk = node_box(d, 1860, 1350, 520, '标准 KMeans', 'slate')
    n7 = node_box(d, CX, 1880, 820, '聚成 K≈4000 个簇\n聚类模型持久化缓存', 'blue', emph=True)
    n8 = node_box(d, CX, 2110, 820, '按簇独立构建 BallTree\n(启动时按需构建, 秒级)', 'teal')

    for a, b in [(n1, n2), (n2, n3), (n3, n4), (n4, n5), (n5, dia)]:
        arrow(d, a['bottom'], b['top'])
    arrow(d, dia['bottom'], n6['top'])
    yn_pill(d, CX + 58, (dia['bottom'][1] + n6['top'][1]) / 2, '是')
    arrow(d, dia['right'], nk['left'])
    yn_pill(d, 1305, 1310, '否')
    polyline(d, [nk['bottom'], (1860, 1880), n7['right']])
    arrow(d, n6['bottom'], n7['top'])
    arrow(d, n7['bottom'], n8['top'])

    note_pill(d, 1650, 1070, '实测准确率优于 PCA / 方差 / 单变量统计', 'violet')
    dashed_line(d, (1218, 1070), (1400, 1070), color=PAL['violet'][0])

    # ---- 增量更新分区 ----
    panel(d, (170, 2260, 2210, 2600), hx('#ECFDF5'), PAL['green'][0])
    chip(d, 1190, 2260, '增量更新 · 每日迭代，无需全量重训', 'green')

    boxes_txt = ['每日新增/修改图片\n(占比 <10%)',
                 '变化图提取 DCT 特征\n(命中单图缓存)',
                 '加载随机森林模型\n降维至 500 维',
                 'predict 分配簇标签\npartial_fit 微调模型',
                 '增量重建受影响簇\nBallTree 索引']
    bw, gap = 385, 26
    total = bw * 5 + gap * 4
    sx = (170 + 2210 - total) / 2
    centers = [sx + bw / 2 + i * (bw + gap) for i in range(5)]
    boxes = [node_box(d, cx_, 2450, bw, t, 'green') for cx_, t in zip(centers, boxes_txt)]
    for a, b in zip(boxes, boxes[1:]):
        arrow(d, a['right'], b['left'])
    # 反馈回指 BallTree 构建
    polyline(d, [(centers[-1], 2371), (centers[-1], 2215), (CX, 2215), (CX, 2189)],
             color=PAL['green'][0], dash=True)

    img.save(os.path.join(OUT_DIR, 'patent_fig_stage23_rf_kmeans.png'))
    print('saved fig_stage23')


# ---------------------------------------------------------------- 图14：单图查询时序图
def fig_query_seq():
    W, H = 2820, 1730
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)
    frame(d, W, H)

    # 阶段背景带
    bands = [(250, 665, '#F1F5F9', '① 特征提取与降维'),
             (665, 915, None, '② 簇定位'),
             (915, 1430, '#F1F5F9', '③ 簇内检索与合并')]
    for y0, y1, fill, name in bands:
        if fill:
            d.rectangle([40, y0, 2780, y1], fill=hx(fill))
        d.line([(40, y0), (2780, y0)], fill=FRAME, width=2)
        chip(d, 230, y0, name, 'slate', font=F_TINY)
    d.line([(40, 1430), (2780, 1430)], fill=FRAME, width=2)

    actors = [('调用方\n(CLI / 界面)', 400, 'blue'),
              ('特征提取模块\n(DCT + 缓存)', 1060, 'violet'),
              ('随机森林\n降维模型', 1620, 'teal'),
              ('簇级索引\n(4000 簇中心)', 2170, 'amber'),
              ('簇内索引\n(BallTree)', 2640, 'green')]
    lifeline_bottom = 1480
    xs = []
    for name, x, style in actors:
        bd, _ = PAL[style]
        lines = name.split('\n')
        wmax, _, lh = text_lines(d, lines, F_MAIN)
        w = max(wmax, 250) + 56
        h = lh * len(lines) + 52
        cy = 130
        dashed_line(d, (x, 200), (x, lifeline_bottom))
        draw_box = (x, cy, w, h, bd)
        d.rounded_rectangle([x - w / 2 + 6, cy - h / 2 + 8, x + w / 2 + 6, cy + h / 2 + 8],
                            radius=16, fill=SHADOW)
        d.rounded_rectangle([x - w / 2, cy - h / 2, x + w / 2, cy + h / 2], radius=16, fill=bd)
        yy = cy - lh * len(lines) / 2
        for ln in lines:
            tw = d.textlength(ln, font=F_MAIN)
            d.text((x - tw / 2, yy), ln, font=F_MAIN, fill='white')
            yy += lh
        xs.append(x)
    A, B, C, D, E = xs
    VIOLET = PAL['violet'][0]

    def msg(x1, x2, y, text, ret=False, emph=False):
        if ret:
            color = hx('#BE123C') if emph else (100, 116, 139)
            arrow(d, (x1, y), (x2, y), color=color, dash=True, width=4 if emph else 3)
        else:
            blue = hx('#2563EB')
            d.line([(x1, y), (x2 - (26 if x2 > x1 else -26), y)], fill=blue, width=4)
            arrow_head(d, (x1, y), (x2, y), color=blue)
        f = F_SMALL
        w = d.textlength(text, font=f) + 30
        h = f.size + 20
        cx = (x1 + x2) / 2
        if emph:
            bd = PAL['rose'][0]
            d.rounded_rectangle([cx - w / 2, y - 42 - h / 2, cx + w / 2, y - 42 + h / 2],
                                radius=h / 2, fill=bd)
            d.text((cx, y - 42), text, font=f, fill='white', anchor='mm')
        else:
            bd = hx('#2563EB') if not ret else (100, 116, 139)
            d.rounded_rectangle([cx - w / 2, y - 42 - h / 2, cx + w / 2, y - 42 + h / 2],
                                radius=h / 2, fill='white', outline=bd, width=2)
            d.text((cx, y - 42), text, font=f, fill=INK, anchor='mm')

    def self_msg(x, y, text):
        pts = [(x, y), (x + 92, y), (x + 92, y + 60)]
        d.line(pts, fill=VIOLET, width=4)
        arrow(d, (x + 92, y + 60), (x + 4, y + 60), color=VIOLET)
        w = d.textlength(text, font=F_TINY) + 26
        h = F_TINY.size + 18
        lx = x + 108
        d.rounded_rectangle([lx, y + 30 - h / 2, lx + w, y + 30 + h / 2],
                            radius=h / 2, fill='white', outline=VIOLET, width=2)
        d.text((lx + 13, y + 30), text, font=F_TINY, fill=INK, anchor='lm')

    msg(A, B, 300, '输入查询图片')
    self_msg(B, 370, '缓存校验：命中则读取 / 未命中则提取 DCT 特征（4096 维）')
    msg(B, C, 520, '4096 维 DCT 特征')
    msg(C, B, 640, '500 维降维特征', ret=True)
    msg(B, D, 750, 'predict(500 维特征)')
    msg(D, B, 870, '所属簇 ID + 最近 1~3 个邻簇', ret=True)
    msg(B, E, 990, '簇内 k 近邻查询')
    msg(E, B, 1110, 'Top-K 候选（路径 + 距离）', ret=True)
    self_msg(B, 1180, '跨簇扩展搜索、合并去重、按距离重排')
    msg(B, A, 1390, 'Top-K 相似图片列表（路径 + 相似度）', ret=True, emph=True)

    # 底部注解
    d.rounded_rectangle([690, 1560, 2130, 1656], radius=20, fill=hx('#DBEAFE'),
                        outline=PAL['blue'][0], width=3)
    d.text((1410, 1608), '两层索引：簇级定位 O(K) + 簇内 O(log n) · 单图查询毫秒级',
           font=F_SMALL, fill=INK, anchor='mm')

    img.save(os.path.join(OUT_DIR, 'patent_fig_stage5a_query_sequence.png'))
    print('saved fig_query_seq')


# ---------------------------------------------------------------- 图15：阶段5b
def fig_scan_cc():
    W, H = 2480, 2160
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)
    frame(d, W, H)
    CX = 620

    n1 = node_box(d, CX, 150, 780, '输入：图片库路径 + 相似度阈值', 'blue', 'stadium')
    n2 = node_box(d, CX, 350, 780, '相似度阈值 → 距离阈值 d', 'slate')
    n3 = node_box(d, CX, 570, 950, '逐簇处理（簇间天然可并行）\n避免全局 O(N^2) 比对', 'blue')
    n4 = node_box(d, CX, 800, 950, 'BallTree.query_radius(d)\n获取簇内每图近邻 → 边集合', 'blue', emph=True)
    n5 = node_box(d, CX, 1030, 950, '以图片为顶点、近邻关系为边\n构建无向相似图 G', 'slate')
    n6 = node_box(d, CX, 1260, 950, 'DFS 提取所有连通分量', 'violet', emph=True)
    n7 = node_box(d, CX, 1470, 950, '每个连通分量 = 一个相似图片组', 'violet', emph=True)
    n8 = node_box(d, CX, 1690, 950, '计算组特征中心\n成员到中心距离 → 相似度得分', 'slate')
    n9 = node_box(d, CX, 1930, 860, '输出：相似组列表与报告\n(组内按相似度得分排序)', 'green', 'stadium', emph=True)

    for a, b in [(n1, n2), (n2, n3), (n3, n4), (n4, n5), (n5, n6), (n6, n7), (n7, n8), (n8, n9)]:
        arrow(d, a['bottom'], b['top'])

    # ---- 右侧示意区 ----
    panel(d, (1360, 400, 2430, 1760), hx('#F8FAFC'), PAL['slate'][0])
    chip(d, 1895, 400, '相似性传递示意', 'slate')

    # 组1：A-B-C（先画边，后画节点）
    A, B, C = (1530, 880), (1780, 730), (2030, 880)
    d.line([A, B], fill=PAL['blue'][0], width=4)
    d.line([B, C], fill=PAL['blue'][0], width=4)
    dashed_line(d, A, C, color=DASHC, dash=12, gap=10)
    circle_node(d, *A, 'A', 'blue')
    circle_node(d, *B, 'B', 'blue')
    circle_node(d, *C, 'C', 'blue')
    note_pill(d, 1780, 958, '距离 > d', 'rose')
    d.text((1895, 1040), 'A~B、B~C 相似 → 归为一组\n（即使 A 与 C 距离 > d）',
           font=F_SMALL, fill=INK, anchor='mm', align='center')

    # 组2：D-E
    Dp, Ep = (2190, 1250), (2330, 1100)
    d.line([Dp, Ep], fill=PAL['green'][0], width=4)
    circle_node(d, *Dp, 'D', 'green')
    circle_node(d, *Ep, 'E', 'green')
    d.text((2190, 1340), 'D~E → 一组', font=F_TINY, fill=INK, anchor='mm')

    # 孤立点 F
    Fp = (1570, 1380)
    circle_node(d, *Fp, 'F', 'slate')
    d.text((1650, 1380), 'F 无近邻 → 不成组', font=F_TINY, fill=INK, anchor='lm')

    # 图例
    d.line([(1450, 1660), (1570, 1660)], fill=PAL['blue'][0], width=4)
    d.text((1590, 1660), '距离 ≤ d（相似边）', font=F_TINY, fill=INK, anchor='lm')
    dashed_line(d, (1980, 1660), (2100, 1660), color=DASHC)
    d.text((2120, 1660), '距离 > d（非边）', font=F_TINY, fill=INK, anchor='lm')

    arrow(d, (1360, 1260), n6['right'])
    note_pill(d, 1750, 1930, '无需预设组数 · 自动发现所有真实相似组', 'violet')

    img.save(os.path.join(OUT_DIR, 'patent_fig_stage5b_scan_components.png'))
    print('saved fig_scan_cc')


if __name__ == '__main__':
    fig_stage1()
    fig_stage23()
    fig_query_seq()
    fig_scan_cc()
    if WARN:
        print('---- WARNINGS ----')
        for w in WARN:
            print(' ', w)
    else:
        print('layout check: OK, no overflow')
    print('all done ->', OUT_DIR)
