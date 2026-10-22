r"""图上标注：把"检出的一堆同心边"画到原图上，每条边带编号与数值，用引线牵出来。

为什么单独一个文件：CLI 与 GUI 都要用它，而且它只做排版、不碰数值。
设计意图见 `docs/通用量测模式计划.md`：**图上标注是一等公民**——不命名之后，
"这条边是哪个尺寸"只能靠人对着图认，图就是主要输出。

两条硬约束（都是这个项目踩过的坑，别改回去）：
1. **图上文字只能是 ASCII。** `cv2.putText` 用的是 Hershey 字体，画不了中文（会变成一串问号）。
   所以极性写成 `B>D`（亮→暗）/ `D>B`（暗→亮）；中文说明留给命令行与界面的文字部分。
2. **存盘必须走 `cv2.imencode` + `tofile`。** `cv2.imwrite` 遇中文路径不报错、只返回 False，
   而本项目的报告与预览文件名就是中文（见 AGENTS.md §5）。
"""
import cv2
import numpy as np

# 颜色：外缘沿用现有调试图的绿；两种极性分别用红/品红（与 detect 的调试图一致）
COLOR_OUTER = (0, 255, 0)
COLOR_SRC = "outer"
COLOR_B2D = (0, 0, 255)      # 亮→暗（下降沿）
COLOR_D2B = (255, 0, 255)    # 暗→亮（上升沿）

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9
FONT_THICK = 2
PAD = 14             # 边距
LEADER = 30          # 引线长度（从弧上往外牵出这么远，再排文字）
FAN_START = 200.0    # 标签散布的起始角度（图像坐标，y 向下）

# 图上文字**只留编号与数值**：极性已经由颜色区分（红=亮→暗、品红=暗→亮），再写一遍
# 只会让标签变长。实测标签长 350px 时，在半径 900px 的圈上就占 22°，13 个标签必然打架。


def _edge_color(edge):
    if edge["src"] == COLOR_SRC:
        return COLOR_OUTER
    return COLOR_B2D if edge["pol"] == -1 else COLOR_D2B


def _anchor(edge, ang_rad):
    """椭圆上取一个锚点（引线的落点）。用椭圆参数方程，与外圈的朝向无关。"""
    cx, cy = edge["cx"], edge["cy"]
    a, b = edge["a1"] / 2, edge["a2"] / 2
    t = np.deg2rad(edge["ang"])
    ct, st = np.cos(t), np.sin(t)
    x = cx + a * np.cos(ang_rad) * ct - b * np.sin(ang_rad) * st
    y = cy + a * np.cos(ang_rad) * st + b * np.sin(ang_rad) * ct
    return float(x), float(y)


def _fan_angles(n):
    """把 n 个标签沿整圈等角摊开。

    为什么必须摊开：同心椭圆的"最左点"几乎都落在同一高度，标签若挤在一列里，
    引线要从中间牵出上千像素、还互相交叉（实测上一版最长 1116px）。按角度分开之后，
    每个标签贴着自己那条弧，引线只剩几十像素。整圈而不是半圈，是因为标签有宽度
    （角向占位），半圈放不下十几个。
    """
    if n <= 0:
        return []
    if n == 1:
        return [FAN_START]
    step = 360.0 / n
    return [(FAN_START + i * step) % 360.0 for i in range(n)]


def _label_text(e, ratio):
    mm = "" if ratio is None else f" {e['dia_px'] * ratio:.2f}mm"
    return f"#{e['idx']} {e['dia_px']:.1f}px{mm}"


def _place_labels(edges, ratio, w, h):
    """算出每个标签的文字框，并解开互相压盖。返回 [(编号, 文字, text_x, base_y, 锚点, 引线终点)]。

    碰撞处理是必须的：标签排到一定密度后，不同半径、不同角度的两个标签仍会撞上
    （实测 13 个里有 2 对）。这里按"下压上"逐对推开、再收回画面内，反复几轮至收敛。
    """
    items = []
    for e, phi in zip(edges, _fan_angles(len(edges))):
        pr = np.deg2rad(phi)
        dx, dy = np.cos(pr), np.sin(pr)
        ax, ay = _anchor(e, pr)
        tx, ty = ax + dx * LEADER, ay + dy * LEADER
        text = _label_text(e, ratio)
        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
        text_x = tx - tw if dx < 0 else tx
        text_x = min(max(text_x, PAD), max(PAD, w - PAD - tw))
        items.append({"idx": e["idx"], "text": text, "x": text_x, "y": ty,
                      "tw": tw, "th": th, "ax": ax, "ay": ay, "tx": tx, "ty": ty})
    for _ in range(20):
        moved = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if (a["x"] < b["x"] + b["tw"] and b["x"] < a["x"] + a["tw"]
                        and a["y"] < b["y"] + b["th"] and b["y"] < a["y"] + a["th"]):
                    push = min(a["y"] + a["th"], b["y"] + b["th"]) - max(a["y"], b["y"]) + 4
                    if (b["y"] + b["th"] / 2) < (a["y"] + a["th"] / 2):
                        push = -push
                    b["y"] += push
                    moved = True
        for it in items:                      # 收回画面内（画布夹紧）
            it["y"] = float(np.clip(it["y"], PAD, h - PAD - it["th"]))
        if not moved:
            break
    return [(it["idx"], it["text"], it["x"], it["y"] + it["th"], (it["ax"], it["ay"]))
            for it in items]


def draw_listing(img, res, ratio=None):
    """在图上画出 `concentric.list_concentric` 的清单，返回标注好的图（新数组）。

    ratio 给了就把 mm 也写进标签（mm = px × 系数）。清单为空时返回原图的副本。
    """
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    edges = res.get("edges") or []
    if not edges:
        return out
    h, w = out.shape[:2]

    for e in edges:                            # 每条边先画出它自己的椭圆
        cv2.ellipse(out, ((e["cx"], e["cy"]), (e["a1"], e["a2"]), e["ang"]),
                    _edge_color(e), 4)

    for idx, text, text_x, base_y, (ax, ay) in _place_labels(edges, ratio, w, h):
        e = edges[idx - 1]
        col = _edge_color(e)
        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
        # 引线牵到文字框**靠近弧的那一侧**：锚点在文字左边就连左边缘，在右边就连右边缘。
        # （写反了会变成"横穿整条文字"的引线，实测差 300px——这个坑是量出来的。）
        edge_x = text_x if ax < text_x else text_x + tw
        cv2.line(out, (int(ax), int(ay)), (int(edge_x), int(base_y - th / 2)), col, 2)
        cv2.circle(out, (int(ax), int(ay)), 5, col, -1)
        cv2.putText(out, text, (int(text_x), int(base_y)), FONT, FONT_SCALE, col,
                    FONT_THICK, cv2.LINE_AA)
    return out


def save_png(path, img):
    """存 PNG。**必须走 imencode + tofile**：cv2.imwrite 遇中文路径会静默失败（返回 False）。"""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"PNG 编码失败: {path}")
    from pathlib import Path
    Path(path).write_bytes(buf.tobytes())
    return Path(path)
