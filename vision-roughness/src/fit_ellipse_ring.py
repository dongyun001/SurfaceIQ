r"""圆环零件完整测量管线（find_circles 原型）：
    外圈/中心孔:  72射线亚像素边缘 + 椭圆拟合
    6 螺栓孔:     粗暗斑挖模板 -> 模板匹配峰值 -> 等角共圆校验(滤假峰/补缺位)
                  -> 每孔 48射线亚像素椭圆拟合

用法:
    python src\fit_ellipse_ring.py <图片路径>
输出: 外圈/中心孔椭圆、6孔直径(px)；调试图 debug/fit_debug.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np

N_RAY = 72

# 外圈候选的可信门槛（自适应定位用；实测依据见 docs/自适应定位改造计划.md）
OUTER_COVER_MIN = 0.6        # 保留点数 ≥ 射线数×此比例（误锁只有 22/72，正确为 54~72/72）
OUTER_AXIS_RATIO_MAX = 1.8   # 外圈长短轴比上限（斜视视角的物理上限）
OUTER_MAX_CANDIDATES = 60    # 候选上限，防自适应搜索跑飞
OUTER_CENTER_MIN_RATIO = 0.5  # 圆心最强径向峰 < 全局最强峰×此比例就丢弃（零件不在这儿）
OUTER_OUT_OF_FRAME_RATIO = 0.92  # 检出圆 < 更大同心圆的此比例 => 零件出画、检出的是内孔
OUTER_CONCENTRIC_TOL = 0.15      # 判"同心"的圆心偏移上限（×检出半径）
INNER_CENTER_MAX_SHIFT = 0.15  # 中心孔中心相对外圈中心的容许偏移（×外圈半径）
INNER_RATIO_TOL = 0.035       # 实测孔径比与档案真值(id_mm/od_mm)的容许偏差
INNER_RING_HALF = 0.035       # 内圈两道边的搜索半宽（归一化半径）
INNER_RING_MIN_GAP = 0.005    # 两道边至少差这么多（×外圈半径），否则视为同一条边


# ---------------- 射线亚像素边缘 ----------------

def ray_profile(gray, cx, cy, ang, max_r):
    d = np.arange(0, max_r, 0.5)
    xs = (cx + np.cos(ang) * d).astype(int).clip(0, gray.shape[1] - 1)
    ys = (cy + np.sin(ang) * d).astype(int).clip(0, gray.shape[0] - 1)
    return d, gray[ys, xs].astype(float)


def find_edge_points(gray, cx, cy, r_lo, r_hi, polarity=-1, min_grad=1.2, n_ray=N_RAY,
                     mode="steepest"):
    """射线法找亚像素边缘点集。polarity=-1 下降沿(亮->暗)；+1 上升沿。

    mode='steepest': 窗口内最陡梯度(默认, 用于外圈/中心孔等单一结构);
    mode='outer_first': 从窗口外侧往内找第一个满足极性的边(用于孔外缘)。
    """
    pts = []
    for k in range(n_ray):
        ang = 2 * np.pi * k / n_ray
        d, g = ray_profile(gray, cx, cy, ang, r_hi * 1.08)
        m = (d >= r_lo) & (d <= r_hi)
        di, gi = d[m], g[m]
        if len(gi) < 8:
            continue
        grads = np.gradient(gi)
        if mode == "outer_first":
            j = None
            for jj in range(len(grads) - 3, 3, -1):
                if polarity < 0:
                    if grads[jj] < -min_grad and grads[jj] <= grads[jj - 1] and grads[jj] <= grads[jj + 1]:
                        j = jj
                        break
                else:
                    if grads[jj] > min_grad and grads[jj] >= grads[jj - 1] and grads[jj] >= grads[jj + 1]:
                        j = jj
                        break
            if j is None:
                continue
            di_j = di[j] - 0.5
        else:
            j = int(np.argmin(grads)) if polarity < 0 else int(np.argmax(grads))
            score = -grads[j] if polarity < 0 else grads[j]
            if score < min_grad:
                continue
            j = min(max(j, 1), len(grads) - 2)
            di_j = di[j]
        a, b, c = grads[j - 1], grads[j], grads[j + 1]
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0
        x, y = cx + np.cos(ang) * (di_j + delta), cy + np.sin(ang) * (di_j + delta)
        pts.append((x, y))
    return np.array(pts).reshape(-1, 2) if pts else np.zeros((0, 2))


def filter_and_fit(pts, cx, cy, r0, outlier=0.09, min_pts=20):
    """圆等价归一化距离滤离群 + 椭圆拟合迭代。返回(cv2 ell, 点数)。"""
    pts = np.asarray(pts, float)
    r = r0
    for _ in range(3):
        if len(pts) < min_pts:
            return None, 0
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        keep = np.abs(d - r) <= r * outlier
        pts = pts[keep]
        if len(pts) < min_pts:
            return None, 0
        ell = cv2.fitEllipse(pts.astype(np.float32))
        (cx, cy), (a1, a2), ang = ell
        r = (a1 + a2) / 4
    return ell, len(pts)


# ---------------- 椭圆归一化坐标系(用于共圆校验) ----------------

def to_norm(pts, ocx, ocy, oa1, oa2, oang):
    """图像坐标 -> 外圈椭圆归一化单位圆坐标。"""
    ca, sa = np.cos(np.deg2rad(-oang)), np.sin(np.deg2rad(-oang))
    pts = np.asarray(pts, float)
    dx = (pts[..., 0] - ocx) * ca - (pts[..., 1] - ocy) * sa
    dy = (pts[..., 0] - ocx) * sa + (pts[..., 1] - ocy) * ca
    return dx / (oa1 / 2), dy / (oa2 / 2)


def from_norm(nx, ny, ocx, ocy, oa1, oa2, oang):
    """归一化单位圆坐标 -> 图像坐标。"""
    ca, sa = np.cos(np.deg2rad(oang)), np.sin(np.deg2rad(oang))
    dx, dy = nx * (oa1 / 2), ny * (oa2 / 2)
    return ocx + dx * ca - dy * sa, ocy + dx * sa + dy * ca


# ---------------- 螺栓孔: 模板匹配 + 等角共圆校验 ----------------

ROUGH_DARK_THRS = (0.62, 0.70, 0.74, 0.55)  # 逐档试；0.55 是"画面变暗"的最后一档兜底


def _best_dark_blob(norm, band, thr, min_area=3000, max_side=None):
    """在 band 内按 norm<thr 找面积够、够圆、且**不像整圈暗槽**的暗连通域。

    **必须用 RETR_LIST**：环面上的暗槽会连成闭合的环，`RETR_EXTERNAL` 只返回最外层
    轮廓，环里面那 6 个沉孔会被当成"洞"整个丢掉（2026-10-22 实测：所有阈值下都只有
    一个 419 万 px 的环形轮廓，沉孔一个都出不来）。
    max_side：外接框长边上限（px）。孔不可能跟环带一个量级，用来把环形暗槽本身挡掉。
    """
    dark = cv2.bitwise_and(((norm < thr).astype(np.uint8) * 255), band)
    cnts, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    best = None
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        x, y, w2, h2 = cv2.boundingRect(c)
        if max_side is not None and max(w2, h2) > max_side:
            continue
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * a / (per * per) if per else 0
        if circ < 0.55:
            continue
        score = circ * np.sqrt(a)
        if best is None or score > best[0]:
            best = (score, c)
    return best


def rough_templates(sm, band, min_area=3000):
    """环带内粗找暗连通域，按圆度x面积选最像孔的区域做模板。

    阈值从前写死 0.62。换机位/光照后沉孔可能只比周围暗 25~30%（norm≈0.7），
    0.62 下会被螺钉头高光切成碎块、圆度不够 → 模板为 None → 螺栓孔一个都测不到
    （2026-10-21 实拍：0.62/0.66 达标暗斑 0 个，0.70 起才有）。
    反过来，相机抬高/画面变暗后沉孔可能更暗（norm 0.55~0.62），0.62 又够不着
    （2026-10-22 抬高机位实测：只有 0.55 档能分出 3 个沉孔）。
    所以按 `ROUGH_DARK_THRS` 逐档试；0.55 放在**最后**当"画面变暗"的兜底——先走原来的
    0.62/0.70/0.74，原有图（含回归用例）不受影响。每档都要过下面的面积/尺寸/圆度筛选。

    另外要挡住"整圈暗槽"：环面上的暗槽会连成一个外接框接近整圈的大暗斑，圆度也高，
    评分必然压过沉孔（2026-10-22 实测外接框 2300px、圆度 0.89，模板因此选错、
    6 个孔全变成"位置推定"）。孔不可能跟环带一个量级，所以外接框长边超过环带半径的
    35% 直接不要。
    """
    bg = cv2.blur(sm, (201, 201))
    norm = sm / np.maximum(bg, 1)
    ys, xs = np.nonzero(band)
    if len(xs) == 0:
        return None
    max_side = 0.35 * 0.5 * max(np.ptp(xs), np.ptp(ys))
    best = None
    for thr in ROUGH_DARK_THRS:
        best = _best_dark_blob(norm, band, thr, min_area, max_side)
        if best is not None:
            break
    if best is None:
        return None
    x, y, w2, h2 = cv2.boundingRect(best[1])
    cx, cy = int(x + w2 / 2), int(y + h2 / 2)
    t = sm[cy - 125:cy + 125, cx - 125:cx + 125].astype(np.float32)
    if t.shape != (250, 250):
        t = cv2.resize(t, (250, 250))
    return t - t.mean()


def template_locate(sm, template, expect=12, scale=3):
    """缩图模板匹配，返回峰 (原图x, 原图y, 响应)，按响应降序。"""
    h, w = sm.shape
    ts = cv2.resize(template, (template.shape[1] // scale, template.shape[0] // scale),
                    interpolation=cv2.INTER_AREA)
    ts = ts - ts.mean()
    s = cv2.resize(sm, (w // scale, h // scale), interpolation=cv2.INTER_AREA).astype(np.float32)
    res = cv2.matchTemplate(s, ts, cv2.TM_CCOEFF_NORMED)
    peaks = []
    rr = res.copy()
    while True:
        _, mx, _, loc = cv2.minMaxLoc(rr)
        if mx < 0.30:
            break
        x = loc[0] * scale + ts.shape[1] // 2 * scale
        y = loc[1] * scale + ts.shape[0] // 2 * scale
        peaks.append((x, y, float(mx)))
        cv2.circle(rr, loc, 150, -1, -1)
        if len(peaks) >= expect:
            break
    return peaks


def ngon_slots(cands, ocx, ocy, oa1, oa2, oang, n=6, tol_ang=13, tol_r=0.15):
    """n 等分假设检验: 以命中数最高的候选为相位锚, 生成 n 个等角槽位。

    返回 [(期望角, 归一化半径, 峰orNone)]。**峰为 None 表示这个槽位是"位置推定"的**，
    调用方必须把它标出来，不能让推定位置的结果看起来跟实测的一样。

    原来写死 6 孔 60°（`hexagon_slots`）。实测那份合成数据说明写死会怎么坏：
    12 孔零件下 6 个槽位**全被真孔填满**、一个都不留空，于是报告干脆利落地列 6 个孔，
    而零件有 12 个——没有留空、没有失败、没有任何提示。孔数改为从档案来（缺省才推断）。
    整体旋转不影响：相位锚是从数据里选的。
    """
    if not cands:
        return None
    step = 360.0 / n
    nx, ny = to_norm([(c[0], c[1]) for c in cands], ocx, ocy, oa1, oa2, oang)
    r2 = np.hypot(nx, ny)
    th2 = (np.degrees(np.arctan2(ny, nx)) + 360) % 360
    best_i, best_score = -1, -1
    for i in range(len(cands)):
        hits = 0
        for j in range(len(cands)):
            if i == j:
                continue
            dmod = ((th2[j] - th2[i]) % step + step) % step
            dd = min(dmod, step - dmod)            # 与锚+n·k 的最短角距
            if dd < tol_ang and abs(r2[j] - r2[i]) / r2[i] < tol_r:
                hits += 1
        score = hits + cands[i][2] * 0.5               # 同命中数时响应高的优先
        if score > best_score:
            best_i, best_score = i, score
    p0 = cands[best_i]
    th0, r0 = th2[best_i], r2[best_i]
    slots = []
    for k in range(n):
        exp_ang = (th0 + step * k) % 360
        dd = (th2 - exp_ang + 180) % 360 - 180       # 绝对角差(最近360)
        m = (np.abs(dd) < tol_ang) & (np.abs(r2 - r0) / r0 < tol_r) & \
            (np.array([c[2] for c in cands]) >= 0.35)
        got = [c for c, ok in zip(cands, m) if ok]
        best = max(got, key=lambda c: c[2]) if got else None
        slots.append((exp_ang, r0, best))
    return slots


# ---------------- 螺栓孔直径: 螺丝头锚定 + 沉孔口阈值交叉 + 净弧椭圆拟合 ----------------

def hole_head_center(norm, x, y, win=160, thr=0.72):
    """暗斑内被包围的亮区(螺丝头)质心 = 孔锚点。

    沉孔里有斜置螺丝头(亮), 外圈是暗环带; 暗环与外圈暗带(磨损带/中心孔
    槽缘)可能粘连, 但螺丝头永远独立且居孔内近心。win=160 局部窗口内做
    norm<0.72 掩膜 + 开/闭运算 + 补洞, 取"被包围亮区"中离模板中心最近
    的组件质心作为锚点。
    """
    x0, y0 = max(0, int(x) - win), max(0, int(y) - win)
    x1, y1 = min(norm.shape[1], int(x) + win), min(norm.shape[0], int(y) + win)
    m = (norm[y0:y1, x0:x1] < thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    ff = pad.copy()
    ffmask = np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, ffmask, (0, 0), 255)  # 外圈pad1px, 种子必在背景
    added = cv2.bitwise_and(pad | cv2.bitwise_not(ff), cv2.bitwise_not(pad))[1:-1, 1:-1]
    nlab, lab, stats, cent = cv2.connectedComponentsWithStats(added, 8)
    best = None
    for k in range(1, nlab):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < 200:
            continue
        dd = np.hypot(cent[k][0] - (x - x0), cent[k][1] - (y - y0))
        if best is None or dd < best[0]:
            best = (dd, k, area)
    if best is None:
        # 回退: 暗斑(包含种子点的组件, 填充后)质心; 种子不在暗区则原样返回
        k = lab[int(y) - y0, int(x) - x0]
        if k == 0 or stats[k, cv2.CC_STAT_AREA] < 2500:
            return x, y, None
        comp = (lab == k).astype(np.uint8) * 255
        pad2 = cv2.copyMakeBorder(comp, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
        ff2 = pad2.copy()
        fmask = np.zeros((pad2.shape[0] + 2, pad2.shape[1] + 2), np.uint8)
        cv2.floodFill(ff2, fmask, (0, 0), 255)
        filled = (pad2 | cv2.bitwise_not(ff2))[1:-1, 1:-1]
        M = cv2.moments(filled)
        if M["m00"] == 0:
            return x, y, None
        return x0 + M["m10"] / M["m00"], y0 + M["m01"] / M["m00"], None
    bx, by = cent[best[1]]
    return x0 + bx, y0 + by, best[2]


def _head_radius(norm, hx, hy, n_dir=12, thr=0.72):
    """12方向首暗点距离的中位数 = 螺丝头半径。"""
    starts = []
    for k in range(n_dir):
        ang = 2 * np.pi * k / n_dir
        d, g = ray_profile(norm, hx, hy, ang, 100)
        jv = np.where(g < thr)[0]
        if len(jv):
            starts.append(d[jv[0]])
    return float(np.median(starts)) if starts else 0.0


def rim_cross_points(norm, x, y, head_r, n_ray=N_RAY, thr=0.72, max_r=150, min_w=12):
    """每射线: 头外侧第一个宽度>=min_w 的连续暗游程末端的亚像素 0.72 交叉点。

    沉孔口是 ~20px 宽的缓坡(沉孔锥面+斜视), 无硬边缘, 用固定阈值交叉点
    做可复现定义。暗游程延续到窗口边缘 => 与暗带粘连, 弃。
    跳过短游程(螺丝槽纹/头影阴弧 ~5-15px), 防止提前交叉。
    """
    head_r = int(np.clip(head_r, 0, 60))
    step = 0.5
    pts, angs = [], []
    for k in range(n_ray):
        ang = 2 * np.pi * k / n_ray
        d, g = ray_profile(norm, x, y, ang, max_r)
        dark = g < thr
        i = head_r + 6
        while i < len(dark):
            while i < len(dark) and not dark[i]:
                i += 1
            j = i
            while j < len(dark) and dark[j]:
                j += 1
            if j - i >= min_w / step:      # 宽游程 = 沉孔口
                if j >= len(dark) - 4:
                    break                  # 延续到窗口边缘 => 粘连, 弃
                if j + 1 < len(g) and g[j + 1] > g[j]:
                    r = d[j] + (thr - g[j]) * (d[j + 1] - d[j]) / (g[j + 1] - g[j])
                else:
                    r = d[j]
                pts.append((x + np.cos(ang) * r, y + np.sin(ang) * r))
                angs.append(np.degrees(ang))
                break                      # 取第一个宽游程(沉孔口), 其后暗带不计
            i = j + 1
    return (np.array(pts).reshape(-1, 2), np.array(angs)) if pts else (np.zeros((0, 2)), np.zeros(0))


def polar_resid(ell, pts):
    """点到椭圆(极角方向)的径向残差 px。"""
    pts = np.asarray(pts, float)
    (cx, cy), (a1, a2), ang = ell
    ca, sa = np.cos(np.deg2rad(-ang)), np.sin(np.deg2rad(-ang))
    th = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    xr = np.cos(th) * ca + np.sin(th) * sa
    yr = -np.cos(th) * sa + np.sin(th) * ca
    # 解 ((rt*xr)/(a1/2))² + ((rt*yr)/(a2/2))² = 1 的闭式解
    rt = 1.0 / np.sqrt((xr / (a1 / 2)) ** 2 + (yr / (a2 / 2)) ** 2)
    return np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - rt


def fit_clean_arc(pts, angs, cx, cy, min_pts=15):
    """沉孔口鲁棒椭圆拟合: 渐进收紧的极化残差迭代。

    阶段1 用宽松容差(2.5×中位, 12px下限)甩开与暗带/槽缘粘连的远点,
    阶段2 收紧到(1.0×中位, 4px下限)只留干净弧段的点。
    结果须通过形状约束(轴比 1.0-2.2, 尺寸 30-320px)防圆弧塌陷;
    塌陷则退回宽松版本(3×中位, 9px下限)。
    """
    pts = np.asarray(pts, float)
    if len(pts) < min_pts:
        return None, 0
    ell = cv2.fitEllipse(pts.astype(np.float32))

    def _shape_ok(e):
        (_, _), (a1, a2), _ = e
        return 30 <= min(a1, a2) and max(a1, a2) <= 320 \
            and 1.0 <= max(a1, a2) / min(a1, a2) <= 2.2

    for rate, floor in [(2.5, 12.0), (1.6, 7.0), (1.1, 5.0), (1.0, 4.0)]:
        res = polar_resid(ell, pts)
        keep = np.abs(res) <= max(np.median(np.abs(res)) * rate, floor)
        pts = pts[keep]
        if len(pts) < min_pts:
            return None, len(pts)
        ell = cv2.fitEllipse(pts.astype(np.float32))
    if _shape_ok(ell):
        return ell, len(pts)
    # 塌陷回退: 宽松版再走一遍
    ell = cv2.fitEllipse(pts.astype(np.float32))
    for _ in range(3):
        res = polar_resid(ell, pts)
        keep = np.abs(res) <= max(np.median(np.abs(res)) * 3, 9)
        pts = pts[keep]
        if len(pts) < min_pts:
            return None, len(pts)
        ell = cv2.fitEllipse(pts.astype(np.float32))
    if _shape_ok(ell):
        return ell, len(pts)
    return None, len(pts)


def measure_bolt(norm, x, y):
    """单螺栓孔: 头锚定 -> 阈值交叉点 -> 净弧拟合(+二遍校正)。返回 (ellipse, n, 锚点)。"""
    hx, hy, _ = hole_head_center(norm, x, y)
    hr = _head_radius(norm, hx, hy)
    pts, angs = rim_cross_points(norm, hx, hy, hr)
    ell, n = fit_clean_arc(pts, angs, hx, hy) if len(pts) >= 15 else (None, 0)
    if ell is None:
        return None, 0, (hx, hy)
    (ex, ey), _, _ = ell
    pts2, angs2 = rim_cross_points(norm, ex, ey, hr)
    if len(pts2) >= 15:
        ell2, n2 = fit_clean_arc(pts2, angs2, ex, ey)
        if ell2 is not None:
            ell, n = ell2, n2
    return ell, n, (hx, hy)


def _edge_map(gray):
    """Otsu 定阈值 -> Canny 边缘图（配方沿用诊断脚本 profile_ring.py）。"""
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.Canny(blur, int(thr * 0.3), int(thr * 0.9))


def _coarse_centers(gray, edges):
    """候选圆心：图像中心(保底) + 边缘密度质心 + Otsu 最大轮廓外接圆心。

    实测（型号2 偏置图，盘心偏出画面中心 604px）：后两个各自都能独立得到正确
    结果，且三者互相一致——所以不需要哪一个特别准，多给几个让打分去挑。
    """
    h, w = gray.shape
    out = [(w / 2.0, h / 2.0)]
    ys, xs = np.nonzero(edges[::16, ::16])
    if len(xs) >= 10:
        out.append((float(xs.mean() * 16), float(ys.mean() * 16)))
    try:
        _, bw = cv2.threshold(cv2.GaussianBlur(gray, (9, 9), 0), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            (ccx, ccy), _ = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
            out.append((float(ccx), float(ccy)))
    except cv2.error:
        pass
    return out


def _radial_peaks(ey, ex, cx, cy, r_min, r_max, top=6, bin_px=2.0, merge_ratio=0.08):
    """边缘点到 (cx,cy) 的距离直方图取峰 = 候选外圈半径。返回 [(峰点数, 半径)]。

    真实圆在"边缘点距离"直方图上形成尖峰，噪声弧则弥散（续 profile_ring.py 的思路）。
    实测型号2：盘缘峰宽仅 6px。

    **邻近峰要合并**（merge_ratio=8%）：圆心估不准时真半径会被抹成一片，
    1143/1149/1157 这种邻居其实都指向同一个答案，全试一遍纯属浪费。
    ey/ex 是 np.nonzero(edges) 的结果，调用方算一次传进来。
    """
    d = np.hypot(ex - cx, ey - cy)
    d = d[(d >= r_min) & (d <= r_max)]
    if len(d) < 100:
        return []
    bins = np.arange(r_min, r_max + bin_px, bin_px)
    hist, _ = np.histogram(d, bins=bins)
    centers = bins[:-1] + bin_px / 2
    raw = [(int(hist[i]), float(centers[i])) for i in range(1, len(hist) - 1)
           if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]]
    raw.sort(key=lambda t: -t[0])
    out = []
    for cnt, r in raw:
        if all(abs(r - r2) > r2 * merge_ratio for _, r2 in out):
            out.append((cnt, r))
        if len(out) >= top:
            break
    return out


def _outer_candidates(gray, w, h, cx0, cy0, init_r, peaks_out=None):
    """外圈候选 (cx, cy, r0, polarity)，**旧先验排最前以保证基线读数不变**。

    `peaks_out` 非 None 时，把自适应阶段的径向峰 `(count, radius)` 填进去——
    `find_outer` 用它检查"外面是否还有更大的同心圆"（缺陷 8）。快速路径命中时
    生成器根本走不到这里，所以旧先验那张图不会有任何额外开销。

    历史先验只假设了"零件在画面正中 + 外圈半径 ≈ 短边的 44% + 边缘是亮→暗"，
    零件一挪位置/换型号就全废。数据驱动的候选由
    「候选圆心 × 径向峰半径 × 两种极性」组成。

    极性必须一起试：型号1 是暗金属在白背景上（下降沿，72/72），型号2 是灰盘放在
    白布上、盘外更亮（上升沿 55/72，下降沿只有 22/72）——写死下降沿直接废掉。

    这是个生成器：旧先验那三个候选排在前面，**边缘图与直方图是等它被继续迭代时
    才算的**。所以旧先验一旦过了门槛就 break，自适应那套开销（实测约 400ms）
    根本不会发生——只有快速路径失败时才付这个代价。
    """
    short = min(w, h)
    for r0 in [init_r or short * 0.44, short * 0.40, short * 0.48]:
        yield cx0, cy0, r0, -1
    try:
        edges = _edge_map(gray)
    except cv2.error:
        return
    ey, ex = np.nonzero(edges)
    if len(ex) < 100:
        return
    per_center = []
    for (ccx, ccy) in _coarse_centers(gray, edges):
        peaks = _radial_peaks(ey, ex, ccx, ccy, short * 0.10, short * 0.55, top=6)
        if peaks:
            per_center.append(((ccx, ccy), peaks))
    if not per_center:
        return
    # 峰最弱的圆心直接丢：零件不在那儿，试也是白试。实测型号2 图像中心的最强峰
    # 只有最佳圆心的 41%，丢得掉；零件真居中时它本来就是最强的，留得下。
    strongest = max(p[1][0][0] for p in per_center)
    n = 0
    for (ccx, ccy), peaks in per_center:
        if peaks[0][0] < strongest * OUTER_CENTER_MIN_RATIO:
            continue
        if peaks_out is not None:
            peaks_out.extend(peaks)
        for _, r in peaks:
            for pol in (-1, +1):
                n += 1
                if n > OUTER_MAX_CANDIDATES:
                    return
                yield ccx, ccy, float(r), pol


def _fit_outer_at(sm, cx, cy, r0, pol):
    """单个粗候选：射线找边 + 鲁棒圆滤 + 椭圆拟合。"""
    pts = find_edge_points(sm, cx, cy, r0 * 0.88, r0 * 1.16, pol, 2.0, n_ray=N_RAY,
                           mode="outer_first")
    return filter_and_fit(pts, cx, cy, r0)


def _refine_outer(sm, ell, n, pol):
    """粗结果精化：以粗椭圆中心/半径做紧窗口再采一轮（仍取最外侧边缘）。"""
    (cx1, cy1), (aa1, aa2), _ = ell
    req = (aa1 + aa2) / 4
    pts = find_edge_points(sm, cx1, cy1, req * 0.93, req * 1.07, pol, 2.0, n_ray=N_RAY,
                           mode="outer_first")
    ell2, n2 = filter_and_fit(pts, cx1, cy1, req * 1.0)
    return (ell2, n2) if ell2 is not None else (ell, n)


def _outer_shape_ok(ell, n):
    """外圈候选的形状判据：点数够 + 轴比合理（**不看画幅**，出画另判）。

    **残差 RMS 不能当门槛**：实测基线自身就有 41px、错误极性在标定图上也有 41px
    （斜视外缘本身是个有厚度的带，不是一条线），两者区分不开。点数才管用：
    型号2 那次误锁用点只有 22/72。
    """
    if n < OUTER_COVER_MIN * N_RAY:
        return False
    _, (a1, a2), _ = ell
    return min(a1, a2) > 0 and max(a1, a2) / min(a1, a2) <= OUTER_AXIS_RATIO_MAX


def _outer_in_frame(ell, w, h):
    """外接框是否完整落在画幅内。"""
    (cx, cy), (a1, a2), _ = ell
    s = max(a1, a2) / 2          # 保守：按长半轴当各向外接半径
    return (cx - s >= 0) and (cx + s <= w) and (cy - s >= 0) and (cy + s <= h)


def _outer_ok(ell, n, w, h):
    """外圈候选是否可信：形状合格 + 不出画幅。"""
    return _outer_shape_ok(ell, n) and _outer_in_frame(ell, w, h)


def _inner_candidates(r_outer, id_ratio=None):
    """中心孔候选 (lo, hi, r0, polarity)，**旧档位排第一保基线**。

    老代码写死"中心孔半径 = 外圈的 0.60~0.78 倍"——换个孔径比不同的型号就测不到
    （而中心孔一挂，`if hole is not None` 还会连带把整段螺栓孔也跳过）。
    现在按不同比例带各试一轮，由 `_inner_ok` 判可信。两种极性也一起试：孔内是暗是亮
    取决于能不能看穿到台面/背景，不该写死。

    r0 单独给、不取波段中点：老代码那一档用的是 0.70（而不是 0.60~0.78 的中点 0.69），
    圆等价滤点边界跟着 r0 走，差这一点读数就会变——基线回归门就是靠这个抓出来的。

    档案给了真值孔径比（id_mm/od_mm）时，**先**给一组以它为中心的窄带候选（±1.5%，
    中心再 ±2% 扫一遍），每条射线在带内取**最陡梯度**。型号1 内圈实测有三圈
    （≈55.9 / ~60 / ~78mm），窄带保证每次只框住最内侧那一圈，不会把它们平均成一条。
    **不要改成"取缓坡最内侧像素"**：孔壁边缘是约 0.02r 宽的缓坡，取坡脚会随窗口位置漂
    （实测 0.620~0.629），最陡梯度才稳定在 0.630。
    调用方再在"实测比落在真值容差内"的候选里挑最接近真值的一个。原来那组候选
    原样保留作兜底——没有真值孔径比的型号走它，行为与以前一致。
    """
    if id_ratio:
        for dr in (-0.02, -0.01, 0.0, 0.01, 0.02):
            r0 = id_ratio + dr
            for pol in (+1, -1):
                yield (r_outer * (r0 - 0.015), r_outer * (r0 + 0.015),
                       r_outer * r0, pol)
    # ---- 兜底：老档位与自适应波段（保持原样，r0 必须严格 0.70）----
    yield r_outer * 0.60, r_outer * 0.78, r_outer * 0.70, +1
    for f_lo, f_hi in [(0.20, 0.36), (0.33, 0.50), (0.46, 0.64), (0.74, 0.92)]:
        for pol in (+1, -1):
            yield (r_outer * f_lo, r_outer * f_hi,
                   r_outer * (f_lo + f_hi) / 2, pol)
    yield r_outer * 0.60, r_outer * 0.78, r_outer * 0.70, -1


def _inner_ok(ell, n, outer_ell):
    """中心孔候选是否可信：点数够 + 轴比合理 + 确实落在外圈内 + 与外圈同心。

    不做"不出画幅"检查——中心孔本来就在画面中间，那个判据没有区分力。
    同心这条是有效的物理约束：中心孔与外圈是同一个零件上的同轴特征，
    实测标定图两者中心只差 23.9px（外圈半径 1356px 的 1.8%）。
    """
    if n < OUTER_COVER_MIN * N_RAY:
        return False
    (icx, icy), (ia1, ia2), _ = ell
    if min(ia1, ia2) <= 0 or max(ia1, ia2) / min(ia1, ia2) > OUTER_AXIS_RATIO_MAX:
        return False
    (ocx, ocy), (oa1, oa2), _ = outer_ell
    r_outer = (oa1 + oa2) / 4
    if (ia1 + ia2) / 4 >= r_outer * 0.95:                       # 必须确实在外圈之内
        return False
    if np.hypot(icx - ocx, icy - ocy) > r_outer * INNER_CENTER_MAX_SHIFT:
        return False
    return True


def _measure_inner_rings(prof, sm, outer):
    """按档案的 `inner_rings` 量内圈：每圈的两条边各拟一次，取中点。

    为什么取中点：型号1 的 ~60/~78 两圈实测都是**环形槽**，各有一条内壁（亮→暗）和
    一条外壁（暗→亮），两条边差 1~2mm。卡尺真值落在两条边中间；2026-10-22 实测中点
    对 60/78 的偏差 +0.25%/+0.24%（见 docs/自适应定位改造计划.md 的「内圈」一节）。

    只做加性测量：找不到两条边就标「失败」，绝不拿一条边硬充中点。
    """
    (ocx, ocy), (oa1, oa2), _ = outer
    r_outer = (oa1 + oa2) / 4
    out = []
    for item in (prof.get("inner_rings") or []):
        name = item.get("name") or "内圈"
        mm = item.get("mm")
        if not mm or not prof.get("od_mm"):
            continue
        exp = float(mm) / float(prof["od_mm"])
        edges = []
        for pol in (-1, +1):
            pts = find_edge_points(sm, ocx, ocy,
                                   (exp - INNER_RING_HALF) * r_outer,
                                   (exp + INNER_RING_HALF) * r_outer,
                                   pol, 2.0, n_ray=N_RAY)
            ell, n = filter_and_fit(pts, ocx, ocy, exp * r_outer)
            if ell is not None and _inner_ok(ell, n, outer):
                edges.append((pol, ell, n))
        fail = {"name": name, "mm_exp": float(mm), "status": "失败",
                "dia": 0.0, "cx": 0.0, "cy": 0.0, "n": 0, "ell": None}
        if len(edges) != 2 or edges[0][0] == edges[1][0]:
            out.append(fail)
            continue
        (_, e1, n1), (_, e2, n2) = edges
        r1 = (e1[1][0] + e1[1][1]) / 4
        r2 = (e2[1][0] + e2[1][1]) / 4
        if abs(r1 - r2) < INNER_RING_MIN_GAP * r_outer:
            out.append(fail)        # 两种极性落在同一条边上，不是槽
            continue
        c1, c2 = e1[0], e2[0]
        cx, cy = (c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2
        a1 = (e1[1][0] + e2[1][0]) / 2
        a2 = (e1[1][1] + e2[1][1]) / 2
        out.append({"name": name, "mm_exp": float(mm), "status": "OK",
                    "dia": r1 + r2, "cx": cx, "cy": cy, "n": n1 + n2,
                    "ell": ((cx, cy), (a1, a2), e1[2])})
    return out


# ---------------- 型号档案驱动的螺栓孔（装好紧固件的暗芯圆孔） ----------------

def core_cross_points(gray, x, y, r_lo, r_hi, n_ray=48, frac=0.5):
    """从暗芯向外：灰度升到 (芯内 + frac×(外侧-芯内)) 的亚像素交叉点。

    型号2 的"孔"是装好的紧固件：中间一个暗芯 + 外面一圈亮金属环，跟型号1 的
    "暗沉孔 + 亮螺丝头"正好反着。实测(2026-10-07)这个**中分界**定义对得上标称
    1.78mm（36.4px vs 标称 36.8px，差 1.1%），而用 find_edge_points 的"最陡上升沿"
    会量成 50.7px（偏 38%）——所以这里刻意不复用最陡梯度那套。
    """
    pts = []
    for k in range(n_ray):
        ang = 2 * np.pi * k / n_ray
        d, gv = ray_profile(gray, x, y, ang, r_hi)
        m = (d >= r_lo) & (d <= r_hi)
        dd, gg = d[m], gv[m]
        if len(gg) < 8:
            continue
        lo_v = float(gg[:max(1, len(gg) // 4)].min())      # 内侧 = 暗芯
        hi_v = float(np.median(gg[len(gg) // 2:]))         # 外侧 = 亮环
        if hi_v - lo_v < 8:                                # 对比度不够就别硬给点
            continue
        thr = lo_v + frac * (hi_v - lo_v)
        idx = np.where(gg > thr)[0]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        if i == 0:
            r = dd[0]
        else:
            dr = gg[i] - gg[i - 1]
            r = dd[i - 1] + (thr - gg[i - 1]) * (dd[i] - dd[i - 1]) / dr if dr else dd[i]
        pts.append((x + np.cos(ang) * r, y + np.sin(ang) * r))
    return np.array(pts).reshape(-1, 2) if pts else np.zeros((0, 2))


def core_seed(sm, x, y, r_exp):
    """把种子点挪到暗芯的亮度重心上。

    相位搜索给的位置有十几 px 误差，而暗芯半径才 ~20px——偏这么多会让射线上的
    中分界系统性外扩（实测某一个孔因此从 40px 涨到 47.7px，是 4 孔里唯一的离群）。
    """
    w = int(max(6.0, r_exp * 1.2))
    x0, y0 = max(0, int(x) - w), max(0, int(y) - w)
    x1, y1 = min(sm.shape[1], int(x) + w), min(sm.shape[0], int(y) + w)
    loc = sm[y0:y1, x0:x1].astype(float)
    if loc.size < 4:
        return float(x), float(y)
    thr = loc.min() + 0.5 * (float(np.median(loc)) - loc.min())
    wgt = np.clip(thr - loc, 0.0, None)
    if wgt.sum() <= 0:
        return float(x), float(y)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return float((xs * wgt).sum() / wgt.sum()), float((ys * wgt).sum() / wgt.sum())


def measure_core_hole(sm, x, y, r_exp):
    """量暗芯圆孔：先把种子挪到暗芯重心 -> 射线找中分界 -> 中位半径当直径。
    返回 ((cx,cy), dia_px, 方向数)。

    这里**刻意不拟合椭圆**：实测暗芯有一侧常连到阴影，一拟合就被拉成 1.3 左右的轴比、
    均值因此偏高 7~11%（4 孔 1.66~2.28mm）；中位半径对少数方向上的粘连不敏感，
    4 孔均值对上标称 1.78mm 只差 1% 量级。
    代价是丢掉椭圆方位——型号2 这种接近正拍的机位不需要它。
    """
    x, y = core_seed(sm, x, y, r_exp)
    pts = core_cross_points(sm, x, y, r_exp * 0.4, r_exp * 2.2)
    if len(pts) < 12:
        return None, 0.0, len(pts)
    d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
    med = float(np.median(d))
    keep = np.abs(d - med) <= max(med * 0.25, 3.0)
    if keep.sum() >= 12:
        pts, d = pts[keep], d[keep]
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    return (cx, cy), 2.0 * float(np.median(d)), int(len(d))


# ======================================================================
# 螺栓孔：两个正交维度的策略表
#   locate   —— 怎么找到各孔位置
#   boundary —— 量哪条边界（决定了"读数"是什么含义）
# 型号档案里各写一个名字。**所有型号平级**：没有哪个型号被写死在代码默认里，
# 缺档案时才落到 DEFAULT_* 兜底，且结果一律标"未经验证"。
# 新增一个零件形态 = 加一个策略函数（约 15 行），不用碰 detect()。
# ======================================================================

DEFAULT_LOCATE = "template_ngon"
DEFAULT_BOUNDARY = "countersink_rim"
NGON_CANDIDATES = (3, 4, 5, 6, 8, 10, 12)   # 档案没给孔数时推断等分数的候选


def ctx_sm2(ctx):
    """按需算 σ=2 的模糊图（暗芯只有几十 px，用 detect 里 σ=5 那张会糊过头）。"""
    if ctx.get("sm2") is None:
        ctx["sm2"] = cv2.GaussianBlur(ctx["gray"], (0, 0), 2)
    return ctx["sm2"]


def locate_by_profile_angles(prof, ctx):
    """按档案给的**相对角度**定位：先扫整体相位，再给出各孔位置。

    绝对相位由图像数据自己找（扫一圈让模板角度尽量落在暗处），零件转个角度也不怕。
    支持非等分布局（型号2 是 2x2 矩形：相邻间隔 72.8/107.4/72.0/107.8°，不是 90°）。
    需要 bolt_mm + mm_per_px 才算得出边界的期望半径；缺了就没法量，如实返回空。
    """
    rel, mmpx, bolt_mm = (prof.get("bolt_angles_rel"), prof.get("mm_per_px"),
                          prof.get("bolt_mm"))
    if not rel or not mmpx or not bolt_mm:
        return []
    rel = np.asarray(rel, float)
    r_norm = float(prof.get("bolt_circle_ratio", 0.8))
    (ocx, ocy), (oa1, oa2), oang = ctx["outer"]
    norm, h, w = ctx["norm"], ctx["h"], ctx["w"]
    best, best_ph = None, 0.0
    for ph in np.arange(0.0, 360.0, 1.0):
        xs, ys = from_norm(np.cos(np.deg2rad(ph + rel)) * r_norm,
                           np.sin(np.deg2rad(ph + rel)) * r_norm, ocx, ocy, oa1, oa2, oang)
        vals = [norm[int(round(y)), int(round(x))] for x, y in zip(xs, ys)
                if 0 <= int(round(y)) < h and 0 <= int(round(x)) < w]
        if len(vals) < len(rel):
            continue
        score = -float(np.mean(vals))          # 孔是暗的：越暗越像
        if best is None or score > best:
            best, best_ph = score, float(ph)
    r_exp = (float(bolt_mm) / 2) / float(mmpx)
    seeds = []
    for a in rel:
        ang = (best_ph + a) % 360
        x, y = from_norm(np.cos(np.deg2rad(ang)) * r_norm,
                         np.sin(np.deg2rad(ang)) * r_norm, ocx, ocy, oa1, oa2, oang)
        seeds.append({"x": float(x), "y": float(y), "r_exp": r_exp,
                      "r_norm": r_norm, "src": "profile"})
    return seeds


def infer_ngon_n(cands, ctx, ns=NGON_CANDIDATES):
    """档案没给孔数时从候选峰推断等分数：取"真命中最多"的 n。

    评分只看**真命中**、不看槽位总数——推定槽位是白送的（任何 n 都能凑满），
    拿它评分会一味偏向 n 大的。这正是原来写死 6 会让 12 孔零件静默只报 6 个的反面。
    """
    (ocx, ocy), (oa1, oa2), oang = ctx["outer"]
    best_n, best_key = 6, None
    for n in ns:
        slots = ngon_slots(cands, ocx, ocy, oa1, oa2, oang, n=n)
        if not slots:
            continue
        filled = sum(1 for _, _, u in slots if u is not None)
        key = (filled, -n)              # 真命中优先；同命中取孔数少的（更保守）
        if best_key is None or key > best_key:
            best_n, best_key = n, key
    return best_n


def locate_by_template_ngon(prof, ctx):
    """数据驱动定位：暗斑模板匹配取峰 -> n 等分假设校验 -> 各槽位位置。

    n 优先用档案的 bolt_count。**中心孔没检出也照样跑**（不再整段跳过）——
    有中心孔就把它从环带里挖掉，没有就用整个盘面；以前那种"中心孔一挂、螺栓孔
    根本不试、却报'没检测到螺栓孔'"的误导就是从这里来的。
    """
    sm, norm = ctx["sm"], ctx["norm"]
    (ocx, ocy), (oa1, oa2), oang = ctx["outer"]
    h, w = ctx["h"], ctx["w"]
    r_outer = (oa1 + oa2) / 4
    band = np.zeros((h, w), np.uint8)
    cv2.ellipse(band, (int(ocx), int(ocy)),
                (int(oa1 / 2 * 0.98), int(oa2 / 2 * 0.98)), oang, 0, 360, 255, -1)
    hole = ctx["hole"][0]
    if hole is not None:
        (hcx, hcy), (ha1, ha2), hang = hole
        cv2.ellipse(band, (int(hcx), int(hcy)),
                    (int(ha1 / 2 * 1.05), int(ha2 / 2 * 1.05)), hang, 0, 360, 0, -1)
    tmpl = rough_templates(sm, band)
    if tmpl is None:
        return []
    cands = []
    for x, y, v in template_locate(sm, tmpl):
        dd = np.hypot(x - ocx, y - ocy)
        if not (r_outer * 0.5 < dd < r_outer * 1.05):
            continue
        if len(find_edge_points(sm, x, y, 40, 140, -1, 0.8, n_ray=48)) < 12:
            continue
        cands.append((x, y, v))
    if not cands:
        return []
    n = int(prof["bolt_count"]) if prof.get("bolt_count") else infer_ngon_n(cands, ctx)
    seeds = []
    for exp_ang, r_norm, use in (ngon_slots(cands, ocx, ocy, oa1, oa2, oang, n=n) or []):
        if use is not None:
            x, y, src = use[0], use[1], "detected"
        else:
            # 槽位没匹配到峰 -> 位置是**推定**的，必须标出来，不能让它看起来像实测
            x, y = from_norm(np.cos(np.deg2rad(exp_ang)) * r_norm,
                             np.sin(np.deg2rad(exp_ang)) * r_norm, ocx, ocy, oa1, oa2, oang)
            src = "synthesized"
        seeds.append({"x": float(x), "y": float(y), "r_exp": None,
                      "r_norm": float(r_norm), "src": src})
    return seeds


def boundary_countersink_rim(ctx, x, y, r_exp):
    """沉孔口边界：螺丝头锚定 -> 宽暗游程末端 0.72 阈值交叉 -> 净弧椭圆拟合。

    "暗沉孔 + 亮螺丝头"那类（型号1）的定义；不需要标称尺寸（头半径从图里量）。
    """
    ell, n, _ = measure_bolt(ctx["norm"], x, y)
    if ell is None:
        return None, n
    (ex, ey), (a1, a2), ang = ell
    return (ex, ey), (a1, a2), ang, n


def boundary_dark_core(ctx, x, y, r_exp):
    """暗芯边界：种子挪到暗芯重心 -> 亮度中分界 -> 中位半径当直径。

    "装好紧固件"那类（型号2）的定义。**需要 r_exp**（由 bolt_mm/mm_per_px 算出），
    搜索带要围绕期望半径；没有标称尺寸就量不了，如实返回失败而不是硬给一个数。
    """
    if not r_exp:
        return None, 0
    c, dia, n = measure_core_hole(ctx_sm2(ctx), x, y, r_exp)
    if c is None:
        return None, n
    return c, (dia, dia), 0.0, n


# ======================================================================
# 型号识别：外圈环带的**极坐标指纹**
#
# 为什么不是"找孔、数孔、比孔圈比"（那套 `build_signature`/`match_profile` 已删除）：
# 那条路要求暗斑面积 ≥3000px²（等效直径 ≥61.8px，**绝对像素门槛**），型号2 的暗芯
# 只有 ~37px，在第一关就被挡掉，孔数=0；型号1 那边孔数推断又被盘面假峰带偏。
# 实测命中率 0。用绝对像素尺寸做判据这条路不要再走。
#
# 指纹为什么能成立：绝对尺寸不能用来认型号（mm/px 是按型号存的，拿绝对外径查档案
# 是循环依赖），但**归一化后的外圈环带**可以——按归一化半径展开、整体标准化，
# 尺度/亮度/对比度就都不影响匹配；沿角度做循环互相关，旋转不变性也是免费的。
# 全程只依赖一个几何量：外圈椭圆。
FP_N_ANG = 720          # 行 = 图像极角，0.5°/行，首尾相接（循环互相关要求它闭合）
FP_N_RAD = 48           # 列 = 从外缘往内的归一化深度
FP_DEPTH = 0.85         # 展开到"外缘往内 0.85×r_ell(θ)"
FP_ANCHOR_LO = 0.90     # 找外缘的窗口（× r_ell(θ)）
FP_ANCHOR_HI = 1.12
FP_ANCHOR_STEP = 0.0025  # 窗口内采样步长（归一化半径，≈3px）
FP_SIGMA_RAD = 3.0      # 沿半径方向的平滑（列数）
FP_HARMONICS = 4        # 锚点曲线保留的谐波阶数（椭圆是纯 2 次谐波）
FP_ACCEPT = 0.30        # 认型号所需的最低相关系数
FP_GAP = 0.15           # 与次高分的最小差距（多个型号都像就拒绝）
# ⚠️ 上面两个门槛是**暂定值**：目前只有 2 个零件标定（型号1/型号2 各一张参考图），
# 值是从"自匹配 vs 交叉匹配"的实测分布上取的（自匹配最差 0.463、交叉最高 0.078）。
# **拿到第三个零件后必须复核**，别当通用常数。更不许为了让某张图认得出来去调它。


def _ell_radius(th, a1, a2, ang):
    """图像极角 th 方向上外圈椭圆的半径（与 polar_resid 里的闭式解同一套）。"""
    ca, sa = np.cos(np.deg2rad(-ang)), np.sin(np.deg2rad(-ang))
    c, s = np.cos(th), np.sin(th)
    xr = c * ca + s * sa
    yr = -c * sa + s * ca
    return 1.0 / np.sqrt((xr / (a1 / 2)) ** 2 + (yr / (a2 / 2)) ** 2)


def _smooth_anchor(u, th):
    """把逐射线的锚点曲线做低阶傅里叶拟合，返回平滑后的曲线。

    为什么要平滑：外缘是一条**有厚度**的带（零件外缘有倒角/圆角，斜视下还一段亮
    一段暗），逐射线取"最强梯度"会在带的两侧之间来回跳（实测抖动 std≈0.042×r_ell）。
    椭圆的极坐标半径本身是 θ 的**纯二次谐波**函数，所以用谐波 0/2/4 拟合既滤掉抖动、
    又保留真实的（轻微非椭圆）偏差。这一步是旋转不变性能用的关键：不平滑时自匹配在
    47°~317° 会掉到 0.15，与交叉得分重叠、根本分不开。
    """
    cols = [np.ones_like(th)]
    for k in range(1, FP_HARMONICS // 2 + 1):
        cols += [np.cos(2 * k * th), np.sin(2 * k * th)]
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, u, rcond=None)
    return A @ coef


def _outer_edge_anchor(norm, outer):
    """每条射线各自量自己的外缘，返回 (归一化锚点深度, r_ell, th)。

    锚点在**拟合椭圆**周围的窄窗口里找，所以不依赖"零件在哪"的先验；而"每条射线各
    锚各的"正好抵消掉全局的半径/圆心误差——拟合椭圆会随零件转动在带内漂（实测型号1
    转 47° 时外径 2712.9→2780.3px，+2.5%），拿它当径向基准会把指纹整体错开约 9 列，
    自匹配因此掉到 0.15。逐射线锚定把这类误差按角度各自消掉。
    """
    (ocx, ocy), (oa1, oa2), oang = outer
    th = np.arange(FP_N_ANG) * (2 * np.pi / FP_N_ANG)
    r_ell = _ell_radius(th, oa1, oa2, oang)
    rho = np.arange(FP_ANCHOR_LO, FP_ANCHOR_HI, FP_ANCHOR_STEP)
    xs = (ocx + np.cos(th)[:, None] * (r_ell[:, None] * rho[None, :])).astype(np.float32)
    ys = (ocy + np.sin(th)[:, None] * (r_ell[:, None] * rho[None, :])).astype(np.float32)
    prof = cv2.remap(norm.astype(np.float32), xs, ys, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    g = np.gradient(prof, axis=1)
    k = np.arange(FP_N_ANG)
    j = np.clip(np.argmax(np.abs(g), axis=1), 1, g.shape[1] - 2)
    a, b, c = g[k, j - 1], g[k, j], g[k, j + 1]   # 抛物线插值取亚像素极值
    den = a - 2 * b + c
    delta = np.where(den != 0, 0.5 * (a - c) / np.where(den == 0, 1.0, den), 0.0)
    u = FP_ANCHOR_LO + (j + np.clip(delta, -1.0, 1.0)) * FP_ANCHOR_STEP
    return _smooth_anchor(u, th), r_ell, th


def polar_fingerprint(gray, outer):
    """外圈环带 -> (FP_N_ANG, FP_N_RAD) 标准化极坐标指纹。

    灰度先做局部对比度归一化（gray / blur(gray,201)，与螺栓孔那段同一配方），再把整张
    指纹标准化；径向用"从实测外缘往内的归一化深度"，横向是该角度的 r_ell(θ)（倾斜由它
    承担）。所以亮度/对比度/尺度/旋转都不影响匹配——全程无量纲，不需要 mm/px。
    """
    (ocx, ocy), _, _ = outer
    sm = cv2.GaussianBlur(gray, (0, 0), 5)
    norm = sm / np.maximum(cv2.blur(sm, (201, 201)), 1)
    u_anchor, r_ell, th = _outer_edge_anchor(norm, outer)
    u = np.linspace(0.0, FP_DEPTH, FP_N_RAD)
    r = (u_anchor[:, None] - u[None, :]) * r_ell[:, None]
    xs = (ocx + np.cos(th)[:, None] * r).astype(np.float32)
    ys = (ocy + np.sin(th)[:, None] * r).astype(np.float32)
    fp = cv2.remap(norm.astype(np.float32), xs, ys, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    fp = cv2.GaussianBlur(fp, (0, 0), sigmaX=FP_SIGMA_RAD, sigmaY=0.15)
    fp = fp - fp.mean()
    return (fp / (fp.std() + 1e-9)).astype(np.float32)


def match_fingerprint(fp, fingerprints):
    """待测指纹 vs 各型号指纹，返回 (型号名 | None, 说明)。

    沿角度做**循环互相关**：圆环展开天然周期，两图又都已标准化，而循环移位不改变
    均值/标准差，所以互相关的峰值就是皮尔逊相关系数——一次 FFT 就把 720 个移位全算完。
    得分**取符号值、不取绝对值**：亮暗反转的零件不该被认成同一个。

    **只在明确命中时才认**，而且保守到底：
      - 最高分 < FP_ACCEPT     -> 未识别（说明最像的是谁、差多少）；
      - 与次高分差距 < FP_GAP  -> 多个型号都像，拒绝并列出候选。
    理由是"认错型号"比"没认出来"危险得多——前者会拿错型号的标称值和 mm/px，给一个
    理直气壮的错读数。
    """
    if fp is None:
        return None, "没能算出待测图的指纹"
    if not fingerprints:
        return None, "没有任何型号的指纹可比对（先用 --save-fingerprint 建档）"
    f1 = fp - fp.mean()
    f1 = f1 / (f1.std() + 1e-9)
    f1f = np.fft.fft(f1, axis=0)
    scores = {}
    for name, ref in fingerprints.items():
        if ref is None or np.shape(ref) != np.shape(fp):
            continue
        f2 = ref - ref.mean()
        f2 = f2 / (f2.std() + 1e-9)
        c = np.fft.ifft(f1f * np.conj(np.fft.fft(f2, axis=0)), axis=0).real
        scores[name] = float(c.sum(axis=1).max() / f1.size)
    if not scores:
        return None, "没有可用的指纹（尺寸对不上，可能是用旧参数生成的）"
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, best_s = ranked[0]
    if best_s < FP_ACCEPT:
        return None, (f"最高相关只有 {best_s:.3f}（最像的是「{best}」），"
                      f"低于门槛 {FP_ACCEPT:.2f}")
    if len(ranked) > 1:
        second, second_s = ranked[1]
        if best_s - second_s < FP_GAP:
            return None, (f"「{best}」({best_s:.3f}) 与「{second}」({second_s:.3f}) "
                          f"太接近（差距 {best_s - second_s:.3f} < {FP_GAP:.2f}）")
        return best, f"极坐标指纹匹配 相关 {best_s:.3f}（次高「{second}」{second_s:.3f}）"
    return best, f"极坐标指纹匹配 相关 {best_s:.3f}（档案里只有这一个指纹）"


LOCATE_FNS = {
    "profile_angles": locate_by_profile_angles,
    "template_ngon": locate_by_template_ngon,
}

BOUNDARY_FNS = {
    "countersink_rim": boundary_countersink_rim,
    "dark_core": boundary_dark_core,
}


def _larger_concentric_out_of_frame(sm, ell, peaks, w, h):
    """检出圆之外是否还有"更大、同心、形状合格、但出画"的圆。

    用来拦住"外圈被画幅切掉、于是锁到内孔"的静默错读（缺陷 8）：内孔本身是个完美圆，
    `_outer_ok` 的三条判据全过，只有"外面还有一圈更大的同心圆"能揭穿它。
    只看明显更大的峰（> 检出半径 ÷ OUTER_OUT_OF_FRAME_RATIO），通常只有一两个。
    """
    (cx, cy), (a1, a2), _ = ell
    r_acc = (a1 + a2) / 4
    for _, r in peaks:
        if r <= r_acc / OUTER_OUT_OF_FRAME_RATIO:
            continue
        for pol in (-1, +1):
            e, n = _fit_outer_at(sm, cx, cy, r, pol)
            if e is None:
                continue
            e, n = _refine_outer(sm, e, n, pol)
            if not _outer_shape_ok(e, n):
                continue
            (ecx, ecy), _, _ = e
            if np.hypot(ecx - cx, ecy - cy) > OUTER_CONCENTRIC_TOL * r_acc:
                continue
            if not _outer_in_frame(e, w, h):
                return True
    return False


def find_outer(gray, init_c=None, init_r=None):
    """外圈自适应定位，返回 ((椭圆, 保留点数) | None, σ=5 模糊图, 失败原因)。

    失败原因：None=成功；"out_of_frame"=外面有更大的同心圆被画幅切掉（检出的只是内孔）；
    "not_found"=没找到可信外圈。调用方据此给具体提示，不再笼统报"外圈拟合失败"。

    单独拿出来是因为**型号识别也要外圈椭圆**（极坐标指纹的径向基准就是它），而识别
    跑在检测之前——若复用 detect()，会把整条螺栓孔管线白跑一遍。
    """
    h, w = gray.shape
    sm = cv2.GaussianBlur(gray, (0, 0), 5)
    # 旧先验优先(保基线)，不行再自适应定位
    cx0, cy0 = init_c or (w / 2, h / 2)
    peaks = []
    for (cx, cy, r0, pol) in _outer_candidates(gray, w, h, cx0, cy0, init_r, peaks):
        ell, n = _fit_outer_at(sm, cx, cy, r0, pol)
        if ell is None:
            continue
        ell, n = _refine_outer(sm, ell, n, pol)
        if not _outer_ok(ell, n, w, h):
            continue
        # 快速路径（旧先验）命中时 peaks 还是空的，这一步是空转、不付开销。
        # 缺陷 8：内孔也是完美圆；若外面还有更大、同心的圆被画幅切掉，现在这个是内孔。
        if _larger_concentric_out_of_frame(sm, ell, peaks, w, h):
            return None, sm, "out_of_frame"
        return (ell, n), sm, None
    return None, sm, "not_found"


def detect(img, init_c=None, init_r=None, profile=None):
    """完整检测管线(BGR图 -> 测量结果 dict)。

    profile: 型号档案(dict)。里面的 locate/boundary 决定螺栓孔怎么测；两者缺一
             或整个档案为 None 时落到 DEFAULT_* 兜底，并把 bolts_unverified 置 True
             （缺档案 = 没有标称值可校验、也没有专属 mm/px，结果不可当真）。

    返回:
        outer: (center,(axis1,axis2),angle[,n]) 或 None
        center_hole: 同上
        inner_rings: [{name, mm_exp, status, dia, cx, cy, n, ell}]（档案 inner_rings 声明几条就几条）
        bolts: [{cx,cy,dia,a1,a2,status,n,src}]，src=detected/profile/synthesized
        bolts_locate / bolts_boundary: 实际用到的策略名
        bolts_unverified: 是否走了兜底（缺档案或不完整）
        outer_fail: 外圈失败原因（None | "out_of_frame" | "not_found"），供调用方给具体提示
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ---- 外圈: 旧先验优先(保基线)，不行再自适应定位 ----
    found, sm, outer_fail = find_outer(gray, init_c, init_r)
    if found is None:
        return {"outer": None, "center_hole": None, "bolts": [], "inner_rings": [],
                "bolts_locate": None, "bolts_boundary": None, "bolts_unverified": True,
                "outer_fail": outer_fail}
    outer, n_o = found
    (ocx, ocy), (oa1, oa2), oang = outer

    # ---- 中心孔: 自适应波段（旧档位排第一保基线）+ 真值孔径比约束 ----
    prof = profile or {}
    r_outer = (oa1 + oa2) / 4
    # 档案同时给了 id_mm / od_mm 才用真值；无中心孔的型号不参与
    id_ratio = None
    if (prof.get("id_mm") and prof.get("od_mm")
            and prof.get("has_center_hole") is not False):
        id_ratio = float(prof["id_mm"]) / float(prof["od_mm"])
    hole = n_h = None
    best_inner = None
    for (lo, hi, r0, pol) in _inner_candidates(r_outer, id_ratio):
        # 带内取最陡梯度（不要取缓坡坡脚，那会随窗口位置漂）
        pts_h = find_edge_points(sm, ocx, ocy, lo, hi, pol, 1.2, n_ray=N_RAY)
        ell, n = filter_and_fit(pts_h, ocx, ocy, r0)
        if ell is None or not _inner_ok(ell, n, outer):
            continue
        if id_ratio is None:
            hole, n_h = ell, n
            break
        # 有真值：在"实测孔径比落在容差内"的候选里挑最接近真值的那条
        diff = abs((ell[1][0] + ell[1][1]) / 4 / r_outer - id_ratio)
        if diff <= INNER_RATIO_TOL and (best_inner is None or diff < best_inner[0]):
            best_inner = (diff, ell, n)
    if best_inner is not None:
        hole, n_h = best_inner[1], best_inner[2]
    if hole is None:
        hcx, hcy, ha1, ha2, hang = ocx, ocy, 0.0, 0.0, 0.0
    else:
        (hcx, hcy), (ha1, ha2), hang = hole

    # ---- 内圈（档案声明了几个就测几个；两条边取中点）----
    inner_rings = _measure_inner_rings(prof, sm, outer)

    # ---- 螺栓孔: locate(怎么定位孔) + boundary(量哪条边界) 两个维度派发 ----
    norm = sm / np.maximum(cv2.blur(sm, (201, 201)), 1)
    ctx = {"sm": sm, "sm2": None, "gray": gray, "norm": norm,
           "outer": outer, "hole": (hole, n_h), "w": w, "h": h}
    locate_name = prof.get("locate")
    boundary_name = prof.get("boundary")
    # 档案缺一维（或整个没给）就落到兜底：**结果一律标"未经验证"**——
    # 没有标称值可校验、也没有本型号专属的 mm/px，不能当真。
    unverified = locate_name not in LOCATE_FNS or boundary_name not in BOUNDARY_FNS
    if locate_name not in LOCATE_FNS:
        locate_name = DEFAULT_LOCATE
    if boundary_name not in BOUNDARY_FNS:
        boundary_name = DEFAULT_BOUNDARY
    bfn = BOUNDARY_FNS[boundary_name]
    bolts = []
    seeds = LOCATE_FNS[locate_name](prof, ctx)
    for s in seeds:
        got = bfn(ctx, s["x"], s["y"], s["r_exp"])
        if got is None or got[0] is None:
            bolts.append({"cx": s["x"], "cy": s["y"], "dia": 0.0, "a1": 0.0, "a2": 0.0,
                          "n": got[1] if got else 0, "src": s["src"], "status": "失败"})
            continue
        (ex, ey), (ea1, ea2), eang, n = got
        bolts.append({"cx": ex, "cy": ey, "dia": (ea1 + ea2) / 2,
                      "a1": ea1, "a2": ea2, "ang": eang, "n": n,
                      "src": s["src"], "status": "OK"})
    return {"outer": (outer, n_o), "center_hole": (hole, n_h), "bolts": bolts,
            "inner_rings": inner_rings,
            "bolts_locate": locate_name, "bolts_boundary": boundary_name,
            "bolts_unverified": unverified, "outer_fail": None}


def main(p: Path):
    img = cv2.imread(str(p))
    if img is None:
        raise SystemExit(f"无法读取图片: {p}")
    res = detect(img)
    outer, n_o = res["outer"]
    hole, n_h = res["center_hole"]
    if outer is None:
        raise SystemExit("外圈拟合失败")
    (ocx, ocy), (oa1, oa2), oang = outer
    print(f"外圈: 中心({ocx:.1f},{ocy:.1f}) 轴{oa1:.1f}x{oa2:.1f}px 角度{oang:.1f}° 点数{n_o}")
    if hole:
        (hcx, hcy), (ha1, ha2), hang = hole
        print(f"中心孔: 中心({hcx:.1f},{hcy:.1f}) 轴{ha1:.1f}x{ha2:.1f}px 角度{hang:.1f}° 点数{n_h}")
    else:
        print("中心孔: 拟合失败")
    for i, b in enumerate(res["bolts"], 1):
        if b["status"] == "OK":
            print(f"  螺栓孔{i}: d={b['dia']:.1f}px ({b['a1']:.1f}x{b['a2']:.1f})"
                  f" 中心({b['cx']:.0f},{b['cy']:.0f}) 保留{b['n']}点")
        else:
            print(f"  螺栓孔{i}: @({b['cx']:.0f},{b['cy']:.0f}) 直径测量失败")

    # ---- 调试图 ----
    dbg = img.copy()
    if outer:
        cv2.ellipse(dbg, outer, (0, 255, 0), 8)
    if hole:
        cv2.ellipse(dbg, hole, (0, 0, 255), 8)
    for b in res["bolts"]:
        if b["status"] == "OK":
            cv2.ellipse(dbg, ((b["cx"], b["cy"]), (b["a1"], b["a2"]), 0), (255, 0, 0), 6)
    dbg_dir = Path("debug")
    dbg_dir.mkdir(exist_ok=True)
    out = dbg_dir / f"fit_debug_{Path(p).stem}.jpg"
    cv2.imwrite(str(out), dbg)
    print(f"已保存 {out}")


if __name__ == "__main__":
    p = Path(sys.argv[1] if len(sys.argv) > 1 else "../data/captures/capture_20260908_115557.jpg")
    main(p)
