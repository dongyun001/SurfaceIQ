r"""几何原语：亚像素射线找边、鲁棒椭圆拟合、外圈自适应定位。

**这里只剩几何，不含任何"业务语义"**——不认型号、不判合格、不区分"外径/中心孔/螺栓孔"。
上层（`concentric.py`）用它把一根根同心边列出来，`measure_ring.py` 再加毫米数。

2026-10-23 删掉的东西（别再以"以前有"为由重建）：
    螺栓孔全链路（模板匹配 / 等角共圆校验 / 沉孔口 / 暗芯中分界 / 两维派发 LOCATE_FNS+BOUNDARY_FNS）、
    中心孔真值孔径比约束、内圈两道边取中点、紧固件锚定回退、整体管线 `detect()`、
    极坐标指纹与孔型识别。

用法（自检：打印外圈）:
    python src\fit_ellipse_ring.py <图片路径>
"""
import sys
from pathlib import Path

import cv2
import numpy as np

N_RAY = 72

# 外圈候选的可信门槛（自适应定位用；依据见 docs/设计决策记录.md）
OUTER_COVER_MIN = 0.6        # 保留点数 ≥ 射线数×此比例（误锁只有 22/72，正确为 54~72/72）
OUTER_AXIS_RATIO_MAX = 1.8   # 外圈长短轴比上限（斜视视角的物理上限）
OUTER_MAX_CANDIDATES = 60    # 候选上限，防自适应搜索跑飞
OUTER_CENTER_MIN_RATIO = 0.5  # 圆心最强径向峰 < 全局最强峰×此比例就丢弃（零件不在这儿）
OUTER_OUT_OF_FRAME_RATIO = 0.92  # 检出圆 < 更大同心圆的此比例 => 零件出画、检出的是内孔
OUTER_CONCENTRIC_TOL = 0.15      # 判"同心"的圆心偏移上限（×检出半径）
INNER_CENTER_MAX_SHIFT = 0.15  # 同心内边中心相对外圈中心的容许偏移（×外圈半径）

# ---------------- 成组孔（偏心孔）：暗斑检测 + 50% 平台中点边界 ----------------
# ⚠️ 这些门槛在 2026-11-02 的 6 孔件上标定；换第三种零件必须复核
# （见 docs/偏心孔检测计划.md §4/§6）。阈值**只用于定位候选**，不参与最终直径——
# 直径统一由 fit_hole_boundary 的 50% 平台中点重算，所以换门槛不会让读数漂移。
HOLE_DARK_THRS = (0.55, 0.60, 0.62, 0.65, 0.70)
HOLE_AREA_LO = 0.002        # 候选面积下限 × R²
HOLE_AREA_HI = 0.05         # 候选面积上限 × R²
HOLE_MIN_CIRC = 0.70        # 圆度下限
HOLE_BBOX_MAX = 0.35        # 外接框长边上限 × R（挡整圈暗槽）
HOLE_MERGE_TOL = 0.02       # 跨阈值合并距离 × R
HOLE_MIN_POINTS = 36        # < 0.5×72 => 边界失败
HOLE_LOW_POINTS = 43        # < 0.6×72 => 低置信度
HOLE_MIN_CONTRAST = 15.0    # 暗/亮平台差 < 此值 => 该射线丢弃
HOLE_MIN_GRAD = 1.0         # 上升沿最大梯度门槛（灰阶/样本）

# 亮环（已装紧固件：螺钉头 / 沉孔口）。暗斑法在这类件上会把暗芯与暗影连成一个大斑，
# 2026-11-03 实测：4 孔件在阈值 1.05~1.20 下亮环法都是恰好 4 个，暗斑法只有 3 个。
HOLE_BRIGHT_THRS = (1.05, 1.10, 1.15, 1.20)
HOLE_RING_AREA_LO = 0.001       # 亮环面积下限 × R²
HOLE_RING_AREA_HI = 0.02        # 亮环面积上限 × R²
HOLE_RING_MIN_CIRC = 0.72       # 亮环圆度下限
HOLE_RING_DIST_LO = 0.55        # 亮环到零件中心的距离下限 × R
HOLE_RING_DIST_HI = 0.95        # 上限 × R

# ---------------- 外缘取边线索选择 ----------------
# 2026-11-02 在 2 个零件 / 3 种背景上标定；换第三种零件必须复核。
# 规则：灰度差够 -> 强度边（保基线）；灰度差很弱但纹理能分开 -> 纹理中点；其余 -> 强度边 + 告警。
EDGE_CUE_DELTA_STRONG = 50.0    # ≥ 此值 -> 直接用强度边
EDGE_CUE_TEXTURE_MIN = 5.0      # 纹理能量比 ≥ 此值才允许走纹理
EDGE_CUE_MIN_GAP = 0.02         # 纹理边离强度边 < 2%R => 背景对比可能不足，必须告警换背景
TEXTURE_SIGMA = 15.0            # 纹理能量的高斯平滑尺度（实测 8~40 之间中点变化约 0.5%）


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


def _fit_outer_at(sm, cx, cy, r0, pol, min_grad=2.0):
    """单个粗候选：射线找边 + 鲁棒圆滤 + 椭圆拟合。

    `min_grad` 默认 2.0（历史基线）；低对比度件可以降下来重试，见 `find_outer`。
    """
    pts = find_edge_points(sm, cx, cy, r0 * 0.88, r0 * 1.16, pol, min_grad, n_ray=N_RAY,
                           mode="outer_first")
    return filter_and_fit(pts, cx, cy, r0)


def _refine_outer(sm, ell, n, pol, min_grad=2.0):
    """粗结果精化：以粗椭圆中心/半径做紧窗口再采一轮（仍取最外侧边缘）。"""
    (cx1, cy1), (aa1, aa2), _ = ell
    req = (aa1 + aa2) / 4
    pts = find_edge_points(sm, cx1, cy1, req * 0.93, req * 1.07, pol, min_grad, n_ray=N_RAY,
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


def edge_ok(ell, n, outer_ell):
    """同心边的可信判据：点数够 + 轴比合理 + 确实落在外圈内 + 与外圈同心。

    公开名（原来是 `_inner_ok`，只服务"中心孔"）：现在通用量测拿它筛**每一条**同心边。
    不做"不出画幅"检查——要列的都是画面里的同心边，那个判据没有区分力。
    同心这条是有效的物理约束：同一零件上的同心特征，实测标定图两者中心只差 23.9px
    （外圈半径 1356px 的 1.8%）。
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


def find_outer(gray, init_c=None, init_r=None, min_grad=2.0):
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
        ell, n = _fit_outer_at(sm, cx, cy, r0, pol, min_grad)
        if ell is None:
            continue
        ell, n = _refine_outer(sm, ell, n, pol, min_grad)
        if not _outer_ok(ell, n, w, h):
            continue
        # 快速路径（旧先验）命中时 peaks 还是空的，这一步是空转、不付开销。
        # 缺陷 8：内孔也是完美圆；若外面还有更大、同心的圆被画幅切掉，现在这个是内孔。
        if _larger_concentric_out_of_frame(sm, ell, peaks, w, h):
            return None, sm, "out_of_frame"
        return (ell, n), sm, None
    return None, sm, "not_found"


# ---------------- 成组孔（偏心孔） ----------------

def hole_band_mask(shape, outer_ell, r_inner_px):
    """孔的搜索环带：外圈椭圆 ×0.98，中间挖掉 r_inner_px 以内的圆。"""
    (cx, cy), (a1, a2), ang = outer_ell
    r_outer = (a1 + a2) / 4
    f = max(0.0, float(r_inner_px) / r_outer)
    mask = np.zeros(shape[:2], np.uint8)
    cv2.ellipse(mask, (int(cx), int(cy)),
                (int(a1 / 2 * 0.98), int(a2 / 2 * 0.98)), ang, 0, 360, 255, -1)
    cv2.ellipse(mask, (int(cx), int(cy)),
                (max(0, int(a1 / 2 * f)), max(0, int(a2 / 2 * f))), ang, 0, 360, 0, -1)
    return mask


def find_dark_blobs(norm, band, r_outer, thrs=HOLE_DARK_THRS):
    """多阈值找暗斑候选，跨阈值合并。

    `norm` 是局部归一化图（sm / blur(sm)），`band` 是搜索环带。
    返回 [{cx, cy, r, votes, circ, area, thr}]；`votes` = 命中它的阈值个数。

    **必须用 RETR_LIST**：环面上的暗槽会连成闭合成圈的轮廓，RETR_EXTERNAL 只返回最外层，
    槽里的孔会被当成"洞"整个丢掉（旧代码 2026-10-22 实测踩过）。
    """
    raw = []
    lo = HOLE_AREA_LO * r_outer * r_outer
    hi = HOLE_AREA_HI * r_outer * r_outer
    for thr in thrs:
        dark = cv2.bitwise_and(((norm < thr).astype(np.uint8) * 255), band)
        cnts, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            a = cv2.contourArea(c)
            if not (lo <= a <= hi):
                continue
            per = cv2.arcLength(c, True)
            circ = 4 * np.pi * a / (per * per) if per > 0 else 0.0
            if circ < HOLE_MIN_CIRC:
                continue
            _, _, bw, bh = cv2.boundingRect(c)
            if max(bw, bh) > HOLE_BBOX_MAX * r_outer:
                continue
            (mx, my), mr = cv2.minEnclosingCircle(c)
            raw.append({"cx": float(mx), "cy": float(my), "r": float(mr),
                        "circ": float(circ), "area": float(a), "thr": thr, "thrs": {thr}})

    merged = []
    for c in raw:
        for m in merged:
            if np.hypot(c["cx"] - m["cx"], c["cy"] - m["cy"]) <= HOLE_MERGE_TOL * r_outer:
                m["thrs"] |= c["thrs"]
                m["votes"] = len(m["thrs"])
                if c["circ"] > m["circ"]:          # 几何取"最像圆"的那一版
                    m.update({"cx": c["cx"], "cy": c["cy"], "r": c["r"],
                              "circ": c["circ"], "area": c["area"], "thr": c["thr"]})
                break
        else:
            n = dict(c)
            n["votes"] = 1          # 只被一档命中：没有走合并分支，必须在这里补上
            merged.append(n)
    for m in merged:
        m.pop("thrs", None)
        m["votes"] = int(m["votes"])
    return merged


def _ray_edge_limit(cx, cy, ang, ocx, ocy, edge_radii):
    """沿一条射线，从孔中心到最近一条同心边（以外圈中心为圆心）的正向距离。

    用于逐射线限制孔的搜索半径：孔常常紧挨某个环形槽，#4 边离孔中心只有几十像素，
    全局取最小值会把所有射线都砍掉；按方向算才能保留外侧/切向的干净射线。
    """
    ux, uy = np.cos(ang), np.sin(ang)
    vx, vy = cx - ocx, cy - ocy
    b = vx * ux + vy * uy
    c0 = vx * vx + vy * vy
    best = None
    for re in edge_radii:
        disc = b * b - (c0 - re * re)
        if disc < 0:
            continue
        s = np.sqrt(disc)
        for t in (-b - s, -b + s):
            if t > 1e-6 and (best is None or t < best):
                best = t
    return best


def find_bright_rings(norm, band, ocx, ocy, r_outer, thrs=HOLE_BRIGHT_THRS):
    """找"亮环"：已装紧固件时，螺钉头/沉孔口是亮环 + 暗芯 + 周围暗影。

    返回 `[{cx, cy, r, votes, circ, area, pts}]`；`pts` 是亮区外轮廓点，供 `filter_and_fit` 用。
    **暗斑法不能替代它**：暗芯和暗影会连成一个大斑，4 个孔只找到 3 个（2026-11-03 实测）。
    """
    raw = []
    lo = HOLE_RING_AREA_LO * r_outer * r_outer
    hi = HOLE_RING_AREA_HI * r_outer * r_outer
    for thr in thrs:
        bright = cv2.bitwise_and(((norm > thr).astype(np.uint8) * 255), band)
        cnts, _ = cv2.findContours(bright, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            a = cv2.contourArea(c)
            if not (lo <= a <= hi):
                continue
            per = cv2.arcLength(c, True)
            circ = 4 * np.pi * a / (per * per) if per > 0 else 0.0
            if circ < HOLE_RING_MIN_CIRC:
                continue
            (mx, my), mr = cv2.minEnclosingCircle(c)
            d = np.hypot(mx - ocx, my - ocy) / r_outer
            if not (HOLE_RING_DIST_LO <= d <= HOLE_RING_DIST_HI):
                continue
            raw.append({"cx": float(mx), "cy": float(my), "r": float(mr),
                        "circ": float(circ), "area": float(a), "thr": thr,
                        "pts": c.reshape(-1, 2).astype(float), "thrs": {thr}})

    merged = []
    for c in raw:
        for m in merged:
            if np.hypot(c["cx"] - m["cx"], c["cy"] - m["cy"]) <= HOLE_MERGE_TOL * r_outer:
                m["thrs"] |= c["thrs"]
                m["votes"] = len(m["thrs"])
                if c["circ"] > m["circ"]:
                    m.update({k: c[k] for k in
                              ("cx", "cy", "r", "circ", "area", "thr", "pts")})
                break
        else:
            n = dict(c)
            n["votes"] = 1
            merged.append(n)
    for m in merged:
        m.pop("thrs", None)
    return merged


def fit_hole_boundary(sm, cx, cy, r_blob, max_r, n_ray=N_RAY,
                      edge_radii=None, part_center=None):
    """单孔边界：72 射线找上升沿的**最大梯度**（steepest）-> 鲁棒椭圆拟合。

    返回 `(椭圆 | None, 有效点数, 对比度中位)`。
    暗平台取 0.50~0.85·r_blob、亮平台取 1.30~1.70·r_blob（被 `max_r` 截断时退到 1.15~1.45），
    两个平台只用于算对比度、判断这条射线可不可用；边界位置取梯度最大点。

    给了 `edge_radii` + `part_center` 时，**每条射线各自**限制在到最近同心边的 0.8 倍以内：
    孔常常紧挨环形槽，用一个全局半径会要么砍掉所有射线、要么让射线穿过槽壁去估亮平台。
    """
    pts, contrasts = [], []
    for k in range(n_ray):
        ang = 2 * np.pi * k / n_ray
        ray_max = max_r
        if edge_radii and part_center is not None:
            t = _ray_edge_limit(cx, cy, ang, part_center[0], part_center[1], edge_radii)
            if t is not None:
                ray_max = min(ray_max, 0.8 * t)
        if ray_max < 1.2 * r_blob:                 # 亮平台放不下，这条射线不可用
            continue
        d, g = ray_profile(sm, cx, cy, ang, ray_max)
        m_dark = (d >= 0.50 * r_blob) & (d <= 0.85 * r_blob)
        m_bright = (d >= 1.30 * r_blob) & (d <= 1.70 * r_blob)
        if m_bright.sum() < 4:                     # 亮平台被 ray_max 截断，向内侧放宽
            m_bright = (d >= 1.15 * r_blob) & (d <= 1.45 * r_blob)
        if m_dark.sum() < 4 or m_bright.sum() < 4:
            continue
        lo = float(np.median(g[m_dark]))
        hi = float(np.median(g[m_bright]))
        if hi - lo < HOLE_MIN_CONTRAST:
            continue
        level = 0.5 * (lo + hi)
        m = (d >= 0.50 * r_blob) & (d <= ray_max)
        di, gi = d[m], g[m]
        if len(gi) < 5:
            continue
        # 边界取"梯度最大"（与 find_edge_points 的 steepest 同一套定义）。
        # 50% 平台中点会被暗平台的绝对亮度带偏：同一个孔，暗平台 37 时量 116px、
        # 70 时量 102px（2026-11-02 实测）；梯度最大给出 108.5 / 104.0，稳定得多。
        gr = np.gradient(gi)
        j = int(np.argmax(gr))
        if gr[j] < HOLE_MIN_GRAD or j == 0 or j >= len(gi) - 1:
            continue
        a_, b_, c_ = gr[j - 1], gr[j], gr[j + 1]
        denom = a_ - 2 * b_ + c_
        delta = 0.5 * (a_ - c_) / denom if denom != 0 else 0.0
        rr = di[j] + 0.5 * delta          # np.gradient 逐样本差分，di 步进 0.5px
        pts.append((cx + np.cos(ang) * rr, cy + np.sin(ang) * rr))
        contrasts.append(hi - lo)

    pts = np.asarray(pts, float)
    contrast = float(np.median(contrasts)) if contrasts else 0.0
    if len(pts) < 20:
        return None, len(pts), contrast
    r0 = float(np.median(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)))
    ell, n = filter_and_fit(pts, cx, cy, r0)
    return ell, n, contrast


def validate_equidistant_circle(cands, ocx, ocy, ang_tol=5.0, r_tol=0.05,
                                ns=(3, 4, 5, 6, 8, 10, 12)):
    """等角共圆假设检验（含相位锚扫描）。返回最优结果 dict，或 None。

    不要求候选数正好等于 n：多出来的孤立候选不参与排列，但**已用的槽位必须全部命中**。
    `indices` 是入选候选在 `cands` 里的下标。

    **不用暗斑外接尺寸做一致性判据**：那个尺寸随阈值在 96~117px 之间漂，拿它卡直径会把
    6 个真孔判成"不一致"、退到 3 孔的子集。最终直径由 50% 平台中点重算，排列只看位置。
    """
    if len(cands) < 3:
        return None
    ang = np.array([np.degrees(np.arctan2(c["cy"] - ocy, c["cx"] - ocx)) % 360.0
                    for c in cands])
    rr = np.array([np.hypot(c["cx"] - ocx, c["cy"] - ocy) for c in cands])
    dias = np.array([2.0 * c["r"] for c in cands])
    best = None
    for n in ns:
        if n > len(cands):
            continue
        step = 360.0 / n
        for anchor in range(len(cands)):
            used, devs = set(), []
            for k in range(n):
                exp = (ang[anchor] + step * k) % 360.0
                dd = np.abs((ang - exp + 180.0) % 360.0 - 180.0)
                ok = ((dd <= ang_tol)
                      & (np.abs(rr - rr[anchor]) <= r_tol * rr[anchor]))
                idx = [j for j in np.where(ok)[0] if j not in used]
                if not idx:
                    used = None
                    break
                j = min(idx, key=lambda j: dd[j])
                used.add(j)
                devs.append(float(dd[j]))
            if used is None or len(used) != n:
                continue
            sub = list(used)
            item = {"kind": "equidistant_circle", "n": n, "indices": sorted(sub),
                    "ang_dev": max(devs),
                    "r_spread": float((rr[sub].max() - rr[sub].min()) / rr[sub].mean()),
                    "dia_spread": float((dias[sub].max() - dias[sub].min()) / dias[sub].mean()),
                    "ok": True}
            if best is None or (item["n"], -item["ang_dev"]) > (best["n"], -best["ang_dev"]):
                best = item
    return best


def validate_rectangle(cands, mid_tol=0.03, len_tol=0.03):
    """矩形四孔假设（旋转无关）：对角线互相平分 + 对角线等长。

    同样不使用暗斑外接尺寸做判据（理由见 `validate_equidistant_circle`）。
    """
    if len(cands) != 4:
        return None
    p = np.array([[c["cx"], c["cy"]] for c in cands], float)
    cen = p.mean(axis=0)
    order = np.argsort(np.arctan2(p[:, 1] - cen[1], p[:, 0] - cen[0]))
    q = p[order]
    d1 = float(np.linalg.norm(q[0] - q[2]))
    d2 = float(np.linalg.norm(q[1] - q[3]))
    mean_d = max((d1 + d2) / 2.0, 1e-6)
    mid_dev = float(np.linalg.norm((q[0] + q[2]) / 2.0 - (q[1] + q[3]) / 2.0) / mean_d)
    len_dev = abs(d1 - d2) / mean_d
    dias = np.array([2.0 * c["r"] for c in cands])
    dia_spread = float((dias.max() - dias.min()) / dias.mean())
    ok = mid_dev <= mid_tol and len_dev <= len_tol
    return {"kind": "rectangle", "n": 4, "indices": sorted(order.tolist()),
            "mid_dev": mid_dev, "len_dev": float(len_dev),
            "dia_spread": dia_spread, "ok": bool(ok)}


def validate_generic(cands):
    """不校验排列：≥3 个候选才收，**必须标 unchecked**。

    这是兜底策略：不给任何排列结论，只把候选和各自的直径报出来，由人判断。
    **门槛是 3 而不是 2**：实心盘上实测会冒出 1~2 个孤立暗斑，2 个候选凑不出任何排列，
    报成"2 个孔"就是假阳性。2 孔零件会落到"未成组"，需要时再单独立规则。
    """
    if len(cands) < 3:
        return None
    dias = np.array([2.0 * c["r"] for c in cands])
    spread = float((dias.max() - dias.min()) / dias.mean())
    return {"kind": "generic", "n": len(cands), "indices": list(range(len(cands))),
            "dia_spread": spread, "unchecked": True, "ok": True}


# ---------------- 外缘取边线索 ----------------

def edge_cue_evidence(sm, outer):
    """量外缘取边的两条证据：灰度差 Δ 与纹理能量比 ρ。

    返回 dict：`cue`（intensity / intensity_weak / texture_mid）、`delta`、`rho`、
    `r_int`、`r_tex`、`r_mid`。

    **强/中对比直接返回，不算纹理**：大图 Laplacian 有开销，而 88mm 这类件
    Δ≈234，永远走强度边，没必要付这笔钱。
    """
    (cx, cy), (a1, a2), _ = outer
    R = (a1 + a2) / 4
    if R <= 1:
        return None

    inner, outer_v = [], []
    for k in range(N_RAY):
        ang = 2 * np.pi * k / N_RAY
        d, p = ray_profile(sm, cx, cy, ang, R * 1.30)
        inner.append(np.median(p[(d >= 0.85 * R) & (d <= 0.95 * R)]))
        outer_v.append(np.median(p[(d >= 1.10 * R) & (d <= 1.22 * R)]))
    delta = abs(float(np.median(inner)) - float(np.median(outer_v)))

    info = {"cue": "intensity", "delta": delta, "rho": 0.0,
            "r_int": R, "r_tex": None, "r_mid": None, "gap_ratio": 0.0}
    if delta >= EDGE_CUE_DELTA_STRONG:
        return info

    # 其余都量纹理：实测 Δ=39.5 的 126mm 件仍然只有纹理边可用，
    # 20~50 这一档不能直接判"强度边够用"（见设计决策记录 §14）。
    lap = cv2.Laplacian(sm, cv2.CV_32F)
    tex = cv2.GaussianBlur(np.abs(lap), (0, 0), TEXTURE_SIGMA)
    curves = []
    for k in range(N_RAY):
        ang = 2 * np.pi * k / N_RAY
        d, e = ray_profile(tex, cx, cy, ang, R * 1.20)
        curves.append(e)
    cur = np.median(np.asarray(curves), axis=0)
    d = np.arange(0, R * 1.20, 0.5)
    e_in = float(np.median(cur[(d > 0.80 * R) & (d < 0.92 * R)]))
    e_out = float(np.median(cur[(d > 1.10 * R) & (d < 1.22 * R)]))
    rho = e_in / max(e_out, 1e-6)

    # 纹理边界：从内往外第一次跌破"内/外平台的 50%"
    lvl = 0.5 * (e_in + e_out)
    r_tex = None
    for i in range(int(0.90 * R / 0.5), int(1.10 * R / 0.5)):
        if cur[i] >= lvl > cur[i + 1]:
            r_tex = float(d[i])
            break

    info.update({"rho": rho, "r_tex": r_tex,
                 "r_mid": 0.5 * (R + r_tex) if r_tex is not None else None,
                 "gap_ratio": (r_tex - R) / R if r_tex is not None else 0.0})
    if rho >= EDGE_CUE_TEXTURE_MIN and r_tex is not None:
        info["cue"] = "texture_mid"
    else:
        info["cue"] = "intensity_weak"
    return info


def select_outer_cue(sm, outer):
    """按证据选外缘取边路径，返回 `(outer_refined, info)`。

    - `intensity` / `intensity_weak`：**原样返回**，不改一个像素（保回归基线）；
    - `texture_mid`：用「纹理边界与强度边的中点」缩放椭圆（中心不变）。

    纹理中点不是"更准的真值"，只是弱对比件在强度边失效时的替代定义；
    报告里必须写明走了哪条，不能让它看起来和强度边一样。
    """
    info = edge_cue_evidence(sm, outer)
    if info is None or info["cue"] != "texture_mid":
        return outer, info
    (cx, cy), (a1, a2), ang = outer
    R = (a1 + a2) / 4
    k = info["r_mid"] / R
    return ((cx, cy), (a1 * k, a2 * k), ang), info


def _main(argv):
    """自检：只打印外圈定位结果（真正的测量入口是 measure_ring.py）。"""
    path = argv[1] if len(argv) > 1 else "data/captures/measure_ring_latest.jpg"
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"无法读取图片: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, _, why = find_outer(gray)
    if found is None:
        raise SystemExit(f"外圈定位失败: {why}")
    ell, n = found
    (cx, cy), (a1, a2), ang = ell
    print(f"外圈: 圆心({cx:.1f},{cy:.1f})  椭圆 {a1:.1f}x{a2:.1f} 角{ang:.1f}°  点数 {n}/{N_RAY}")


if __name__ == "__main__":
    _main(sys.argv)


