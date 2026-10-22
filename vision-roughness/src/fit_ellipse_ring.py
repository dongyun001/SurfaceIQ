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


