r"""通用量测核心：定圆心 + 列出所有同心边。**不命名、不给合格结论。**

这是 `docs/通用量测模式计划.md` 阶段 1/2 的产物。它与型号档案、型号识别、OK/NG 判定
全都无关，只回答一句话："我在这些半径处看到了边。"

几何原语**一律从 `fit_ellipse_ring` 复用**（`find_outer` / `find_edge_points` /
`filter_and_fit` / `_inner_ok`），不在这里另写一份——那份实现是这个项目的读数基线，
复制出来就会变成两套数。阶段 4 把它们搬过来、去掉下划线改成公开名，届时本文件的 import 会跟着改。

**既定边界（见计划文档 §4）**：圆心沿用 `find_outer`，所以**外缘必须可见**。
深色件 / 外缘低对比度的件（型号2 那种）不在本模式射程内，`find_outer` 会失败，
本模块如实回报 `not_found`，不要指望它兜住。

**同一道物理边的两个面各列一条**：外缘是一条有厚度的带，环形槽有两道壁，
两种极性（亮→暗 / 暗→亮）各列一条，由人对着图取舍——这是刻意的，不是重复。

自检入口（**临时的**，阶段 3 会把结果并进 `measure_ring.RingResult` 供 CLI/GUI 用）：

    python src\concentric.py <图片路径> [mm_per_px]
"""
import sys

import numpy as np

from fit_ellipse_ring import (
    N_RAY, _inner_ok, filter_and_fit, find_edge_points, find_outer,
)

# ---------------- 扫描参数 ----------------
# 半宽与步进的关系是这套做法成立的关键：步进必须 **小于** 半宽，带与带之间才重叠，
# 每条边才至少完整落进某一个带里（带里若挤进两条边，"最陡梯度"只会挑走陡的那条）。
LIST_F_LO = 0.16          # 向内扫到外圈半径的这个比例（再往里多是台面/噪声）
LIST_F_HI = 0.94          # 向外扫到的上限：外缘自己由 find_outer 单列，而 _inner_ok
                          #   本来就要求半径 < 0.95×外圈，扫到这儿足够
LIST_BAND_HALF = 0.018    # 候选带半宽（×外圈半径）=> 带宽 0.036
LIST_BAND_STEP = 0.025    # 相邻带的步进（×外圈半径）
LIST_MIN_GRAD = 1.2       # 亚像素找边的梯度门槛：宁可多列，靠"支撑点数"让人看出哪条虚
LIST_MERGE_TOL = 0.012    # 同极性、半径差在此比例内视为同一条边（×外圈半径）

# 极性 -> 人看的说法（报告与界面都用这句，别在两处各写一遍）
POLARITY_NAME = {-1: "亮→暗", +1: "暗→亮"}


def _scan_candidates(sm, outer):
    """沿径向枚举候选带 × 两种极性，产出通过可信判据的候选。

    返回 [(pol, ell, n, r_px, band_r0)]；`band_r0` 是带中心（×外圈半径），
    合并时用来记录"有多少个带都指向这条边"（votes）。
    """
    (ocx, ocy), (oa1, oa2), _ = outer
    r_outer = (oa1 + oa2) / 4
    out = []
    r0 = LIST_F_LO
    while r0 <= LIST_F_HI + 1e-9:
        for pol in (-1, +1):
            lo = (r0 - LIST_BAND_HALF) * r_outer
            hi = (r0 + LIST_BAND_HALF) * r_outer
            pts = find_edge_points(sm, ocx, ocy, lo, hi, pol,
                                   LIST_MIN_GRAD, n_ray=N_RAY)
            ell, n = filter_and_fit(pts, ocx, ocy, r0 * r_outer)
            # 同一条边会落进好几个重叠的带，这里不去重（交给 _merge_votes），
            # 因为"几个带都同意"本身就是这条边有多可靠的信号。
            if ell is not None and _inner_ok(ell, n, outer):
                out.append((pol, ell, n, (ell[1][0] + ell[1][1]) / 4, r0))
        r0 += LIST_BAND_STEP
    return out, r_outer


def _merge_votes(cands, r_outer):
    """把指向同一条边的候选合并，`votes` = 有多少个带同意，几何取支撑点数最多的那个。

    判"同一条边"= 极性相同 且 半径差 ≤ LIST_MERGE_TOL×外圈半径。
    **两种极性不合并**：外缘的两个面、环形槽的两道壁，是两条不同的边。
    """
    merged = []
    for pol, ell, n, r_px, r0 in cands:
        for m in merged:
            if m["pol"] == pol and abs(m["r_px"] - r_px) <= LIST_MERGE_TOL * r_outer:
                m["votes"] += 1
                if n > m["n"]:                       # 几何以支撑点数最多的一版为准
                    m.update({"ell": ell, "n": n, "r_px": r_px, "band_r0": r0})
                break
        else:
            merged.append({"pol": pol, "ell": ell, "n": n, "r_px": r_px,
                           "band_r0": r0, "votes": 1})
    return merged


def list_concentric(img, init_c=None, init_r=None):
    """定圆心 + 列出所有同心边，按半径**由外向内**编号。

    img: BGR 图（与其它入口一致）。返回 dict：
        outer       (椭圆, 点数) 或 None
        fail        None | "not_found" | "out_of_frame"（外缘没定住时的原因，原样透传）
        center / r_outer
        edges       [{idx, dia_px, r_px, a1, a2, ang, cx, cy, n, n_ray, pol, votes, src}]
                    idx 从 1 开始，第 1 条是最外面那条（外缘），src="outer" | "scan"
    """
    import cv2

    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, sm, fail = find_outer(gray, init_c, init_r)
    if found is None:
        return {"outer": None, "fail": fail, "center": None, "r_outer": None,
                "edges": []}

    outer, n_outer = found
    (ocx, ocy), (oa1, oa2), oang = outer
    r_outer = (oa1 + oa2) / 4

    cands, _ = _scan_candidates(sm, outer)
    merged = _merge_votes(cands, r_outer)
    merged.sort(key=lambda m: -m["r_px"])

    edges = [{
        "idx": 1, "dia_px": (oa1 + oa2) / 2, "r_px": r_outer,
        "a1": oa1, "a2": oa2, "ang": oang, "cx": ocx, "cy": ocy,
        "n": n_outer, "n_ray": N_RAY, "pol": None, "votes": 1, "src": "outer",
    }]
    for m in merged:
        (ecx, ecy), (ea1, ea2), eang = m["ell"]
        edges.append({
            "idx": len(edges) + 1, "dia_px": (ea1 + ea2) / 2, "r_px": m["r_px"],
            "a1": ea1, "a2": ea2, "ang": eang, "cx": ecx, "cy": ecy,
            "n": m["n"], "n_ray": N_RAY, "pol": m["pol"], "votes": m["votes"],
            "src": "scan",
        })
    return {"outer": outer, "fail": None, "center": (ocx, ocy),
            "r_outer": r_outer, "edges": edges}


def _main(argv):
    import cv2

    if len(argv) < 2:
        raise SystemExit("用法: python src\\concentric.py <图片路径> [mm_per_px]")
    path = argv[1]
    ratio = float(argv[2]) if len(argv) > 2 else None

    img = cv2.imread(path)
    if img is None:                     # 中文路径下 imread 会静默返回 None
        raise SystemExit(f"无法读取图片: {path}")

    res = list_concentric(img)
    if res["outer"] is None:
        raise SystemExit(f"外缘没定住（{res['fail']}）——通用模式要求外缘可见，"
                         "低对比度件不在射程内（见 docs/通用量测模式计划.md §4）")

    print(f"图: {path}    外圈半径 {res['r_outer']:.1f}px")
    if ratio is None:
        print("（没给 mm/px：只报像素）")
    else:
        print(f"换算系数: {ratio:.6f} mm/px")
    print(f"\n检出 {len(res['edges'])} 条同心边（按半径由外向内编号）：")
    for e in res["edges"]:
        tag = "外缘(雷达定位)" if e["pol"] is None else POLARITY_NAME[e["pol"]]
        mm = "" if ratio is None else f"  {e['dia_px'] * ratio:9.3f} mm"
        print(f"  #{e['idx']:<3d} {e['dia_px']:9.1f} px  "
              f"椭圆 {e['a1']:.1f}x{e['a2']:.1f} 角{e['ang']:.1f}°{mm}  "
              f"[{tag}  点 {e['n']}/{e['n_ray']}  带同意 {e['votes']}  {e['src']}]")


if __name__ == "__main__":
    _main(sys.argv)
