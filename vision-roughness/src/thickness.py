r"""侧视厚度：量「零件顶面 → 基准台面」的像素差，再乘侧视 mm/px。

为什么单独一个文件：
    `measure_ring.py` 是业务入口（同心边 + 孔 + 报告），`fit_ellipse_ring.py` 是椭圆
    几何内核。侧视厚度是**一维水平边**的另一件事（逐列找第一条强边 + 两电平聚类），
    塞进任何一边都会把职责搅在一起。

它只回答一个问题：**这条侧影的顶面边与台面线差多少毫米**。不猜、不补：
看不到两条边就报失败；mm/px 没标定时只给像素，不给毫米。

物理前提（写进报告，别让读数看起来像"厚度真值"）：
    - 零件直接坐在基准台面上（有垫块时量到的是"顶面高度"，不是零件厚度）；
    - 相机纯侧视、水平；台面线在零件旁边露出；
    - 量的是侧影的顶面边，外缘倒角会吃掉一点（可用卡尺真值给 offset_mm）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

DEBUG_DIR = Path("debug")
DEBUG_JPG = DEBUG_DIR / "thickness_debug.jpg"
CACHE_PATH = DEBUG_DIR / "thickness_latest.json"

# 侧视几何默认值；现场参数在 src/config.py 的 THICKNESS 里覆盖。
DEFAULT_CFG = {
    "roi": None,             # (x0, x1) 列范围；None = 全宽
    "polarity": "dark_part", # dark_part=亮背景、暗零件/台面（背光）；bright_part=反过来
    "smooth_sigma": 1.5,
    "min_grad": 0.0,         # 0 = 按图自适应
    "min_profiles": 20,      # 每一电平至少这么多列
    "min_gap_px": 80.0,      # 顶面与台面至少差这么多像素
    "edge_search_px": 24,    # 首次越阈后，往后找梯度峰值的窗口（离焦时峰值才无偏）
    "tilt_max_deg": 0.5,     # 台面/顶面线允许的倾斜
    "max_mad_px": 5.0,       # 每一电平的离散度上限（MAD）
}


@dataclass
class ThicknessResult:
    """一次厚度测量的全部结果（CLI 打印它，GUI 显示它，绝不各自再算一遍）。"""

    ok: bool                       # 结构上量到了两条边（不代表 mm 已标定）
    note: str = ""                 # 失败原因 / 告警
    mm: Optional[float] = None     # 没标定 mm/px 时为 None
    px: Optional[float] = None
    part_y: Optional[float] = None
    base_y: Optional[float] = None
    part_tilt_deg: float = 0.0
    base_tilt_deg: float = 0.0
    n_part: int = 0
    n_base: int = 0
    part_mad_px: float = 0.0
    base_mad_px: float = 0.0
    n_columns: int = 0
    mm_per_px: Optional[float] = None
    offset_mm: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    src_desc: str = ""
    lines: list = field(default_factory=list)
    debug_img: Optional[np.ndarray] = None
    debug_jpg: Optional[Path] = None
    report_txt: Optional[Path] = None


# ---------------- 几何 ----------------

def _edges_by_column(gray: np.ndarray, cfg: dict):
    """逐列找「从上往下第一条强边」，返回 (xs, ys, min_grad)。

    dark_part：边是 亮→暗（背光背景 → 零件/台面），取 -d 的极大；
    bright_part：边是 暗→亮，取 +d 的极大。
    亚像素用三点抛物线（与项目里"最陡梯度"同一套定义，便于和同心边对照）。
    """
    g = gray.astype(np.float32)
    sigma = float(cfg.get("smooth_sigma") or 0.0)
    blur = cv2.GaussianBlur(g, (0, 0), sigma) if sigma > 0 else g

    d = np.empty_like(blur)
    d[:-1] = blur[1:] - blur[:-1]
    d[-1] = 0.0
    dd = -d if cfg.get("polarity", "dark_part") == "dark_part" else d

    min_grad = float(cfg.get("min_grad") or 0.0)
    if min_grad <= 0:
        # 用整幅的 99.5 分位当"强边"参考：两条边占的像素远少于 0.5% 以外的主体，
        # 所以这个分位落在边的梯度尾部，再乘 0.15 能照顾到离焦的缓坡。
        min_grad = max(2.0, 0.15 * float(np.percentile(np.abs(dd), 99.5)))

    h, w = blur.shape
    x0, x1 = cfg.get("roi") or (0, w)
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    xs, ys = [], []
    for x in range(x0, x1):
        col = dd[:, x]
        idx = np.flatnonzero(col >= min_grad)
        if idx.size == 0:
            continue
        y = int(idx[0])
        if y < 3 or y >= h - 3:
            continue                     # 贴边：顶面可能出画，不可信
        # 首次越阈只是"边开始出现"，离焦时那一点会随模糊宽度外移。真正的边在
        # 梯度峰值处（对称 PSF 下无偏），所以往后搜一个窗口再取极大。
        win = max(4, int(cfg.get("edge_search_px") or 24))
        lo, hi = y, min(h - 2, y + win)
        yy = lo + int(np.argmax(col[lo:hi]))
        if yy <= 0 or yy >= h - 1:
            continue
        a, b, c = float(col[yy - 1]), float(col[yy]), float(col[yy + 1])
        den = a - 2 * b + c
        dy = 0.5 * (a - c) / den if abs(den) > 1e-6 else 0.0
        xs.append(float(x))
        ys.append(float(yy) + (dy if abs(dy) <= 1.0 else 0.0))
    return np.asarray(xs, float), np.asarray(ys, float), min_grad


def _split_two_levels(ys: np.ndarray, min_profiles: int, min_gap_px: float):
    """把每列的边 y 分成上下两簇（零件顶面 / 台面）。

    用"排序后最大间隙"切分：两条真实水平线之间是几百像素，簇内离散只有几个像素，
    最大间隙一定落在两线之间；两边各要够 min_profiles 列，避免一个离群点把图切开。
    """
    if ys.size < 2 * min_profiles:
        return None
    order = np.argsort(ys)
    sy = ys[order]
    gaps = np.diff(sy)
    k = int(np.argmax(gaps))
    if gaps[k] < min_gap_px:
        return None
    upper, lower = order[:k + 1], order[k + 1:]
    if upper.size < min_profiles or lower.size < min_profiles:
        return None
    return upper, lower, float(gaps[k])


def _mad(v: np.ndarray) -> float:
    m = float(np.median(v))
    return float(np.median(np.abs(v - m)) * 1.4826)


def _tilt_deg(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 3 or float(np.ptp(xs)) < 1.0:
        return 0.0
    slope, _ = np.polyfit(xs, ys, 1)
    return float(np.degrees(np.arctan(slope)))


def measure_thickness(gray: np.ndarray, cfg: Optional[dict] = None,
                      mm_per_px: Optional[float] = None, offset_mm: float = 0.0,
                      src_desc: str = "") -> ThicknessResult:
    """量一次厚度。`ok=True` 只表示结构上找到了两条边；`mm` 需要 mm_per_px。"""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    res = ThicknessResult(ok=False, mm_per_px=mm_per_px, offset_mm=float(offset_mm),
                          src_desc=src_desc)

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    xs, ys, min_grad = _edges_by_column(gray, cfg)
    res.n_columns = int(xs.size)
    if xs.size == 0:
        res.note = ("没有找到水平边：检查背光是否亮、镜头是否对焦、"
                    "极性（THICKNESS.polarity）是否对、ROI 是否挡住了边")
        res.debug_img = _draw_debug(gray, xs, ys, res, cfg)
        res.lines = format_report(res)
        return res

    split = _split_two_levels(ys, int(cfg["min_profiles"]), float(cfg["min_gap_px"]))
    if split is None:
        res.note = (f"只找到 {xs.size} 列有效边，没能分成「零件顶面 + 台面」两条："
                    "把零件边缘（不是中心）移进画面，让台面在零件旁边露出；"
                    "仍不行就检查极性/ROI")
        res.debug_img = _draw_debug(gray, xs, ys, res, cfg)
        res.lines = format_report(res)
        return res

    upper, lower, _gap = split
    res.part_y = float(np.median(ys[upper]))
    res.base_y = float(np.median(ys[lower]))
    res.n_part, res.n_base = int(upper.size), int(lower.size)
    res.part_mad_px, res.base_mad_px = _mad(ys[upper]), _mad(ys[lower])
    res.part_tilt_deg = _tilt_deg(xs[upper], ys[upper])
    res.base_tilt_deg = _tilt_deg(xs[lower], ys[lower])
    res.px = res.base_y - res.part_y

    if res.px <= 0:
        res.note = "顶面在台面下方（px<=0）：检查画面方向与极性"
        res.debug_img = _draw_debug(gray, xs, ys, res, cfg)
        res.lines = format_report(res)
        return res

    res.ok = True
    if mm_per_px:
        res.mm = res.px * float(mm_per_px) + float(offset_mm)
    # 没标定时不写 note：报告已用 mm_per_px is None 明确写「只给像素」，
    # 免得标定成功后同一句话还留在报告里。

    warns = []
    if max(res.part_mad_px, res.base_mad_px) > float(cfg["max_mad_px"]):
        warns.append(f"边的离散度偏大（MAD {max(res.part_mad_px, res.base_mad_px):.1f}px）："
                     "相机/零件可能倾斜，或对焦/背光不均")
    tmax = float(cfg["tilt_max_deg"])
    if max(abs(res.part_tilt_deg), abs(res.base_tilt_deg)) > tmax:
        warns.append(f"水平线倾斜（顶面 {res.part_tilt_deg:+.2f}°、台面 {res.base_tilt_deg:+.2f}°）"
                     f"：把相机调水平（滚转），否则厚度带斜向误差")
    if abs(res.part_tilt_deg - res.base_tilt_deg) > tmax:
        warns.append("顶面线与台面线不平行：零件可能没放平（垫了东西/有毛刺）")
    if res.part_y is not None and res.part_y < 8:
        warns.append("顶面贴到画面顶部：零件可能超出画面，把相机抬高一点或后退")
    if warns:
        res.note = "；".join(warns)

    res.debug_img = _draw_debug(gray, xs, ys, res, cfg)
    res.lines = format_report(res)
    return res


# ---------------- 输出 ----------------

def format_report(res: ThicknessResult) -> list:
    """文字报告（与 CLI 打印、GUI 提醒共用同一份）。"""
    L = ["\n===== 厚度测量（侧视）====="]
    L.append(f"时间: {res.timestamp.strftime('%Y-%m-%d %H:%M:%S')}    图像: {res.src_desc}")
    if res.mm_per_px is not None:
        L.append(f"侧视标定: {res.mm_per_px:.6f} mm/px"
                 + (f"，偏置 {res.offset_mm:+.3f} mm" if res.offset_mm else ""))
    else:
        L.append("侧视标定: 无（本次只给像素；用 --side-ref-mm 标定后才有毫米）")

    if res.part_y is None:
        L.append(f"测量失败: {res.note}")
        L.append("提示: 侧视需要「亮背景 + 暗零件/台面」或反过来（THICKNESS.polarity）；"
                 "台面线必须在零件旁边露出。")
        return L

    L.append(f"零件顶面: y={res.part_y:9.1f}px（{res.n_part} 列，"
             f"MAD {res.part_mad_px:.1f}px，倾斜 {res.part_tilt_deg:+.2f}°）")
    L.append(f"基准台面: y={res.base_y:9.1f}px（{res.n_base} 列，"
             f"MAD {res.base_mad_px:.1f}px，倾斜 {res.base_tilt_deg:+.2f}°）")
    if res.mm is not None:
        L.append(f"厚度: {res.mm:8.3f} mm   （{res.px:.1f}px × {res.mm_per_px:.6f}"
                 + (f" + {res.offset_mm:+.3f}" if res.offset_mm else "") + "）")
    else:
        L.append(f"厚度: 未标定   （像素差 {res.px:.1f}px）")
    L.append(f"一致性: 有效列 {res.n_columns}，峰-峰 {res.px:.1f}px，"
             f"两线斜率差 {abs(res.part_tilt_deg - res.base_tilt_deg):+.2f}°")
    if res.note:
        L.append(f"⚠️  {res.note}")
    L.append("前提: 零件直接坐在基准台面上（无垫块）；量的是侧影顶面边，"
             "外缘倒角会让读数略小于图纸厚度（可用卡尺真值给 offset_mm）")
    return L


def _draw_debug(gray: np.ndarray, xs, ys, res: ThicknessResult, cfg: dict) -> np.ndarray:
    """调试图：标出两条线。文字只能用 ASCII（cv2.putText 画不了中文）。"""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
    h, w = img.shape[:2]
    x0, x1 = cfg.get("roi") or (0, w)
    cv2.rectangle(img, (int(x0), 1), (int(x1) - 1, h - 2), (0, 200, 255), 2)

    if res.part_y is not None and xs.size:
        up = ys < (res.part_y + res.base_y) / 2.0
        for x, y in zip(xs[up], ys[up]):
            cv2.circle(img, (int(x), int(round(y))), 1, (0, 255, 0), -1)
        for x, y in zip(xs[~up], ys[~up]):
            cv2.circle(img, (int(x), int(round(y))), 1, (255, 160, 0), -1)
        yp, yb = int(round(res.part_y)), int(round(res.base_y))
        cv2.line(img, (int(x0), yp), (int(x1) - 1, yp), (0, 255, 0), 2)
        cv2.line(img, (int(x0), yb), (int(x1) - 1, yb), (255, 160, 0), 2)
        if res.mm is not None:
            txt = f"t={res.mm:.3f}mm  px={res.px:.1f}"
        else:
            txt = f"px={res.px:.1f} (uncalibrated)"
        cv2.putText(img, txt, (int(x0) + 8, max(28, yp - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"top {res.part_tilt_deg:+.2f}deg  base {res.base_tilt_deg:+.2f}deg",
                    (int(x0) + 8, min(h - 12, yb + 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 160, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, "no two-level edges", (int(x0) + 8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def save_outputs(res: ThicknessResult) -> None:
    """落盘调试图 + 文字报告。路径含中文时 imwrite 会静默失败，所以用 imencode。"""
    DEBUG_DIR.mkdir(exist_ok=True)
    if res.debug_img is not None:
        ok, buf = cv2.imencode(".jpg", res.debug_img)
        if ok:
            buf.tofile(str(DEBUG_JPG))
            res.debug_jpg = DEBUG_JPG
    rep = DEBUG_DIR / f"厚度报告_{res.timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        rep.write_text("\n".join(res.lines), encoding="utf-8")
        res.report_txt = rep
    except OSError:
        res.report_txt = None


def side_mm_per_px(res: ThicknessResult, ref_mm: float) -> float:
    """标定：mm/px = 量块真值 / 实测像素差。只许用量块/卡尺真值，不许用标称直径。"""
    if not res.px or res.px <= 0:
        raise ValueError("没有量出顶面-台面的像素差，不能标定")
    return float(ref_mm) / float(res.px)


# ---------------- 厚度缓存（给外径测量用） ----------------

def write_cache(res: ThicknessResult, source: str = "side_camera") -> None:
    """把刚测的厚度写进 debug/thickness_latest.json（带时间戳，供外径测量读取）。"""
    if res.mm is None:
        return
    DEBUG_DIR.mkdir(exist_ok=True)
    data = {
        "mm": round(float(res.mm), 4),
        "px": round(float(res.px or 0.0), 2),
        "at": res.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "mm_per_px": res.mm_per_px,
        "offset_mm": res.offset_mm,
    }
    try:
        CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    except OSError:
        pass


def read_cache(max_age_min: float = 30.0):
    """读厚度缓存，返回 (mm 或 None, 测量时间字符串, 说明)。过期/损坏都拒绝。"""
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "", "没有厚度缓存（先跑 --mode thickness 测一次）"
    try:
        at = datetime.strptime(str(data["at"]), "%Y-%m-%d %H:%M:%S")
        age = (datetime.now() - at).total_seconds() / 60.0
        mm = float(data["mm"])
    except (KeyError, TypeError, ValueError):
        return None, "", "厚度缓存格式不对（删掉 debug/thickness_latest.json 重测）"
    if age > max_age_min:
        return None, str(data.get("at", "")), (f"厚度缓存已过期（{age:.0f} 分钟前，"
                                               f"上限 {max_age_min:.0f} 分钟）")
    return mm, str(data.get("at", "")), ""
