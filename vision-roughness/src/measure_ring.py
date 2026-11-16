r"""零件尺寸测量：**列出检出的一条条同心边**（不命名、不判合格），一份机位系数换毫米。

这是"通用量测模式"的入口（`docs/设计决策记录.md` §11）。它只回答两个问题：
**我在这些半径处看到了边**、**这些边换算成毫米是多少**。"哪条边是外径/中心孔/内孔"
和"合不合格"都不归它管——由人对着标注图和图纸判断。所以：

- **不做型号识别**（2026-10-23 删除）：不指定型号就没有"型号"这回事。
- **不做 OK/NG 判定**：没有命名就没有可判的量。标称值要判的话请人工对照。
- **换算系数只有一份**（`models/rig.json`）：mm/px 是机位的属性，不是零件的属性。

三种用法：
    1) 图形界面（日常用，推荐）：双击项目里的 `测量.bat`
       - 相机已接：自动拍照 → 弹出结果窗口（标注图 + 清单）
       - 想测已有照片：把照片文件拖到 `测量.bat` 图标上
       - 换上 0.3x 远心镜头后，点「测厚度」按钮走侧视厚度
    2) 命令行（标定 / 排错用）：项目根目录 vision-roughness/ 下
       python src\measure_ring.py --image data\captures\xxx.jpg
       python src\measure_ring.py --image <该机位的图> --ref-mm 88.28   # 标定机位系数
       python src\measure_ring.py --mode thickness --side-ref-mm 9.0    # 标定侧视厚度系数
       python src\measure_ring.py --mode thickness                      # 测厚度（写缓存）
       python src\measure_ring.py --image <图> --thickness 6.5          # 外径+高度补偿

每次测量会在 debug\ 下存两份记录：measure_ring_debug.jpg（标注图）
和 报告_年月日_时分秒.txt（文字报告）。

原理：
    mm/px 是「相机 + 镜头 + 高度 + 倾角 + 曝光」的函数，整台设备只存一份（models/rig.json）。
    ⚠️ 相机动了/零件高度变了/换镜头/改曝光，必须重新标定；不确定度约 ±1.4%
    （同一机位两个零件反推系数差 1.40%，原因见计划文档：外缘是一条有厚度的带）。
    没有系数时只输出像素——拿别的机位的系数印毫米数是错的，所以不做。

检测管线（`concentric.list_concentric`）：
    定圆心 → `find_outer`（外缘必须可见；低对比度场景会如实报失败）
    列边   → 沿径向枚举候选带 × 两种极性，亚像素射线找边 + 鲁棒椭圆拟合，
             同极性的重复候选合并，按半径由外向内编号
"""
import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, "src")
import annotate
import concentric
import config
import fit_ellipse_ring
import thickness
from camera_adapter import create_camera

RIG_PATH = Path("models/rig.json")     # 机位档案（mm/px 的唯一真相源）
DEBUG_DIR = Path("debug")
BORDER_MARGIN = 5  # 轮廓距图像边缘小于此值视为"零件超出视野"


# ---------------- 机位系数（唯一真相源） ----------------

def load_rig() -> dict:
    """读机位档案。读不到就返回空 dict（调用方据此只给像素，不编毫米）。"""
    try:
        return json.loads(RIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_rig(ratio: float, src: str) -> None:
    """标定结果写回机位档案。**必须保留说明与不确定度**，否则后人不知道这个数怎么来的。"""
    rig = load_rig()
    rig["mm_per_px"] = float(ratio)
    rig["calibrated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    rig["calibrated_from"] = src
    rig.setdefault("uncertainty", "±1.4%")
    RIG_PATH.write_text(json.dumps(rig, ensure_ascii=False, indent=2), encoding="utf-8")


class MeasureError(Exception):
    """可以预料、需要原样提示给用户看的测量失败（不是程序缺陷）。"""


# ---------------- 结果 ----------------

@dataclass
class RingResult:
    """一次测量的全部结果：清单 + 文本报告 + 标注图。

    供命令行直接打印，也供 gui_ring.py 显示——两条路走同一份数据，
    不会出现"窗口一个数、报告另一个数"。
    """
    src_desc: str
    timestamp: datetime
    ratio: Optional[float]          # None = 没标定过机位，只给像素
    ratio_src: str
    outer_px: float                 # 最外那条边（#1）的均值直径
    outer_ellipse: tuple
    outer_mm: Optional[float]
    edges: list                     # **要报的**清单（默认只含强边 + 外缘）
    hidden: int = 0                 # 检出但没列的弱/中边条数（默认隐藏，见 --all-edges）
    holes: list = field(default_factory=list)   # 成组偏心孔（与 edges 并列的另一条分支）
    holes_note: str = ""                        # 孔的排列结论 / 失败原因
    outer_cue: str = "intensity"                # 外缘取边路径：intensity / intensity_weak / texture_mid
    outer_cue_info: dict = field(default_factory=dict)
    # ---- 厚度 / 高度补偿 ----
    thickness_mm: Optional[float] = None        # 本次用的厚度（None = 没有，未补偿）
    thickness_src: str = ""                     # 来源说明（缓存/命令行/未提供）
    ratio_base: Optional[float] = None          # 补偿前的机位系数
    height_factor: float = 1.0                  # 补偿倍数（1.0 = 未补偿）
    height_applied: bool = False
    height_note: str = ""                       # 未补偿的原因 / 告警
    tilt: float = 1.0
    touches_border: bool = False
    warnings: list = field(default_factory=list)
    lines: list = field(default_factory=list)      # 文本报告（与命令行输出一致）
    debug_jpg: Optional[Path] = None
    report_txt: Optional[Path] = None
    debug_img: Optional[np.ndarray] = None         # 标注图（GUI 缩略图用）


@dataclass
class RectResult:
    """一次**方形/矩形件**测量的全部结果（与 RingResult 并列，不混用字段）。

    `side_a_px` / `side_b_px` 是两组**对边间距**（a = 较长的那组）；四条边长是把
    直边延长到虚拟角点后算的，圆角不会让它们变短。`aspect = a/b`：真方形接近 1，
    矩形件就是长宽比——**这不是"方形度合格判据"**，只是证据。
    """
    src_desc: str
    timestamp: datetime
    ratio: Optional[float]
    ratio_src: str
    side_a_px: float
    side_b_px: float
    side_a_mm: Optional[float]
    side_b_mm: Optional[float]
    side_lengths: list
    side_lengths_mm: list
    diagonals: list
    diag_diff_px: float
    diag_diff_pct: float
    aspect: float
    angle: float
    parallel_dev: list
    perp_dev: list
    sides: list
    center: tuple
    thresh: float
    rectness: float
    # ---- 厚度 / 高度补偿（与圆件同一套模型）----
    thickness_mm: Optional[float] = None
    thickness_src: str = ""
    ratio_base: Optional[float] = None
    height_factor: float = 1.0
    height_applied: bool = False
    height_note: str = ""
    warnings: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    debug_jpg: Optional[Path] = None
    report_txt: Optional[Path] = None
    debug_img: Optional[np.ndarray] = None


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。GUI 复用同一套，保证拖照片/标定两种用法参数一致。"""
    p = argparse.ArgumentParser(description="零件尺寸测量（不命名、不判合格）")
    p.add_argument("--shape", choices=["circle", "rect"], default="circle",
                   help="circle=圆形件（默认，列同心边）；"
                        "rect=方形/矩形件（四条边，按对边距出宽高）")
    p.add_argument("--mode", choices=["diameter", "thickness"], default="diameter",
                   help="diameter=同心边/孔（默认）；thickness=侧视厚度"
                        "（同一台相机换 0.3x 远心镜头后使用）")
    p.add_argument("--image", help="测已有照片；不指定则相机现拍")
    p.add_argument("--ref-mm", type=float,
                   help="卡尺真值(mm)，用于标定**机位**的换算系数（会写回 models/rig.json）；"
                        "圆形件=外径，方形件=长边")
    p.add_argument("--mm-per-px", type=float, help="直接指定 mm/px（临时用，不写回档案）")
    p.add_argument("--thickness", type=float,
                   help="本次零件厚度/顶面高度(mm)，用于高度补偿；优先于厚度缓存")
    p.add_argument("--no-height", action="store_true",
                   help="本次不做高度补偿（对比/排错用）")
    p.add_argument("--side-ref-mm", type=float,
                   help="厚度标定：量块真值高度(mm)。配合 --mode thickness，"
                        "把侧视 mm/px 写回 models/rig.json")
    p.add_argument("--side-mm-per-px", type=float,
                   help="临时指定侧视 mm/px（不写回档案，测试用）")
    p.add_argument("--side-offset-mm", type=float,
                   help="厚度偏置校正(mm)；默认用档案里的值（倒角/取边偏置）")
    p.add_argument("--all-edges", action="store_true",
                   help="连'弱边'一起列（仅圆形模式）。默认只列强边（+外缘）——弱边在低对比度件上"
                        "**不稳定**（实测连拍 5 帧条数在 11~15 之间跳、直径极差 7~11%%），"
                        "列出来会误导人")
    return p


def grab_image(args, save_name: str = "measure_ring_latest.jpg") -> np.ndarray:
    """从相机拍一张，或从文件读一张，返回 BGR 图像。"""
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise MeasureError(f"无法读取图片: {args.image}")
        return img
    print("打开相机拍照...")
    with create_camera(config.CAMERA_TYPE, **config.CAMERA_KWARGS) as cam:
        frame = cam.capture()
    out = Path("data/captures") / save_name
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"已保存: {out}")
    return frame


def _height_model() -> dict:
    return load_rig().get("height_model") or {}


def resolve_ratio(args, ref_px, thickness_mm=None, what="外径"):
    """确定 mm/px，返回 (ratio 或 None, 来源说明, 高度补偿信息)。

    唯一真相源是**机位档案** models/rig.json（全局一份）。拿不到就返回 None：
    调用方据此只报像素、不给毫米，而不是编一个数。

    `ref_px` 是标定用的实测像素量：圆形件传外径，方形件传**长边对边距**；
    `what` 只影响报告里的措辞（"外径真值" / "长边真值"），不改变算法。

    高度补偿（`height_model.enabled=true` 且拿到厚度 t 时才生效）：
        s(t) = mm_per_px × (1 + scale_per_mm × (t - thickness_ref_mm))
    符号：顶面越高（越靠近镜头），真实 mm/px 越小 → scale_per_mm 是负数。
    **只改毫米换算，不动任何像素读数。**

    `--ref-mm` 标定时若同时给了厚度，会把系数**归一化到基准高度**再存盘，
    保证 mm_per_px 始终是"基准厚度下"的系数，不出现第二套定义。
    """
    hm = _height_model()
    model_on = bool(hm.get("enabled")) and not getattr(args, "no_height", False)
    t_ref = hm.get("thickness_ref_mm")
    scale = hm.get("scale_per_mm")

    info = {
        "enabled": bool(hm.get("enabled")), "applied": False, "base": None,
        "factor": 1.0, "t": thickness_mm, "t_ref": t_ref, "scale": scale,
        "note": "",
    }

    factor = 1.0
    if model_on and thickness_mm is not None and t_ref is not None and scale is not None:
        factor = 1.0 + float(scale) * (float(thickness_mm) - float(t_ref))
        info.update(factor=factor, applied=True)
    elif not hm.get("enabled"):
        info["note"] = "高度模型未启用（height_model.enabled=false）"
    elif getattr(args, "no_height", False):
        info["note"] = "命令行 --no-height：本次不做高度补偿"
    else:
        info["note"] = "本次没有厚度，未做高度补偿（先测厚度或用 --thickness）"

    if args.mm_per_px:
        base = float(args.mm_per_px)
        src = "命令行指定（临时，不写回档案）"
    elif args.ref_mm:
        measured = float(args.ref_mm) / ref_px
        if info["applied"]:
            base = measured / factor
            src = (f"本次标定：{what}真值 {args.ref_mm}mm / 检出 {ref_px:.1f}px"
                   f"（已按厚度 {thickness_mm:.2f}mm 归一化到基准 {float(t_ref):.2f}mm）")
        else:
            base = measured
            src = (f"本次标定：{what}真值 {args.ref_mm}mm / 检出 {ref_px:.1f}px"
                   f"（图像 {args.image or '相机现拍'}）")
        save_rig(base, src)
        src += "，已写回机位档案"
        if info["enabled"] and not info["applied"] and t_ref is not None:
            info["note"] = ("标定时没给 --thickness：系数按基准 "
                            f"{float(t_ref):.2f}mm 存；若标定件不是这个厚度，高度补偿会偏")
    else:
        rig = load_rig()
        if not rig.get("mm_per_px"):
            return None, "机位没标定过（models/rig.json 里没有 mm_per_px）", info
        base = float(rig["mm_per_px"])
        src = f"机位档案（{rig.get('calibrated_at', '未知时间')} 标定）"

    info["base"] = base
    ratio = base * factor if info["applied"] else base
    return ratio, src, info


def resolve_thickness(args):
    """本次测量要用的厚度，返回 (mm 或 None, 来源说明, 备注)。

    优先级：`--thickness` > `config.THICKNESS.source`（latest=厚度缓存 / manual=不自动取）。
    缓存带时间戳，过期就拒绝——宁可少补偿，也不拿上一次的厚度套到另一个零件上。
    """
    if getattr(args, "thickness", None) is not None:
        return float(args.thickness), "命令行 --thickness", ""
    cfg = getattr(config, "THICKNESS", {}) or {}
    src = cfg.get("source", "latest")
    if src == "manual":
        return None, "未提供（THICKNESS.source=manual）", "没有厚度"
    t, at, note = thickness.read_cache(float(cfg.get("max_age_min", 30)))
    if t is None:
        return None, note, note
    return t, f"厚度缓存（{at}，侧视）", ""


# ---------------- 侧视厚度（同一台相机换 0.3x 远心镜头） ----------------

def load_side_calib() -> dict:
    return (_height_model().get("side") or {})


def save_side_calibration(mm_per_px: float, offset_mm: float, src: str) -> None:
    """侧视 mm/px 写回机位档案的 height_model.side（与顶视 mm_per_px 分名字空间）。"""
    rig = load_rig()
    hm = rig.setdefault("height_model", {})
    side = hm.setdefault("side", {})
    side["mm_per_px"] = float(mm_per_px)
    side["offset_mm"] = float(offset_mm)
    side["calibrated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    side["from"] = src
    RIG_PATH.write_text(json.dumps(rig, ensure_ascii=False, indent=2), encoding="utf-8")


def run_thickness_on_frame(frame: np.ndarray, args):
    """对一帧侧视画面测厚度（CLI 与 GUI 共用；含落盘与缓存）。"""
    side = load_side_calib()
    mpp = getattr(args, "side_mm_per_px", None)
    if mpp is None:
        mpp = side.get("mm_per_px")
    off = getattr(args, "side_offset_mm", None)
    if off is None:
        off = side.get("offset_mm", 0.0) or 0.0
    cfg = getattr(config, "THICKNESS", {}) or {}
    res = thickness.measure_thickness(
        frame, cfg, mm_per_px=mpp, offset_mm=off,
        src_desc=getattr(args, "image", None) or "相机现拍")
    if res.ok and res.mm is not None:
        thickness.write_cache(res)
    thickness.save_outputs(res)
    return res


def run_thickness_mode(args) -> None:
    """CLI 的 --mode thickness：标定或测厚度，打印报告后返回。"""
    img = grab_image(args, save_name="thickness_latest.jpg")

    if args.side_ref_mm is not None:
        cfg = getattr(config, "THICKNESS", {}) or {}
        res = thickness.measure_thickness(img, cfg, mm_per_px=None, offset_mm=0.0,
                                          src_desc=args.image or "相机现拍")
        if not res.ok or not res.px or res.px <= 0:
            thickness.save_outputs(res)
            raise MeasureError("厚度标定失败：" + (res.note or "没有量出像素差"))
        mpp = thickness.side_mm_per_px(res, float(args.side_ref_mm))
        off = float(args.side_offset_mm or 0.0)
        src = (f"量块真值 {args.side_ref_mm}mm / 实测 {res.px:.1f}px"
               f"（图像 {args.image or '相机现拍'}）")
        save_side_calibration(mpp, off, src)
        res.mm_per_px, res.offset_mm = mpp, off
        res.mm = float(args.side_ref_mm)
        res.lines = thickness.format_report(res)
        res.lines.append(f"\n侧视标定已写回 {RIG_PATH}：{mpp:.6f} mm/px（{src}）")
        thickness.save_outputs(res)
        for ln in res.lines:
            print(ln)
        return

    res = run_thickness_on_frame(img, args)
    for ln in res.lines:
        print(ln)
    if not res.ok:
        raise MeasureError("厚度测量失败（原因见上面的报告；调试图 debug/thickness_debug.jpg）")
    if res.mm is None:
        raise MeasureError("厚度没有标定：先跑 `--mode thickness --side-ref-mm <量块真值>`")


# ---------------- 失败提示 ----------------

def locate_fail_hint(img: np.ndarray) -> str:
    """外圈定位失败时，尽量说清到底是"太暗/过曝"还是别的原因。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean, bright = float(g.mean()), float((g > 200).mean())
    if mean < 60 or bright < 0.005:
        return (f"外圈拟合失败：画面太暗（灰度均值 {mean:.0f}/255，亮像素 {bright * 100:.1f}%）。\n"
                "先补光或把曝光调高再重拍。")
    if bright > 0.30:
        return (f"外圈拟合失败：画面过曝（亮像素 {bright * 100:.1f}%）。\n"
                "把曝光调低或减少直射反光再重拍。")
    return "外圈拟合失败。检查：对焦、光照、零件是否在画面中心"


def _outer_fail_message(img: np.ndarray, reason) -> str:
    """外圈定位失败的具体原因（`out_of_frame` 必须说清，否则会拿内孔直径当外径）。"""
    if reason == "out_of_frame":
        return ("外圈超出画面：检出的是内孔（更外面的圆被画幅切掉了）。\n"
                "把零件移入画面中心，或把相机抬高/后拉，让整圈外缘都在画面内再测。")
    return locate_fail_hint(img)


# ---------------- 成组孔（偏心孔） ----------------

def measure_holes(img: np.ndarray, res: dict, ratio):
    """成组偏心孔：与同心边并列的另一条分支。

    检测层与排列无关（只找暗色圆斑），排列校验是可插拔策略
    （`fit_ellipse_ring.validate_equidistant_circle` / `validate_rectangle` / `validate_generic`）。

    返回 `(holes, note, warnings)`。找不到就返回空列表——**不抛异常**：
    "这件没有成组孔"是正常结果，不是测量失败。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sm = cv2.GaussianBlur(gray, (0, 0), 5)

    (ocx, ocy), (oa1, oa2), _ = res["outer"]
    r_outer = float(res["r_outer"])
    # 局部归一化窗口随 R 缩放（旧代码写死 201，对应 R≈1013；换零件/机位才不跑偏）
    win = max(31, int(round(0.2 * r_outer)) | 1)
    norm = sm / np.maximum(cv2.blur(sm, (win, win)), 1.0)

    # 内边界取"最内侧的非外缘边"×1.05；只有外缘时（实心盘）退到 0.55R
    inner = [e for e in res["edges"] if e.get("src") != "outer"]
    r_inner = (min(e["r_px"] for e in inner) * 1.05 if inner else 0.55 * r_outer)
    band = fit_ellipse_ring.hole_band_mask(gray.shape, res["outer"], r_inner)

    cands = fit_ellipse_ring.find_dark_blobs(norm, band, r_outer)

    # 先只试暗斑的强排列（等角圆 / 矩形）。**generic 放到亮环之后再兜底**：
    # 装好紧固件的件上，暗芯+暗影会凑出 3 个假候选，generic 会把它们当孔收下，
    # 反而挡住更可靠的亮环路径（2026-11-03 实测）。
    checks = []
    if cands:
        checks = [
            fit_ellipse_ring.validate_equidistant_circle(cands, ocx, ocy),
            fit_ellipse_ring.validate_rectangle(cands),
        ]
    verdict = next((v for v in checks if v and v.get("ok")), None)
    if verdict is None:
        # 暗斑不成组 -> 试"亮环"（已装紧固件的孔：螺钉头/沉孔口）
        rings = fit_ellipse_ring.find_bright_rings(norm, band, ocx, ocy, r_outer)
        rv = fit_ellipse_ring.validate_rectangle(rings) if len(rings) == 4 else None
        if rv and rv.get("ok"):
            holes, warns = [], []
            for i, j in enumerate(rv["indices"], start=1):
                ring = rings[j]
                pts = np.asarray(ring["pts"], float)
                ell, n = (fit_ellipse_ring.filter_and_fit(
                    pts, ring["cx"], ring["cy"], ring["r"])
                    if len(pts) >= 20 else (None, 0))
                d_c = float(np.hypot(ring["cx"] - ocx, ring["cy"] - ocy))
                ang = float(np.degrees(np.arctan2(ring["cy"] - ocy,
                                                  ring["cx"] - ocx)) % 360)
                h = {"idx": i, "votes": int(ring["votes"]),
                     "circularity": float(ring["circ"]), "contrast": 0.0,
                     "boundary": "bright_ring"}
                if ell is None:
                    h.update({"cx": float(ring["cx"]), "cy": float(ring["cy"]),
                              "r_norm": d_c / r_outer, "angle_deg": ang,
                              "dia_px": 0.0, "a1": 0.0, "a2": 0.0, "ang": 0.0,
                              "n_points": int(n), "aspect": 0.0, "status": "失败"})
                else:
                    (ex, ey), (ea1, ea2), eang = ell
                    h.update({"cx": float(ex), "cy": float(ey),
                              "r_norm": float(np.hypot(ex - ocx, ey - ocy) / r_outer),
                              "angle_deg": float(np.degrees(np.arctan2(
                                  ey - ocy, ex - ocx)) % 360),
                              "dia_px": float((ea1 + ea2) / 2),
                              "a1": float(ea1), "a2": float(ea2), "ang": float(eang),
                              "n_points": int(n),
                              "aspect": float(max(ea1, ea2) / max(min(ea1, ea2), 1e-6)),
                              "status": ("OK" if n >= fit_ellipse_ring.HOLE_LOW_POINTS
                                         else "低置信度")})
                holes.append(h)
            warns.append("孔：走的是亮环边界（螺钉头/沉孔口），不是通孔内壁；含义与暗斑孔不同")
            n_bad = sum(1 for h in holes if h["status"] != "OK")
            if n_bad:
                warns.append(f"孔：{n_bad} 个亮环边界低置信度")
            return holes, f"{len(holes)} 个（矩形排列，亮环边界；已装紧固件）", warns
        # 亮环也不成组 -> 暗斑 generic 兜底（≥3 才收，明确标未校验）
        generic = fit_ellipse_ring.validate_generic(cands)
        if generic and generic.get("ok"):
            verdict = generic
        else:
            if not cands and not rings:
                return [], "未发现暗色成组孔或亮环孔", []
            return [], (f"找到 {len(cands)} 个暗候选、{len(rings)} 个亮环候选，"
                        "均未成组"), [f"孔：{len(cands)} 暗 + {len(rings)} 亮环，未成组"]

    edge_r = [float(e["r_px"]) for e in res["edges"]]
    holes, warns = [], []
    for i, j in enumerate(verdict["indices"], start=1):
        c = cands[j]
        d_c = float(np.hypot(c["cx"] - ocx, c["cy"] - ocy))
        ang = float(np.degrees(np.arctan2(c["cy"] - ocy, c["cx"] - ocx)) % 360)
        # 倒角外缘：只作参考（用户要的是里面那条螺孔内缘，不是这个）
        max_r = min(2.6 * c["r"], 0.5 * r_outer)
        ell_out, _, _ = fit_ellipse_ring.fit_hole_boundary(
            sm, c["cx"], c["cy"], c["r"], max_r,
            edge_radii=edge_r, part_center=(ocx, ocy))
        dia_out = 0.0 if ell_out is None else (ell_out[1][0] + ell_out[1][1]) / 2
        # 螺孔内缘：螺纹/孔壁 -> 亮倒角的第一条梯度峰；r_ref 用倒角外缘的拟合半径
        r_ref = (dia_out / 2) if dia_out > 0 else c["r"]
        ell, n, contrast = fit_ellipse_ring.fit_hole_inner_boundary(
            sm, c["cx"], c["cy"], r_ref)
        boundary = "hole_inner"
        h = {"idx": i, "votes": int(c["votes"]), "circularity": float(c["circ"]),
             "contrast": float(contrast), "boundary": boundary,
             "dia_outer_px": float(dia_out)}
        if ell is None:
            h.update({"cx": float(c["cx"]), "cy": float(c["cy"]), "r_norm": d_c / r_outer,
                      "angle_deg": ang, "dia_px": 0.0, "a1": 0.0, "a2": 0.0, "ang": 0.0,
                      "n_points": int(n), "aspect": 0.0, "status": "失败"})
        else:
            (ex, ey), (ea1, ea2), eang = ell
            h.update({"cx": float(ex), "cy": float(ey),
                      "r_norm": float(np.hypot(ex - ocx, ey - ocy) / r_outer),
                      "angle_deg": float(np.degrees(np.arctan2(ey - ocy, ex - ocx)) % 360),
                      "dia_px": float((ea1 + ea2) / 2), "a1": float(ea1), "a2": float(ea2),
                      "ang": float(eang), "n_points": int(n),
                      "aspect": float(max(ea1, ea2) / max(min(ea1, ea2), 1e-6)),
                      "status": ("OK" if (n >= fit_ellipse_ring.HOLE_INNER_MIN_POINTS
                                          and c["votes"] >= 4
                                          and boundary == "hole_inner") else "低置信度")})
        holes.append(h)

    if verdict["kind"] == "equidistant_circle":
        note = f"{len(holes)} 个（等角圆排列，角度残差最大 {verdict['ang_dev']:.1f}°）"
    elif verdict["kind"] == "rectangle":
        note = f"{len(holes)} 个（矩形排列，对角线中点偏差 {verdict['mid_dev'] * 100:.1f}%）"
    else:
        note = f"{len(holes)} 个（未校验排列）"
    if verdict.get("unchecked"):
        warns.append("孔：未做排列校验（generic），候选里可能混有非孔暗斑，读数只能参考")
    n_bad = sum(1 for h in holes if h["status"] != "OK")
    if n_bad:
        warns.append(f"孔：{n_bad} 个边界失败/低置信度（点数少或轴比大），读数只能参考")
    return holes, note, warns


# ---------------- 测量 ----------------

def measure(img: np.ndarray, args) -> RingResult:
    """定圆心 → 列同心边 → 换算 → 落盘（标注图 + 文字报告），返回结构化结果。"""
    res = concentric.list_concentric(img)
    if res["outer"] is None:
        raise MeasureError(_outer_fail_message(img, res["fail"]))

    outer = res["outer"]                    # ((cx,cy),(a1,a2),ang)；点数是 edges[0]["n"]
    low_contrast = bool(res.get("low_contrast"))
    (ocx, ocy), (oa1, oa2), oang = outer
    outer_dia_px = (oa1 + oa2) / 2
    detected = res["edges"]
    # 只报"强边 + 外缘"：弱边在低对比度件上时隐时现（实测连拍 5 帧条数 11~15 跳、直径极差 7~11%），
    # 列出来等于给人一堆不可复现的数。--all-edges 才全列。
    if args.all_edges:
        edges = detected
    else:
        edges = [e for e in detected if e["strength"] == 2 or e["src"] == "outer"]
    hidden = len(detected) - len(edges)

    lines: list = []
    warnings: list = []

    def _p(s=""):
        lines.append(s)

    timestamp = datetime.now()
    src_desc = args.image or "相机现拍"
    t_mm, t_src, _ = resolve_thickness(args)
    ratio, ratio_src, hinfo = resolve_ratio(args, outer_dia_px, t_mm)
    od_mm = None if ratio is None else outer_dia_px * ratio

    touches_border = (oa1 > img.shape[1] - BORDER_MARGIN * 2 or
                      oa2 > img.shape[0] - BORDER_MARGIN * 2)
    if touches_border:
        _p("⚠️  警告：零件轮廓贴到图像边缘，零件没拍全，测量结果不可信！")
        warnings.append("零件超出画面：轮廓贴到图像边缘，结果不可信，请把相机架高重拍")

    _p("\n===== 测量报告 =====")
    _p(f"时间: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}    图像: {src_desc}")
    _p("模式: 通用量测（**不命名**——哪条边是哪个尺寸，由人对着标注图和图纸判断）")
    if low_contrast:
        _p(f"⚠️  外缘对比度低：默认梯度门槛 2.0 下没定住，已降到 {concentric.OUTER_LOW_GRAD} 重试成功。")
        _p("    这类读数对光照/对焦更敏感，不确定度大于 ±1.4%，别当精密量具用。")
        warnings.append("外缘对比度低（已降低梯度门槛），读数不确定度更大")
    # ---- 外缘取边路径：报告必须写明走了哪条，别让它看起来和强度边一样 ----
    cue = res.get("outer_cue", "intensity")
    cue_info = res.get("outer_cue_info") or {}
    if cue == "texture_mid":
        _p(f"外缘取边：纹理中点（灰度差 {cue_info.get('delta', 0):.0f}、"
           f"纹理比 {cue_info.get('rho', 0):.1f}×）")
        _p("    强度边不足以定边，改用「纹理边界与强度边的中点」；这是另一套定义，不确定度更大。")
        warnings.append("外缘改用纹理中点（弱对比件）：定义与强度边不同，不确定度更大")
        if cue_info.get("gap_ratio", 1.0) < fit_ellipse_ring.EDGE_CUE_MIN_GAP:
            _p(f"    ⚠️ 纹理边界离强度边只有 {cue_info['gap_ratio'] * 100:.1f}%，"
               "背景对比可能不足；建议换乳白背景")
            warnings.append("纹理边界离强度边太近：背景对比不足，建议换乳白背景")
    elif cue == "intensity_weak":
        _p(f"⚠️  外缘取边：强度边（灰度差仅 {cue_info.get('delta', 0):.0f}，边界不可靠）")
        warnings.append("外缘灰度对比不足，边界不可靠；建议换背景或补光")
    if ratio is None:
        _p(f"换算系数: 无 —— {ratio_src}")
        _p("        没有系数就**只报像素、不给毫米**（拿别的机位的系数印毫米数是错的）")
        warnings.append("机位没标定过，本次只给像素；用 --ref-mm 标定后才有毫米")
    else:
        unc = load_rig().get("uncertainty", "")
        _p(f"换算系数: {ratio:.6f} mm/px （来源: {ratio_src}）"
           + (f"  不确定度 {unc}" if unc else ""))

    # ---- 厚度 / 高度补偿：它会让毫米读数变化，必须写清楚依据 ----
    if t_mm is not None:
        _p(f"厚度: {t_mm:.2f} mm（来源: {t_src}）")
    if hinfo["applied"] and ratio is not None:
        base = float(hinfo["base"])
        pct = (float(hinfo["factor"]) - 1.0) * 100.0
        _p(f"高度补偿: 基准 {float(hinfo['t_ref']):.2f} mm，"
           f"{float(hinfo['scale']) * 100:+.3f}%/mm → "
           f"{base:.6f} → {ratio:.6f} mm/px（{pct:+.2f}%）")
        _p("    假定零件直接坐在同一基准台面上（有垫块时量的是顶面高度）；"
           "残余约 ±0.17% 是取边带，不是厚度误差")
    elif hinfo["enabled"] and hinfo["note"]:
        _p(f"高度补偿: 未做 —— {hinfo['note']}")
        warnings.append(hinfo["note"])

    tilt = max(oa1, oa2) / min(oa1, oa2)
    if tilt > 1.02:
        tilt_deg = np.degrees(np.arccos(min(oa1, oa2) / max(oa1, oa2)))
        _p(f"⚠️  透视提醒：外圈椭圆长短轴比 {tilt:.3f}（相当于斜视 {tilt_deg:.0f}°），"
           "各尺寸的毫米读数带位置相关误差；把相机垂直朝下正对零件可消除")
        warnings.append(f"相机斜视约 {tilt_deg:.0f}°（外圈椭圆比 {tilt:.3f}）：读数带位置相关误差")

    _p(f"\n检出 {len(edges)} 条同心边（按半径由外向内编号）：")
    for e in edges:
        tag = "外缘" if e["pol"] is None else concentric.POLARITY_NAME[e["pol"]]
        mm = "" if ratio is None else f"  {e['dia_px'] * ratio:8.3f} mm"
        lvl = concentric.STRENGTH_NAME[e["strength"]]
        _p(f"  #{e['idx']:<3d} {e['dia_px']:9.1f} px{mm}  "
           f"椭圆 {e['a1']:.1f}x{e['a2']:.1f} 角{e['ang']:.1f}°"
           f"  [{tag}  点 {e['n']}/{e['n_ray']}  强度 {lvl} 落差 {e['step']:.0f}]")

    # 边强度汇总：回答"**这张图的边整体强不强**"，不是"哪条是真边"
    n_strong = sum(1 for e in edges if e["strength"] == 2)
    n_mid = sum(1 for e in edges if e["strength"] == 1)
    n_weak = sum(1 for e in edges if e["strength"] == 0)
    _p(f"\n边强度汇总：强 {n_strong} / 中 {n_mid} / 弱 {n_weak}"
       f"（落差 ≥{concentric.STRONG_STEP:.0f} 灰阶为强、<{concentric.WEAK_STEP:.0f} 为弱）")
    if hidden:
        _p(f"另有 {hidden} 条中/弱边**未列**（低对比度下它们不稳定，加 --all-edges 可看）")
        warnings.append(f"另有 {hidden} 条弱边未列（不稳定）；见 --all-edges")
    if edges and n_strong == 0:
        worst = max(e["step"] for e in edges)
        _p(f"⚠️  本图没有强边（最大落差 {worst:.0f} 灰阶）——多半是低对比度/欠曝/离焦，先补光或重对焦。")
        _p("    强度只说明边在图像里显不显眼，**不代表它是不是你要量的尺寸**，别靠它判真假；")
        _p("    但整张图都弱时位置精度明显变差，这类读数不确定度大于 ±1.4%。")
        warnings.append("本图没有强边（整体对比度低）：先补光/对焦，读数不确定度更大")
    if len(detected) <= 1:
        _p("（只找到外缘：这件零件没有检出同心内边——实心光滑盘就是这样）")

    # ---- 成组孔（与同心边并列的另一条分支）----
    holes, holes_note, hole_warns = measure_holes(img, res, ratio)
    warnings.extend(hole_warns)
    _p(f"\n孔：{holes_note}")
    for h in holes:
        if h["dia_px"] > 0:
            mm = "" if ratio is None else f'  {h["dia_px"] * ratio:8.3f} mm'
            mark = "" if h["status"] == "OK" else f'  ⚠️ {h["status"]}'
            if h.get("boundary") == "bright_ring":
                support, tag = f'轮廓点 {h["n_points"]}', "  亮环"
            elif h.get("boundary") == "hole_inner":
                support, tag = f'点 {h["n_points"]}/{fit_ellipse_ring.N_RAY}', "  内缘"
            else:
                support, tag = f'点 {h["n_points"]}/{fit_ellipse_ring.N_RAY}', ""
            extra = ""
            if h.get("boundary") == "hole_inner" and h.get("dia_outer_px"):
                cm = "" if ratio is None else f"={h['dia_outer_px'] * ratio:.2f}mm"
                extra = f"  倒角口 {h['dia_outer_px']:.1f}px{cm}"
            _p(f'  H{h["idx"]:<3d} {h["dia_px"]:8.1f} px{mm}  '
               f'角{h["angle_deg"]:6.1f}°  d/R {h["r_norm"]:.3f}  '
               f'[{support}  票 {h["votes"]}  轴比 {h["aspect"]:.2f}{tag}{extra}]{mark}')
        else:
            co = (f'  倒角口 {h["dia_outer_px"]:.1f}px'
                  if h.get("dia_outer_px") else "")
            _p(f'  H{h["idx"]:<3d} 边界失败（点 {h["n_points"]}）  '
               f'角{h["angle_deg"]:6.1f}°  d/R {h["r_norm"]:.3f}{co}')
    dias = [h["dia_px"] for h in holes if h["dia_px"] > 0]
    if len(dias) >= 2:
        med = float(np.median(dias))
        spread = max(dias) - min(dias)
        med_mm = "" if ratio is None else f" = {med * ratio:.3f} mm"
        _p(f'  {len(dias)} 孔直径中位 {med:.1f} px{med_mm}，'
           f'极差 {spread:.1f}px（{100 * spread / med:.1f}%）')
    if holes:
        if any(h.get("boundary") == "bright_ring" for h in holes):
            _p("  提示：孔径取「亮环外轮廓」（螺钉头/沉孔口），不是通孔内壁；与暗斑孔的读数含义不同")
        elif any(h.get("boundary") == "hole_inner" for h in holes):
            _p("  提示：孔径取「螺孔内缘」（螺纹/孔壁 → 亮倒角的第一条上升沿），倒角口只作参考")
        else:
            _p("  提示：孔径取「梯度最大（steepest）」，与同心边同一套定义；孔和边一样有取边问题")

    if ratio is not None:
        _p(f"\n提示: 单像素量化误差 = {ratio:.3f} mm；"
           "读数的不确定度主要来自'外缘是一条有厚度的带'，实测约 ±1.4%")

    # ---- 标注图 + 报告落盘 ----
    keep_ids = {e["idx"] for e in edges}
    dbg = annotate.draw_listing(img, res, ratio,
                                label_filter=lambda e: e["idx"] in keep_ids,
                                holes=holes)
    DEBUG_DIR.mkdir(exist_ok=True)
    dbg_path = DEBUG_DIR / "measure_ring_debug.jpg"
    if not cv2.imwrite(str(dbg_path), dbg):     # 路径是 ASCII，但仍判返回值
        warnings.append("标注图写失败（debug/ 不可写？）")
    _p(f"标注图: {dbg_path}")

    rep_path = DEBUG_DIR / f"报告_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
    rep_path.write_text("\n".join(lines), encoding="utf-8")

    return RingResult(
        src_desc=src_desc, timestamp=timestamp, ratio=ratio, ratio_src=ratio_src,
        outer_px=outer_dia_px, outer_ellipse=outer, outer_mm=od_mm, edges=edges,
        hidden=hidden,
        holes=holes, holes_note=holes_note,
        outer_cue=cue, outer_cue_info=cue_info,
        thickness_mm=t_mm, thickness_src=t_src,
        ratio_base=hinfo.get("base"), height_factor=float(hinfo.get("factor", 1.0)),
        height_applied=bool(hinfo.get("applied")), height_note=hinfo.get("note", ""),
        tilt=tilt, touches_border=touches_border, warnings=warnings, lines=lines,
        debug_jpg=dbg_path, report_txt=rep_path, debug_img=dbg,
    )


def _rect_fail_message(img: np.ndarray, res: dict) -> str:
    """方形件定位失败的具体原因。别退回一句笼统的"检查对焦/光照"。"""
    fail = res.get("fail")
    if fail == "out_of_frame":
        return ("方形件超出画面：有边被画幅切掉了。\n"
                "把零件挪到画面中间，或把相机抬高/后拉，让四条边都完整入画再测。")
    if fail == "not_rect":
        return ("找到的最大轮廓不是矩形——可能是圆形件、或背景里的大块阴影。\n"
                "圆形件请用默认模式（--shape circle）；方形件请让四条边完整入画、"
                "背景尽量干净。")
    if fail == "too_small":
        return ("方形件在画面里太小（短边不足画幅的 8%）。\n"
                "请靠近或换更长焦的镜头，让零件至少占满画面的一半。")
    if fail == "bad_sides":
        bad = [s for s in (res.get("sides") or []) if not s.get("ok")]
        detail = "；".join(f"第{s['idx']}条 {s['n_inlier']}/{s['n']} 点"
                           for s in bad[:4])
        return (f"有 {len(bad)} 条边拟合失败（{detail}）。\n"
                "先补光、重新对焦，确保四条边都有足够对比度、且不被遮挡。")
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean, bright = float(g.mean()), float((g > 200).mean())
    if mean < 60 or bright < 0.005:
        return (f"没找到方形件的外轮廓：画面太暗（灰度均值 {mean:.0f}/255）。\n"
                "先补光或调高曝光再重拍。")
    if bright > 0.30:
        return (f"没找到方形件的外轮廓：画面过曝（亮像素 {bright * 100:.1f}%）。\n"
                "降低曝光或减少直射反光再重拍。")
    return ("没找到方形件的外轮廓。检查：对焦、光照、零件是否完整在画面内、"
            "背景是否干净（不要让背景里的大块阴影比零件更像矩形）。")


def measure_rect(img: np.ndarray, args) -> RectResult:
    """方形/矩形件：四条边亚像素 → 对边距/四边长/对角 → 换算 → 落盘。"""
    res = concentric.list_rect(img)
    if not res.get("ok"):
        raise MeasureError(_rect_fail_message(img, res))

    a_px = float(res["side_a_px"])
    b_px = float(res["side_b_px"])
    lines: list = []
    warnings: list = []

    def _p(s=""):
        lines.append(s)

    timestamp = datetime.now()
    src_desc = args.image or "相机现拍"
    t_mm, t_src, _ = resolve_thickness(args)
    ratio, ratio_src, hinfo = resolve_ratio(args, a_px, t_mm, what="长边")
    a_mm = None if ratio is None else a_px * ratio
    b_mm = None if ratio is None else b_px * ratio
    side_lengths = list(res["side_lengths"])
    side_lengths_mm = ([] if ratio is None
                       else [v * ratio for v in side_lengths])

    warnings.extend(res.get("warnings") or [])
    rectness = float(res.get("rectness") or 0.0)
    thresh = float(res.get("thresh") or 0.0)
    if 0.0 < rectness < 0.95:
        warnings.append(f"矩形度偏低（{rectness:.3f}）：轮廓可能带阴影/缺口/圆角过大")

    _p("\n===== 测量报告 =====")
    _p(f"时间: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}    图像: {src_desc}")
    _p("模式: 方形件（四条边；**不命名**——哪条边是哪个名义尺寸，"
       "由人对着标注图和图纸判断）")
    _p(f"轮廓: 阈值 {thresh:.0f}，矩形度 {rectness:.3f}"
       "（圆≈0.785、矩形≈0.99；低于 "
       f"{fit_ellipse_ring.RECT_RECTANGULARITY:.2f} 不会走方形模式）")

    if ratio is None:
        _p(f"换算系数: 无 —— {ratio_src}")
        _p("        没有系数就**只报像素、不给毫米**（拿别的机位的系数印毫米数是错的）")
        warnings.append("机位没标定过，本次只给像素；用 --ref-mm（方形=长边真值）标定后才有毫米")
    else:
        unc = load_rig().get("uncertainty", "")
        _p(f"换算系数: {ratio:.6f} mm/px （来源: {ratio_src}）"
           + (f"  不确定度 {unc}" if unc else ""))
        if getattr(args, "ref_mm", None) is not None:
            _p("        方形模式的 --ref-mm 指**长边真值**（对边距 A），不是短边，也不是对角线")

    if t_mm is not None:
        _p(f"厚度: {t_mm:.2f} mm（来源: {t_src}）")
    if hinfo["applied"] and ratio is not None:
        base = float(hinfo["base"])
        pct = (float(hinfo["factor"]) - 1.0) * 100.0
        _p(f"高度补偿: 基准 {float(hinfo['t_ref']):.2f} mm，"
           f"{float(hinfo['scale']) * 100:+.3f}%/mm → "
           f"{base:.6f} → {ratio:.6f} mm/px（{pct:+.2f}%）")
        _p("    假定零件直接坐在同一基准台面上（有垫块时量的是顶面高度）；"
           "残余约 ±0.17% 是取边带，不是厚度误差")
    elif hinfo["enabled"] and hinfo["note"]:
        _p(f"高度补偿: 未做 —— {hinfo['note']}")
        warnings.append(hinfo["note"])

    _p(f"\n检出 4 条边（按轮廓顺时针编号，不命名）：")
    for s in res["sides"]:
        L = float(s.get("length_px") or s["coarse_len_px"])
        mm = "" if ratio is None else f"  {L * ratio:8.3f} mm"
        _p(f"  S{s['idx']:<2d} {L:9.1f} px{mm}  "
           f"角 {s['angle_deg']:+6.2f}°  点 {s['n_inlier']}/{s['n']}  "
           f"残差 {s['resid_px']:.2f}px  落差 {s['step']:.0f}  梯度 {s['grad_med']:.1f}")

    _p(f"\n对边距 A（长边）: {a_px:.2f} px"
       + ("" if a_mm is None else f"  = {a_mm:.3f} mm"))
    _p(f"对边距 B（短边）: {b_px:.2f} px"
       + ("" if b_mm is None else f"  = {b_mm:.3f} mm"))
    if side_lengths_mm:
        pairs = "  ".join(f"S{i + 1}={v:.3f}" for i, v in enumerate(side_lengths_mm))
        _p(f"四条边长（直边延长到虚拟角点）: "
           + "  ".join(f"S{i + 1}={v:.2f}px" for i, v in enumerate(side_lengths))
           + f"\n                    = {pairs} mm")
    else:
        _p("四条边长（直边延长到虚拟角点）: "
           + "  ".join(f"S{i + 1}={v:.2f}px" for i, v in enumerate(side_lengths)))
    _p(f"对角线: {res['diagonals'][0]:.2f} / {res['diagonals'][1]:.2f} px，"
       f"差 {res['diag_diff_px']:.2f}px（{res['diag_diff_pct']:.2f}%）")
    _p(f"平行偏差: A {res['parallel_dev'][0]:.3f}° / B {res['parallel_dev'][1]:.3f}°"
       f"    垂直偏差: 最大 {max(abs(v) for v in res['perp_dev']):.3f}°")
    if abs(res["aspect"] - 1.0) <= 0.02:
        _p(f"对边距比 A/B: {res['aspect']:.4f}（接近 1 → 方形件）")
    else:
        _p(f"对边距比 A/B: {res['aspect']:.4f}（矩形件；这是长宽比，不是缺陷）")

    if warnings:
        _p("\n⚠️  告警：")
        for w in warnings:
            _p(f"  - {w}")
    if ratio is not None:
        _p(f"\n提示: 单像素量化误差 = {ratio:.3f} mm；方形件的边同样是"
           "'有厚度的带'（倒角/圆角），跨零件比较的精度天花板仍是 ±1.4%；"
           "同一件重复性要好得多")

    # ---- 标注图 + 报告落盘 ----
    dbg = annotate.draw_rect(img, res, ratio)
    DEBUG_DIR.mkdir(exist_ok=True)
    dbg_path = DEBUG_DIR / "measure_rect_debug.jpg"
    ok, buf = cv2.imencode(".jpg", dbg)
    if ok:
        try:
            buf.tofile(str(dbg_path))       # 中文路径安全；imwrite 会静默失败
        except OSError:
            warnings.append("标注图写失败（debug/ 不可写？）")
    else:
        warnings.append("标注图编码失败")
    _p(f"标注图: {dbg_path}")

    rep_path = DEBUG_DIR / f"报告_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
    rep_path.write_text("\n".join(lines), encoding="utf-8")

    return RectResult(
        src_desc=src_desc, timestamp=timestamp, ratio=ratio, ratio_src=ratio_src,
        side_a_px=a_px, side_b_px=b_px, side_a_mm=a_mm, side_b_mm=b_mm,
        side_lengths=side_lengths, side_lengths_mm=side_lengths_mm,
        diagonals=list(res["diagonals"]), diag_diff_px=float(res["diag_diff_px"]),
        diag_diff_pct=float(res["diag_diff_pct"]), aspect=float(res["aspect"]),
        angle=float(res["angle"]), parallel_dev=list(res["parallel_dev"]),
        perp_dev=list(res["perp_dev"]), sides=list(res["sides"]),
        center=tuple(res["center"]), thresh=thresh, rectness=rectness,
        thickness_mm=t_mm, thickness_src=t_src,
        ratio_base=hinfo.get("base"), height_factor=float(hinfo.get("factor", 1.0)),
        height_applied=bool(hinfo.get("applied")), height_note=hinfo.get("note", ""),
        warnings=warnings, lines=lines,
        debug_jpg=dbg_path, report_txt=rep_path, debug_img=dbg,
    )


def main():
    args = build_parser().parse_args()
    # 参数放错模式时不静默忽略：它们都会写档案/改变读数，静默最危险。
    if args.mode == "thickness" and args.ref_mm is not None:
        raise SystemExit("--ref-mm 是顶视机位标定，不能和 --mode thickness 一起用")
    if args.mode == "thickness" and args.thickness is not None:
        raise SystemExit("--thickness 用于外径测量的高度补偿，不用于 --mode thickness")
    if args.mode != "thickness" and args.side_ref_mm is not None:
        raise SystemExit("--side-ref-mm 只能配合 --mode thickness 使用")
    if args.shape == "rect" and args.mode == "thickness":
        raise SystemExit("--shape rect 不能和 --mode thickness 一起用"
                         "（厚度是同一台相机换侧视镜头的另一条测量）")
    if args.shape == "rect" and args.all_edges:
        raise SystemExit("--all-edges 只用于圆形模式（方形件固定就是那四条边）")
    try:
        if args.mode == "thickness":
            run_thickness_mode(args)
            return
        img = grab_image(args)
        result = measure_rect(img, args) if args.shape == "rect" else measure(img, args)
    except MeasureError as e:
        raise SystemExit(str(e)) from None
    for ln in result.lines:
        print(ln)


if __name__ == "__main__":
    main()
