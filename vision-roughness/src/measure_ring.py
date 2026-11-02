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
    2) 命令行（标定 / 排错用）：项目根目录 vision-roughness/ 下
       python src\measure_ring.py --image data\captures\xxx.jpg
       python src\measure_ring.py --image <该机位的图> --ref-mm 88.28   # 标定机位系数

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
    tilt: float = 1.0
    touches_border: bool = False
    warnings: list = field(default_factory=list)
    lines: list = field(default_factory=list)      # 文本报告（与命令行输出一致）
    debug_jpg: Optional[Path] = None
    report_txt: Optional[Path] = None
    debug_img: Optional[np.ndarray] = None         # 标注图（GUI 缩略图用）


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。GUI 复用同一套，保证拖照片/标定两种用法参数一致。"""
    p = argparse.ArgumentParser(description="零件同心边尺寸测量（不命名、不判合格）")
    p.add_argument("--image", help="测已有照片；不指定则相机现拍")
    p.add_argument("--ref-mm", type=float,
                   help="零件外径卡尺真值(mm)，用于标定**机位**的换算系数（会写回 models/rig.json）")
    p.add_argument("--mm-per-px", type=float, help="直接指定 mm/px（临时用，不写回档案）")
    p.add_argument("--all-edges", action="store_true",
                   help="连'弱边'一起列。默认只列强边（+外缘）——弱边在低对比度件上**不稳定**"
                        "（实测连拍 5 帧条数在 11~15 之间跳、直径极差 7~11%%），列出来会误导人")
    return p


def grab_image(args) -> np.ndarray:
    """从相机拍一张，或从文件读一张，返回 BGR 图像。"""
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise MeasureError(f"无法读取图片: {args.image}")
        return img
    print("打开相机拍照...")
    with create_camera(config.CAMERA_TYPE, **config.CAMERA_KWARGS) as cam:
        frame = cam.capture()
    out = Path("data/captures") / "measure_ring_latest.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"已保存: {out}")
    return frame


def resolve_ratio(args, outer_dia_px):
    """确定 mm/px，返回 (ratio 或 None, 来源说明)。

    唯一真相源是**机位档案** models/rig.json（全局一份）。拿不到就返回 None：
    调用方据此只报像素、不给毫米，而不是编一个数。
    """
    if args.mm_per_px:
        return float(args.mm_per_px), "命令行指定（临时，不写回档案）"
    if args.ref_mm:
        ratio = float(args.ref_mm) / outer_dia_px
        src = (f"本次标定：外径真值 {args.ref_mm}mm / 检出 {outer_dia_px:.1f}px"
               f"（图像 {args.image or '相机现拍'}）")
        save_rig(ratio, src)
        return ratio, src + "，已写回机位档案"
    rig = load_rig()
    if rig.get("mm_per_px"):
        when = rig.get("calibrated_at", "未知时间")
        return float(rig["mm_per_px"]), f"机位档案（{when} 标定）"
    return None, "机位没标定过（models/rig.json 里没有 mm_per_px）"


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
        # 只作采样上限；逐射线的边界限制由 fit_hole_boundary 按方向算
        # （孔常常紧挨环形槽，全局取"最近边"会把所有射线都砍掉）
        max_r = min(2.6 * c["r"], 0.5 * r_outer)
        ell, n, contrast = fit_ellipse_ring.fit_hole_boundary(
            sm, c["cx"], c["cy"], c["r"], max_r,
            edge_radii=edge_r, part_center=(ocx, ocy))
        h = {"idx": i, "votes": int(c["votes"]), "circularity": float(c["circ"]),
             "contrast": float(contrast), "boundary": "dark_blob"}
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
                      "status": ("OK" if (n >= fit_ellipse_ring.HOLE_LOW_POINTS
                                          and c["votes"] >= 4) else "低置信度")})
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
    ratio, ratio_src = resolve_ratio(args, outer_dia_px)
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
            else:
                support, tag = f'点 {h["n_points"]}/{fit_ellipse_ring.N_RAY}', ""
            _p(f'  H{h["idx"]:<3d} {h["dia_px"]:8.1f} px{mm}  '
               f'角{h["angle_deg"]:6.1f}°  d/R {h["r_norm"]:.3f}  '
               f'[{support}  票 {h["votes"]}  轴比 {h["aspect"]:.2f}{tag}]{mark}')
        else:
            _p(f'  H{h["idx"]:<3d} 边界失败（点 {h["n_points"]}）  '
               f'角{h["angle_deg"]:6.1f}°  d/R {h["r_norm"]:.3f}')
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
        tilt=tilt, touches_border=touches_border, warnings=warnings, lines=lines,
        debug_jpg=dbg_path, report_txt=rep_path, debug_img=dbg,
    )


def main():
    args = build_parser().parse_args()
    try:
        img = grab_image(args)
        result = measure(img, args)
    except MeasureError as e:
        raise SystemExit(str(e)) from None
    for ln in result.lines:
        print(ln)


if __name__ == "__main__":
    main()
