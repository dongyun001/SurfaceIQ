r"""圆环/法兰类零件尺寸测量：外径、中心孔、螺栓孔，一键出毫米数。

**型号档案是唯一入口**（models/part_profiles.json）：每个零件型号一份档案，写清它的标称值
和该走哪条检测策略。传 --profile <型号名> 人工指定；传 --profile none 明确按“无档案”试测。
不传 --profile 时**不再猜测**：本次只给像素、结果标未验证，并提示人工选型号
（型号识别已于 2026-10-23 删除，理由见文件末尾）。

三种用法：
    1) 图形界面（日常用，推荐）：双击项目里的 `测量.bat`
       - 相机已接：自动拍照 → 弹出结果窗口，大字显示各尺寸（见 gui_ring.py）
       - 想测已有照片：把照片文件拖到 `测量.bat` 图标上
    2) 命令行（标定 / 排错用）：项目根目录 vision-roughness/ 下
       python src\measure_ring.py --image data\captures\xxx.jpg
       python src\measure_ring.py --profile <型号名> --ref-mm 88.60   # 标定该型号的系数

每次测量会在 debug\ 下存两份记录：measure_ring_debug.jpg（画出轮廓的图）
和 报告_年月日_时分秒.txt（文字报告）。

原理：
    “毫米/像素”是「相机+镜头+拍摄距离」的函数，而不同型号往往要换机位，所以**每个型号
    各存一份系数**，标定时用卡尺量出该型号的外径真值（--profile + --ref-mm，会写回档案）。
    ⚠️ 相机动了/零件高度变了/换镜头了，必须重新标定该型号。
    没有本型号的系数时只输出像素——拿别的型号的系数印毫米数是错的，所以不做。

检测管线（fit_ellipse_ring.detect）：
    外圈   : 旧先验优先，不行则多候选(圆心×径向峰半径×两种极性)打分选优，自适应定位
    中心孔 : 多比例带 + 两种极性，由“在外圈内且与外圈同心”判可信
    螺栓孔 : locate(怎么定位孔) + boundary(量哪条边界) 两个策略维度，型号档案里各指定一个
             locate   = profile_angles(档案角度+相位搜索) | template_ngon(模板匹配+n等分校验)
             boundary = countersink_rim(沉孔口阈值交叉)    | dark_core(暗芯中分界)
⚠️ 型号识别（极坐标指纹 + 孔型判据）已于 2026-10-23 整体删除：它是缺陷 2/5 的来源，
    而回归门本来就要求显式 --profile，两条用例上一次都没用上它。
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
import config
from camera_adapter import create_camera
from fit_ellipse_ring import DEFAULT_BOUNDARY, DEFAULT_LOCATE, detect

PROFILE_PATH = Path("models/part_profiles.json")
NO_PROFILE = ("none", "off", "无")  # --profile 显式要求“没有档案”时的取值
DEBUG_DIR = Path("debug")
BORDER_MARGIN = 5  # 轮廓距图像边缘小于此值视为“零件超出视野”

# 型号是怎么定下来的（界面据此提示，别用字符串去猜）。GUI 读 RingResult.ident_source。
IDENT_EXPLICIT = "explicit"          # 命令行/界面显式指定了型号
IDENT_NOPROFILE = "noprofile"        # 显式要求按“无档案”试测
IDENT_UNIDENTIFIED = "unidentified"  # 没指定型号 -> 只给像素、请人工选型号

# 螺栓孔位置的来源标记：推定的位置必须让操作员看得出来，不能和实测的长得一样
SRC_LABEL = {"detected": "实测峰", "profile": "档案角度", "synthesized": "位置推定"}
BOUNDARY_HINT = {
    "countersink_rim": "螺栓孔按沉孔口(norm<0.72 阈值交叉)定义；已用游程宽度过滤+渐缩残差处理暗带/槽缘粘连",
    "dark_core": "螺栓孔按紧固件中间的暗芯通孔定义（射线中分界 + 中位半径）",
}



def load_profiles():
    """读整份型号档案。识别型号要遍历所有型号，所以整份读进来。"""
    if not PROFILE_PATH.exists():
        raise MeasureError(f"找不到型号档案: {PROFILE_PATH}")
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def named_profile(profiles, name):
    """按名字取一份档案（命令行显式指定型号时用）。"""
    if name not in profiles:
        have = "、".join(k for k in profiles if not k.startswith("_"))
        raise MeasureError(f"型号档案里没有「{name}」。现有型号：{have}")
    prof = dict(profiles[name])
    prof["_name"] = name
    return prof


def save_profile_mm_per_px(prof, ratio):
    """把本次标定出的 mm/px 写回该型号档案（只有同时给了 --profile 和 --ref-mm 才调）。"""
    profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    profiles[prof["_name"]]["mm_per_px"] = ratio
    PROFILE_PATH.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")


# 型号识别与指纹存取已于 2026-10-23 整体删除（`polar_fingerprint` / `match_fingerprint` /
# `identify_by_bolts` / `--save-fingerprint` / `models/fingerprints/`）。理由：它是缺陷 2/5 的
# 来源，而回归门本来就要求显式 `--profile`，自动识别在两条用例上一次都没用上。


class MeasureError(Exception):
    """可以预料、需要原样提示给用户看的测量失败（不是程序缺陷）。"""


@dataclass
class RingResult:
    """一次测量的全部结果：数字 + 文本报告 + 落盘路径。

    供命令行直接打印，也供 gui_ring.py 大字显示——两条路走同一份数据，
    不会出现"窗口一个数、报告另一个数"。
    """
    src_desc: str
    timestamp: datetime
    ratio: Optional[float]          # None = 缺本型号的换算系数，只给像素
    ratio_src: str
    outer_mm: Optional[float]
    outer_px: float
    outer_ellipse: tuple
    center_hole_mm: Optional[float] = None
    center_hole_px: Optional[float] = None
    center_hole_ellipse: Optional[tuple] = None
    inner_rings: list = field(default_factory=list)  # [{name,mm,px,status,mm_exp}]
    bolts: list = field(default_factory=list)      # [{mm,px,a1,a2,cx,cy,status,ok}]
    tilt: float = 1.0
    touches_border: bool = False
    overall: str = "OK"
    overall_judged: bool = False                   # 没给标称值时无 OK/NG 可言
    warnings: list = field(default_factory=list)   # 给 GUI 的通俗提醒
    lines: list = field(default_factory=list)      # 文本报告（与命令行输出一致）
    debug_jpg: Optional[Path] = None
    report_txt: Optional[Path] = None
    debug_img: Optional[np.ndarray] = None         # 画了轮廓的图（GUI 缩略图用）
    profile_name: Optional[str] = None             # 本次用的型号档案名
    locate: Optional[str] = None                   # 实际用到的定位/边界策略名
    boundary: Optional[str] = None
    unverified: bool = False                       # 是否走了兜底（缺档案）
    center_hole_expected: bool = True              # 本型号是否本该有中心孔（"没有"≠"没测到"）
    identified: str = ""                           # 型号是怎么定下来的（给人看的一句话）
    ident_source: str = IDENT_UNIDENTIFIED         # 见 IDENT_*：界面据此决定是否提示选型号


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。GUI 复用同一套，保证拖照片/标定两种用法参数一致。"""
    p = argparse.ArgumentParser(description="圆环类零件尺寸测量")
    p.add_argument("--image", help="测已有照片；不指定则相机现拍")
    p.add_argument("--profile", default=None,
                   help="零件型号名（见 models/part_profiles.json）；不传就只给像素并提示人工选型号；"
                        "传 none 明确按“无档案”试测")
    p.add_argument("--ref-mm", type=float,
                   help="零件外径真值(mm)，用于标定该型号的换算系数（必须配 --profile）")
    p.add_argument("--mm-per-px", type=float, help="直接指定 mm/px")
    p.add_argument("--spec-od", type=float, help="外径标称值(mm)，用于 OK/NG")
    p.add_argument("--spec-id", type=float, help="中心孔标称值(mm)，用于 OK/NG")
    p.add_argument("--spec-bolt", type=float, help="螺栓孔标称值(mm)，用于 OK/NG")
    p.add_argument("--tol", type=float, default=0.1, help="公差 ±mm，默认 0.1")
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


def resolve_ratio(args, outer_dia_px, profile=None):
    """确定 mm/px，返回 (ratio 或 None, 来源说明)。

    唯一真相源是**型号档案**：mm/px 是“相机+镜头+拍摄距离”的函数，而不同型号往往要换
    机位，所以每型号各存一份（实测型号2 是型号1 的 1.48 倍，不能跨型号复用）。
    以前那个全局缓存 models/mm_per_pixel.json 已退休——它是同一个数的第二份真相源，
    而且属于“上次标定的某个型号”，拿去给另一个型号印毫米数就是拿错的系数印数。

    拿不到系数就返回 None：调用方据此**只报像素、不给毫米**，而不是编一个数。
    """
    if args.mm_per_px:
        return args.mm_per_px, "命令行指定"
    if args.ref_mm:
        # 必须显式 --profile：标定是型号相关的重动作，绝不能因为“某个型号恰好是默认型号”
        # 就写进那个型号的档案里（--ref-mm 单独跑过去会把默认型号的系数悄悄改掉）。
        if not profile or not profile.get("_explicit"):
            raise MeasureError(
                "用 --ref-mm 标定必须**显式**指明型号：\n"
                "    python src\measure_ring.py --profile <型号名> --ref-mm <外径真值mm>\n"
                "    （不写 --profile 会落到默认型号，那会把它的系数悄悄改掉；\n"
                "      换型号必须各自标定：不同型号往往换机位，系数不能跨型号复用）"
            )
        ratio = args.ref_mm / outer_dia_px
        save_profile_mm_per_px(profile, ratio)
        return ratio, f"本次标定（外径={args.ref_mm}mm），已写回型号档案「{profile['_name']}」"
    if profile and profile.get("mm_per_px"):
        return float(profile["mm_per_px"]), f"型号档案「{profile['_name']}」"
    if profile:
        return None, f"型号档案「{profile['_name']}」还没标定过 mm/px"
    return None, "缺档案：没有本型号的换算系数"



def locate_fail_hint(img: np.ndarray) -> str:
    """外圈定位失败时，尽量说清到底是"太暗/过曝"还是别的原因。

    以前一律报「外圈拟合失败。检查：对焦、光照、零件是否在画面中心」——实测现场
    多次失败其实是**光不足**（标定图均值 110、亮像素 5.2%；那批失败图只有 15~31、
    亮像素 0.0%），操作员光看这句话不知道该去补光。

    门槛留了很大余量：均值 60、亮像素 0.5% / 30%。
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean, bright = float(g.mean()), float((g > 200).mean())
    if mean < 60 or bright < 0.005:
        return (f"外圈拟合失败：画面太暗（灰度均值 {mean:.0f}/255，亮像素 {bright * 100:.1f}%）。\n"
                "先补光或把曝光调高再重拍。参考：能稳定测出的标定图均值 110、亮像素 5%。")
    if bright > 0.30:
        return (f"外圈拟合失败：画面过曝（亮像素 {bright * 100:.1f}%）。\n"
                "把曝光调低或减少直射反光再重拍。参考：能稳定测出的标定图亮像素 5%。")
    return "外圈拟合失败。检查：对焦、光照、零件是否在画面中心"


def _outer_fail_message(img: np.ndarray, reason) -> str:
    """外圈定位失败的具体原因。

    缺陷 8：零件出画时，外圈会被画幅切掉，自适应搜索会退而锁到内孔（它是个完美圆，
    三条判据全过）。`find_outer` 现在会把这种情况标成 out_of_frame，这里必须说清，
    否则操作员会拿"内孔直径"当外径。
    """
    if reason == "out_of_frame":
        return ("外圈超出画面：检出的是内孔（更外面的圆被画幅切掉了）。\n"
                "把零件移入画面中心，或把相机抬高/后拉，让整圈外缘都在画面内再测。")
    return locate_fail_hint(img)


def measure(img: np.ndarray, args) -> RingResult:
    """检测 → 换算 → 判定 → 落盘（调试图 + 文字报告），返回结构化结果。

    失败的测量（含贴边等）不一定抛异常，但提示会进 warnings/lines。
    """
    profiles = load_profiles()
    explicit = getattr(args, "profile", None)
    profile, ident = None, ""
    ident_source = IDENT_UNIDENTIFIED
    if explicit and explicit.lower() in NO_PROFILE:
        ident = "命令行/界面要求按「无档案」处理：通用试测，结果未验证"
        ident_source = IDENT_NOPROFILE
    elif explicit:
        profile = named_profile(profiles, explicit)
        profile["_explicit"] = True          # 只有显式指定的型号才允许 --ref-mm 写回
        ident = f"命令行/界面指定型号「{profile['_name']}」"
        ident_source = IDENT_EXPLICIT
    else:
        # 没有指定型号就**不猜**（2026-10-23 删掉自动识别之后只剩这一条路）：
        # 本次只给像素、结果标未验证，让操作员在界面里选或命令行 --profile。
        # 绝不"按某个型号测"——那会拿错型号的标称值和 mm/px 给出一个理直气壮的错读数。
        ident = "没有指定型号（自动识别已删除）→ 请人工选择型号；本次只给像素"
    # --mm-per-px 是"本次测量用这个系数"，检测里用到系数的路径（如 dark_core 的搜索带）
    # 也必须跟着走，否则报告和检测会用两套比例。回归门靠它把系数钉死。
    if args.mm_per_px and profile:
        profile = dict(profile)
        profile["mm_per_px"] = float(args.mm_per_px)
    res = detect(img, profile=profile)
    if res["outer"] is None:
        raise MeasureError(_outer_fail_message(img, res.get("outer_fail")))

    lines: list = []
    warnings: list = []

    def _p(s=""):
        lines.append(s)

    timestamp = datetime.now()
    src_desc = args.image or "相机现拍"

    (outer, n_o) = res["outer"]
    (ocx, ocy), (oa1, oa2), oang = outer
    outer_dia_px = (oa1 + oa2) / 2
    ratio, ratio_src = resolve_ratio(args, outer_dia_px, profile)
    unverified = res["bolts_unverified"]
    loc, bnd = res["bolts_locate"], res["bolts_boundary"]

    def _size_line(label, v_mm, v_px, a1, a2, ang):
        tail = f"椭圆 {a1:.1f}x{a2:.1f}, 角 {ang:.1f}°"
        if v_mm is None:
            return f"{label} {v_px:.1f} px  ({tail})"
        return f"{label} {v_mm:.3f} mm  ({v_px:.1f} px, {tail})"

    touches_border = (oa1 > img.shape[1] - BORDER_MARGIN * 2 or
                      oa2 > img.shape[0] - BORDER_MARGIN * 2)
    if touches_border:
        _p("⚠️  警告：零件轮廓贴到图像边缘，零件没拍全，测量结果不可信！")
        _p("    把相机架高一点（25mm 镜头：视野宽 ≈ 距离 × 0.3）再重拍。")
        warnings.append("零件超出画面：轮廓贴到图像边缘，结果不可信，请把相机架高重拍")

    od_mm = None if ratio is None else outer_dia_px * ratio
    _p("\n===== 测量报告 =====")
    _p(f"时间: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}    图像: {src_desc}")
    _p(f"型号: {profile['_name'] if profile else '未确定（无档案）'}    {ident}")
    _p(f"      定位={loc} 边界={bnd}")
    if ident_source == IDENT_UNIDENTIFIED:
        _p("⚠️  **未识别到型号**：没有本型号的标称值与换算系数，本次**只给像素、不给毫米**。")
        _p("    请人工确认零件型号后重测：界面上在「型号」里选，或命令行 --profile <型号名>。")
        warnings.append("未识别到型号，本次只给像素；请在界面「型号」里选或命令行 --profile")
    elif unverified:
        _p("⚠️  型号未指定：按通用方式试测，结果**未经验证**（没有标称值可核对）")
        warnings.append("型号未指定，结果未经验证")
    if ratio is None:
        _p(f"换算系数: 无 —— {ratio_src}")
        _p("        没有本型号的系数就**只报像素、不给毫米**"
           "（拿别的型号的系数印毫米数是错的）")
        warnings.append("没有本型号的换算系数，本次只给像素值")
    else:
        _p(f"换算系数: {ratio:.6f} mm/px （来源: {ratio_src}）")
    _p(_size_line("外径:  ", od_mm, outer_dia_px, oa1, oa2, oang))
    if res.get("outer_src") == "bolts":
        _p("        （外缘对比度低：圆心/搜索半径由紧固件锚定，外径仍为实测）")
        warnings.append("外圈由紧固件锚定（外缘对比度低）")

    tilt = max(oa1, oa2) / min(oa1, oa2)
    if tilt > 1.02:
        tilt_deg = np.degrees(np.arccos(min(oa1, oa2) / max(oa1, oa2)))
        _p(f"⚠️  透视提醒：外圈椭圆长短轴比 {tilt:.3f}（相当于斜视 {tilt_deg:.0f}°），"
           "各尺寸的毫米读数带位置相关误差；把相机垂直朝下正对零件可消除")
        warnings.append(f"相机斜视约 {tilt_deg:.0f}°（外圈椭圆比 {tilt:.3f}）：各读数带位置相关误差，"
                        "把相机垂直朝下可消除")

    overall = "OK"
    overall_judged = False
    if args.spec_od:
        overall_judged = True
        dev = od_mm - args.spec_od
        ok = abs(dev) <= args.tol
        overall = "OK" if ok else "NG"
        _p(f"        标称 {args.spec_od} ±{args.tol}  偏差 {dev:+.3f}  -> {'OK' if ok else 'NG'}")

    # ---- 中心孔 ----
    no_center_hole = profile is not None and profile.get("has_center_hole") is False
    if res["center_hole"] is None or res["center_hole"][0] is None:
        hole = None
        hole_mm = hole_px = None
        if no_center_hole:
            _p("中心孔: 本型号无中心孔（型号档案 has_center_hole=false），不计入判定")
        else:
            _p("中心孔: 未检测到（若零件有孔，检查孔内是否反光/遮挡）")
            warnings.append("中心孔没测到：检查孔内是否反光或被遮挡")
    else:
        (hole, n_h) = res["center_hole"]
        (hcx, hcy), (ha1, ha2), hang = hole
        hole_px = (ha1 + ha2) / 2
        hole_mm = None if ratio is None else hole_px * ratio
        _p(_size_line("中心孔:", hole_mm, hole_px, ha1, ha2, hang))
        if args.spec_id and hole_mm is not None:
            overall_judged = True
            dev = hole_mm - args.spec_id
            ok = abs(dev) <= args.tol
            if not ok:
                overall = "NG"
            _p(f"        标称 {args.spec_id} ±{args.tol}  偏差 {dev:+.3f}  -> {'OK' if ok else 'NG'}")

    # ---- 内圈（档案声明的；每圈取两道边的中点）----
    inner_out = []
    for it in res.get("inner_rings", []):
        if it.get("status") != "OK" or not it.get("dia"):
            _p(f"{it['name']}: 未测到（期望 {it['mm_exp']:.2f} mm）")
            warnings.append(f"{it['name']} 没测到")
            inner_out.append({"name": it["name"], "mm": None, "px": None,
                              "status": "失败", "mm_exp": it["mm_exp"]})
            continue
        it_mm = None if ratio is None else it["dia"] * ratio
        if it_mm is None:
            _p(f"{it['name']}: {it['dia']:.1f} px（无换算系数）")
        else:
            _p(f"{it['name']}: {it_mm:.3f} mm  ({it['dia']:.1f} px，"
               f"期望 {it['mm_exp']:.2f} 偏差 {it_mm - it['mm_exp']:+.3f})")
        inner_out.append({"name": it["name"], "mm": it_mm, "px": it["dia"],
                          "status": "OK", "mm_exp": it["mm_exp"]})

    # ---- 螺栓孔 ----
    bolts_out = []
    if res["bolts"]:
        for i, b in enumerate(res["bolts"], 1):
            src = b.get("src", "detected")
            tag = "" if src == "detected" else f"  ← {SRC_LABEL.get(src, src)}"
            if b["status"] != "OK":
                _p(f"螺栓孔{i}: 直径测量失败 @({b['cx']:.0f},{b['cy']:.0f}){tag}")
                warnings.append(f"螺栓孔{i} 直径测量失败")
                bolts_out.append({"idx": i, "mm": None, "px": None, "ok": None,
                                  "status": b["status"], "cx": b["cx"], "cy": b["cy"],
                                  "src": src})
                continue
            b_mm = None if ratio is None else b["dia"] * ratio
            tail = (f"椭圆 {b['a1']:.1f}x{b['a2']:.1f}, 位置({b['cx']:.0f},{b['cy']:.0f})")
            if b_mm is None:
                _p(f"螺栓孔{i}: {b['dia']:.1f} px  ({tail}){tag}")
            else:
                _p(f"螺栓孔{i}: {b_mm:.3f} mm  ({b['dia']:.1f} px, {tail}){tag}")
            b_ok = None
            if args.spec_bolt and b_mm is not None:
                overall_judged = True
                dev = b_mm - args.spec_bolt
                b_ok = abs(dev) <= args.tol
                if not b_ok:
                    overall = "NG"
                _p(f"        标称 {args.spec_bolt} ±{args.tol}  偏差 {dev:+.3f}  -> {'OK' if b_ok else 'NG'}")
            bolts_out.append({"idx": i, "mm": b_mm, "px": b["dia"], "ok": b_ok,
                              "status": b["status"], "a1": b["a1"], "a2": b["a2"],
                              "cx": b["cx"], "cy": b["cy"], "ang": b["ang"], "src": src})
        if any(b.get("src") == "synthesized" for b in res["bolts"]):
            _p("        注：标了「位置推定」的孔是按等角假设推出来的位置，不是实测到的峰")
        _p("提示: " + BOUNDARY_HINT.get(bnd, bnd))
    else:
        _p("未检测到螺栓孔（定位已尝试，但没找到可信的孔）")
        warnings.append("没找到螺栓孔")

    _p(f"综合判定: {overall}")
    if ratio is None:
        _p("提示: 本次没有换算系数，只输出像素；标定后再测才能给毫米读数")
    else:
        _p(f"提示: 本测量为像素级拟合，单像素量化误差 = {ratio:.3f} mm；"
           "要冲微米级需远心镜头+背光+亚像素（见方案讨论）")

    # ---- 调试图：画出检测到的椭圆 ----
    dbg = img.copy()
    cv2.ellipse(dbg, outer, (0, 255, 0), 8)
    if hole is not None:
        cv2.ellipse(dbg, hole, (0, 0, 255), 8)
    for it in res.get("inner_rings", []):
        if it.get("status") == "OK" and it.get("ell") is not None:
            cv2.ellipse(dbg, it["ell"], (255, 0, 255), 6)
    for b in res["bolts"]:
        if b["status"] == "OK":
            cv2.ellipse(dbg, ((b["cx"], b["cy"]), (b["a1"], b["a2"]), b["ang"]), (255, 0, 0), 6)
    DEBUG_DIR.mkdir(exist_ok=True)
    dbg_path = DEBUG_DIR / "measure_ring_debug.jpg"
    cv2.imwrite(str(dbg_path), dbg)
    _p(f"调试图: {dbg_path}")

    # 报告存盘（留痕/追溯）
    rep_path = DEBUG_DIR / f"报告_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
    rep_path.write_text("\n".join(lines), encoding="utf-8")

    return RingResult(
        src_desc=src_desc, timestamp=timestamp, ratio=ratio, ratio_src=ratio_src,
        outer_mm=od_mm, outer_px=outer_dia_px, outer_ellipse=outer,
        center_hole_mm=hole_mm, center_hole_px=hole_px, center_hole_ellipse=hole,
        inner_rings=inner_out, bolts=bolts_out, tilt=tilt, touches_border=touches_border,
        overall=overall, overall_judged=overall_judged, warnings=warnings,
        lines=lines, debug_jpg=dbg_path, report_txt=rep_path, debug_img=dbg,
        profile_name=(profile["_name"] if profile else None),
        locate=loc, boundary=bnd, unverified=unverified,
        center_hole_expected=not no_center_hole, identified=ident,
        ident_source=ident_source,
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
    print(f"报告已存: {result.report_txt}")


if __name__ == "__main__":
    main()
