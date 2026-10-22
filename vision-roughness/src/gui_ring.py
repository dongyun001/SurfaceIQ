r"""圆环尺寸测量窗口（tkinter，日常双击 测量.bat 就是这个）。

为什么有这个文件：车间里没人愿意读命令行里的一堆文字。窗口一屏给全：
    左列 —— 外径 / 中心孔 / 6 个螺栓孔的大字读数 + OK/NG 状态；
    右列 —— **实时取景画面**（相机连续取帧，实测 12MP ≈ 7.7 fps），
             测完之后画面冻结在检测轮廓图上（绿=外圈 / 红=中心孔 / 蓝=螺栓孔）。

设计要点：
    - 数字与报告都来自 measure_ring.measure() 的同一份结果，不会"窗口一个数、
      报告另一个数"；
    - **相机是独占资源**：取帧与测量只在同一个后台线程里做，绝不并发 capture()；
      tkinter 只能在主线程碰，所以后台线程只往队列塞「编码好的 PNG 字节」和事件，
      主线程负责变成 tk.PhotoImage 画到画布上；
    - **启动即自动测一次**（车间双击就该直接出结果），先给约 1.2 秒实时画面让你看清
      零件位置，再自动测；之后 空格/F5 再测、P 暂停取景；
    - 用 pythonw 启动（无控制台），所以所有异常都必须翻译成中文显示在窗口里，
      并写日志，绝不能"双击了没反应"；
    - 沿用 测量.bat 的参数习惯：--image 测照片、不传则相机现拍。

按键：空格 / F5 = 测一次    P = 暂停 / 继续取景    Esc = 关闭
"""
import base64
import os
import queue
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

# 项目根目录 = 本文件所在 src/ 的上一级；models/ debug/ data/ 都是相对它的路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _bootstrap_path():
    """保证能 import 到同目录模块，并把 cwd 切到项目根目录（相对路径依赖它）。"""
    src_dir = str(Path(__file__).resolve().parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


_bootstrap_path()

import cv2  # noqa: E402  (必须在 sys.path 处理之后)
import tkinter as tk  # noqa: E402
from tkinter import messagebox, ttk  # noqa: E402

import config  # noqa: E402
from camera_adapter import create_camera  # noqa: E402
from concentric import POLARITY_NAME as POLARITY_CN  # noqa: E402
from measure_ring import MeasureError, build_parser, measure  # noqa: E402

LOG_PATH = PROJECT_ROOT / "debug" / "gui_error.log"
RUN_LOG = PROJECT_ROOT / "debug" / "gui_run.log"

COLOR_BG = "#ffffff"
COLOR_TEXT = "#111827"
COLOR_MUTED = "#6b7280"
COLOR_OK = "#1a7f37"
COLOR_NG = "#c62828"
COLOR_BUSY = "#546e7a"
COLOR_WARN_BG = "#fff6d6"
COLOR_WARN_FG = "#8a5300"
COLOR_ERR_BG = "#fdecea"
COLOR_ERR_FG = "#a01b14"
COLOR_BTN = "#1565c0"
COLOR_BTN_ALT = "#eceff1"
COLOR_CANVAS = "#111827"

AUTO_MEASURE_DELAY_MS = 1200   # 启动后先亮多久实时画面，再自动测第一次


def friendly_error(exc: BaseException) -> str:
    """把底层异常翻译成车间里看得懂的一句话。"""
    msg = str(exc)
    if "0x80000203" in msg:
        return ("相机被别的程序占用了（通常是 MVS 客户端）。\n"
                "请先关闭 MVS 软件，再点「再测一次」。")
    if "找不到海康相机" in msg:
        return ("没找到相机。请检查：网线是否插好、相机电源指示灯是否亮、"
                "网卡 IP 是否和相机同一网段。")
    if "全部无法取图" in msg:
        return ("相机能看见但取不到图像。常见原因：被 MVS 客户端占用、网线松动、IP 冲突。\n"
                "先关掉 MVS，再点「再测一次」。")
    if "取图超时" in msg:
        return "取图超时。请检查网线（相机网口指示灯是否亮）以及是否被其他程序占用。"
    if "MVS Python SDK" in msg:
        return "没找到海康相机 SDK，MVS 软件可能没装完整。请把 debug\\gui_error.log 发给技术支持。"
    if "打开摄像头" in msg:
        return "打不开摄像头，请检查驱动或换个 USB 口。"
    # MeasureError 的文案本身就是给用户看的
    return msg if isinstance(exc, MeasureError) else f"测量失败：{msg}"


def write_log(text: str):
    """把详细错误写进日志文件（窗口里只显示翻译过的一句话）。"""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")
    except OSError:
        pass


def pick_font(root) -> str:
    """挑一个系统里有的中文字体，避免显示成方块。"""
    from tkinter import font as tkfont
    available = set(tkfont.families(root))
    for name in ("Microsoft YaHei UI", "Microsoft YaHei", "微软雅黑",
                 "SimHei", "SimSun", "Segoe UI"):
        if name in available:
            return name
    return "TkDefaultFont"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _open_file(path):
    """用系统默认程序打开文件（Windows: 看图器 / 记事本）。"""
    try:
        os.startfile(str(path))          # noqa: S606  (Windows 专用)
    except OSError as e:
        messagebox.showwarning("打不开文件", f"{path}\n{e}")


# ======================================================================
# 引擎层：相机取帧 + 测量。不碰 tkinter，可脱离显示器无头验证。
# ======================================================================
class LiveSource:
    """相机 + 测量引擎（窗口只通过队列和它打交道）。

    数据出口：
        preview_q   最新一帧的 PNG 字节（满 1 帧就丢旧的，保证画面不过期）
        event_q     ("opened", None) / ("result", (RingResult, 缩略图字节)) / ("error", 异常)
    """

    def __init__(self, args, view_w: int):
        self.args = args
        self.view_w = view_w
        self.boost = False              # 仅显示用的亮度增强，不参与测量
        self.fps = 0.0
        self.frames = 0                 # 累计取到的帧数
        self.measured = 0               # 累计测量次数
        self._cam = None
        self._thread = None
        self._stop = threading.Event()
        self._measure_req = threading.Event()
        self._paused = threading.Event()
        self.preview_q = queue.Queue(maxsize=1)
        self.event_q = queue.Queue()

    # ---------------- 对外接口 ----------------
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0):
        """停线程并释放相机。capture() 最长阻塞 timeout_ms(3s)，故 join 给 3s。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def request_measure(self):
        self._measure_req.set()

    def set_paused(self, on: bool):
        if on:
            self._paused.set()
        else:
            self._paused.clear()

    def is_paused(self) -> bool:
        return self._paused.is_set()

    def is_ready(self) -> bool:
        """相机是否已打开（窗口靠它决定"测量"按钮能不能点）。"""
        return self._cam is not None

    def is_alive(self) -> bool:
        """取景线程是否还活着（相机掉了/取图失败后线程会退出）。"""
        return self._thread is not None and self._thread.is_alive()

    # ---------------- 后台线程 ----------------
    def _open_camera(self):
        if self.args.image:
            # 测已有照片：把照片当"相机"，画面静止
            return create_camera("file", image_path=self.args.image)
        return create_camera(config.CAMERA_TYPE, **config.CAMERA_KWARGS)

    def _run(self):
        try:
            self._cam = self._open_camera()
        except BaseException as e:          # noqa: BLE001 线程里必须兜住一切
            self.event_q.put(("error", e))
            return
        self.event_q.put(("opened", None))
        static_src = bool(self.args.image)  # 静态图没必要反复取帧（否则白烧 CPU）
        last = time.perf_counter()
        try:
            while not self._stop.is_set():
                # 测量优先于取景：先答应测量请求，避免"点了按钮要等好几帧"
                if self._measure_req.is_set():
                    self._measure_req.clear()
                    self._do_measure()
                    last = time.perf_counter()
                    continue
                if self._paused.is_set():
                    time.sleep(0.03)
                    continue
                try:
                    frame = self._cam.capture()
                except BaseException as e:  # noqa: BLE001
                    self.event_q.put(("error", e))
                    return
                now = time.perf_counter()
                dt = now - last
                last = now
                if dt > 0 and not static_src:
                    inst = 1.0 / dt
                    self.fps = inst if self.frames == 0 else 0.85 * self.fps + 0.15 * inst
                self.frames += 1
                self._push_preview(frame)
                if static_src:
                    self._paused.set()      # 静态图取一帧就够
        finally:
            cam, self._cam = self._cam, None
            if cam is not None:
                try:
                    cam.release()
                except Exception:           # noqa: BLE001 释放失败不该盖住真正原因
                    pass

    def _do_measure(self):
        try:
            frame = self._cam.capture()
            if not self.args.image:
                # 留痕：相机现拍时存一张原图，方便事后核对"当时拍的是什么"
                out = Path("data/captures") / "measure_ring_latest.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), frame)
            res = measure(frame, self.args)     # 内含落盘：调试图 + 文字报告
        except BaseException as e:              # noqa: BLE001
            self.event_q.put(("error", e))
            return
        self.measured += 1
        # 画好轮廓的结果图也编码一份，窗口直接显示（画面就此冻结，便于读数）
        thumb = self._encode(res.debug_img) if res.debug_img is not None else None
        self.event_q.put(("result", (res, thumb)))

    # ---------------- 图像处理 ----------------
    def _encode(self, frame):
        """降采样 → 显示增强 → 编码 PNG 字节流。

        重活（resize / PNG 编码）都留在后台线程，主线程只做 tk.PhotoImage。
        """
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return None
        s = self.view_w / float(w)
        small = cv2.resize(frame, (self.view_w, max(1, int(round(h * s)))),
                           interpolation=cv2.INTER_AREA)
        if self.boost:
            # 纯显示增强：现场光不够时也能看清零件位置。**不参与测量**，
            # 所以开启它不会影响读数（检测始终用相机原始亮度）。
            small = cv2.convertScaleAbs(small, alpha=3.0, beta=0)
        ok, buf = cv2.imencode(".png", small)
        return buf.tobytes() if ok else None

    def _push_preview(self, frame):
        buf = self._encode(frame)
        if buf is None:
            return
        try:
            self.preview_q.get_nowait()     # 丢掉上一帧：宁可跳帧也不要画面延迟
        except queue.Empty:
            pass
        self.preview_q.put(buf)

    def drain_preview(self):
        """清掉画布上可能残留的旧预览帧（显示测量结果前调用）。"""
        while True:
            try:
                self.preview_q.get_nowait()
            except queue.Empty:
                return


# ======================================================================
# 窗口层：只负责显示与交互，数值一律来自 LiveSource 回传的 RingResult
# ======================================================================
class MeasureWindow:
    def __init__(self, root: tk.Tk, args):
        self.root = root
        self.args = args
        self.ui = pick_font(root)
        self.result = None
        self._auto_done = False
        self._measuring = False
        self._img_ref = None            # 必须持引用，否则 PhotoImage 被回收显示空白

        screen_w = root.winfo_screenwidth()
        self.view_w = 880 if screen_w >= 1600 else 620
        self.view_h = int(self.view_w * 3036 / 4024)   # 12MP，固定画布避免窗口跳动

        self.src = LiveSource(args, self.view_w)

        self._build()
        # 窗口先出现（显示"正在打开相机"），再去干重活
        root.after(80, self._start_engine)

    # ---------------- 界面搭建 ----------------
    def _build(self):
        root = self.root
        root.title("圆环尺寸测量 - SurfaceIQ")
        root.configure(bg=COLOR_BG)

        # 顶部：标题 / 测量时间 / 帧率 / 状态徽标
        head = tk.Frame(root, bg=COLOR_BG)
        head.pack(fill="x", padx=18, pady=(14, 4))
        tk.Label(head, text="圆环尺寸测量", font=(self.ui, 17, "bold"),
                 bg=COLOR_BG, fg=COLOR_TEXT).pack(side="left")
        self.lbl_time = tk.Label(head, text="", font=(self.ui, 10),
                                 bg=COLOR_BG, fg=COLOR_MUTED)
        self.lbl_time.pack(side="left", padx=12)
        self.lbl_fps = tk.Label(head, text="", font=(self.ui, 10),
                                bg=COLOR_BG, fg=COLOR_MUTED)
        self.lbl_fps.pack(side="left")
        self.lbl_status = tk.Label(head, text="打开相机…", font=(self.ui, 17, "bold"),
                                   bg=COLOR_BUSY, fg="white", padx=16, pady=2)
        self.lbl_status.pack(side="right")

        body = tk.Frame(root, bg=COLOR_BG)
        body.pack(fill="both", expand=True, padx=18, pady=(8, 0))

        # 左列：最外那条（外缘）+ 同心边清单（条数不固定，所以是动态重建的 Frame）
        left = tk.Frame(body, bg=COLOR_BG)
        left.pack(side="left", anchor="n")
        self.lbl_od = self._number_row(left, "外缘（#1）", 40)
        tk.Label(left, text="同心边清单（按半径由外向内；不命名，请对照图纸判断）",
                 font=(self.ui, 10), bg=COLOR_BG, fg=COLOR_MUTED).pack(anchor="w",
                                                                      pady=(4, 2))
        self.list_frame = tk.Frame(left, bg=COLOR_BG)
        self.list_frame.pack(anchor="w")

        # 右列：实时取景画面
        right = tk.Frame(body, bg=COLOR_BG)
        right.pack(side="right", anchor="n", padx=(16, 0))
        self.canvas = tk.Canvas(right, width=self.view_w, height=self.view_h,
                                bg=COLOR_CANVAS, highlightthickness=1,
                                highlightbackground="#cfd8dc")
        self.canvas.pack()
        # 缩放/平移状态（结果图的"看细节"用；实时取景不受影响）
        self.full_img = None          # 结果的全分辨率标注图
        self._zoom = 1.0
        self._cx = self._cy = 0.0     # 视图中心（原图坐标）
        self._frozen = False          # True = 画面上冻结着一次结果（预览不许冲掉它）
        self._drag = None
        self.canvas.bind("<ButtonPress-1>", self._on_drag_press)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas_text = self.canvas.create_text(
            self.view_w // 2, self.view_h // 2, text="正在打开相机…",
            font=(self.ui, 13), fill="#9ca3af")
        tk.Label(right, text="绿=外缘   红=亮→暗边   品红=暗→亮边（编号与数值标在图上）",
                 font=(self.ui, 9), bg=COLOR_BG, fg=COLOR_MUTED).pack(pady=(4, 0))

        # 提醒 / 报错区（默认不显示，有内容才 pack）
        self.msg_frame = tk.Frame(root, bg=COLOR_WARN_BG)
        self.msg_label = tk.Label(self.msg_frame, text="", font=(self.ui, 11),
                                  bg=COLOR_WARN_BG, fg=COLOR_WARN_FG,
                                  justify="left", anchor="w", wraplength=1000,
                                  padx=10, pady=6)
        self.msg_label.pack(fill="x")

        # 按钮
        btns = tk.Frame(root, bg=COLOR_BG)
        btns.pack(fill="x", padx=18, pady=(10, 4))
        again_text = "重新测量 (空格)" if self.args.image else "再测一次 (空格)"
        self.btn_again = self._button(btns, again_text, self.do_measure, primary=True)
        self.btn_again.pack(side="left")
        self.btn_pause = self._button(btns, "暂停取景 (P)", self.toggle_pause)
        self.btn_pause.pack(side="left", padx=8)
        self.btn_big = self._button(btns, "看大图", self.open_debug)
        self.btn_big.pack(side="left")
        self.btn_rep = self._button(btns, "打开报告", self.open_report)
        self.btn_rep.pack(side="left", padx=8)
        self._button(btns, "放大 +", lambda: self._zoom_by(1.25)).pack(side="left")
        self._button(btns, "缩小 −", lambda: self._zoom_by(1 / 1.25)).pack(side="left", padx=6)
        self._button(btns, "复位", self._zoom_reset).pack(side="left")
        self._button(btns, "关闭 (Esc)", self.root.destroy).pack(side="right")

        # 显示增强：只改画面亮度，不改测量用的像素值。
        # takefocus=0 —— 否则点过复选框后焦点留在它身上，空格会去切换复选框
        # 而不是触发"测量"。
        self.var_boost = tk.BooleanVar(value=False)
        tk.Checkbutton(root, text="预览增亮 ×3（仅显示，不影响测量）",
                       variable=self.var_boost, command=self._on_boost,
                       font=(self.ui, 10), bg=COLOR_BG, fg=COLOR_TEXT,
                       activebackground=COLOR_BG, selectcolor="white",
                       takefocus=0, anchor="w").pack(fill="x", padx=18)

        self.lbl_foot = tk.Label(root, text="", font=(self.ui, 9), bg=COLOR_BG,
                                 fg=COLOR_MUTED, justify="left", anchor="w")
        self.lbl_foot.pack(fill="x", padx=18, pady=(2, 10))

        root.bind("<Escape>", lambda e: self.root.destroy())
        root.bind("<F5>", lambda e: self.do_measure())
        root.bind("<space>", lambda e: self.do_measure())
        root.bind("<p>", lambda e: self.toggle_pause())
        root.bind("<P>", lambda e: self.toggle_pause())
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        root.update_idletasks()
        w, h = root.winfo_width(), root.winfo_height()
        x = max(0, (root.winfo_screenwidth() - w) // 2)
        y = max(0, (root.winfo_screenheight() - h) // 4)
        root.geometry(f"+{x}+{y}")
        root.minsize(w, h)

    def _number_row(self, parent, title, size, pady=(0, 10)):
        """一行大数字：小标题 + 特大数字 + 单位。返回数字 Label 供更新。"""
        row = tk.Frame(parent, bg=COLOR_BG)
        row.pack(anchor="w", pady=pady)
        tk.Label(row, text=title, font=(self.ui, 12), bg=COLOR_BG,
                 fg=COLOR_MUTED).pack(anchor="w")
        line = tk.Frame(row, bg=COLOR_BG)
        line.pack(anchor="w")
        val = tk.Label(line, text="—", font=(self.ui, size, "bold"),
                       bg=COLOR_BG, fg=COLOR_TEXT)
        val.pack(side="left")
        tk.Label(line, text=" mm", font=(self.ui, max(10, size * 2 // 5)),
                 bg=COLOR_BG, fg=COLOR_MUTED).pack(side="left", anchor="s",
                                                   pady=(0, size // 4))
        return val

    def _button(self, parent, text, command, primary=False):
        kw = dict(text=text, command=command,
                  font=(self.ui, 12, "bold" if primary else "normal"),
                  relief="flat", padx=16, pady=6, cursor="hand2", takefocus=0)
        if primary:
            kw.update(bg=COLOR_BTN, fg="white", activebackground="#0d47a1",
                      activeforeground="white")
        else:
            kw.update(bg=COLOR_BTN_ALT, fg=COLOR_TEXT, activebackground="#dde3e8")
        return tk.Button(parent, **kw)

    # ---------------- 启停 ----------------
    def _start_engine(self):
        self.src.start()
        self.root.after(30, self._poll)

    def on_close(self):
        self.src.stop()
        self.root.destroy()

    # ---------------- 交互动作 ----------------
    def _on_boost(self):
        self.src.boost = bool(self.var_boost.get())

    def do_measure(self):
        self._frozen = False
        if not self.src.is_ready() or self._measuring:
            return                          # 相机没就绪，或上一次还没测完
        self._measuring = True
        # 先冻结画面再测，结果图才不会被后续预览帧冲掉
        self.src.set_paused(True)
        self._sync_pause_button()
        self._set_status("测量中", COLOR_BUSY)
        self.btn_again.config(state="disabled")
        self.src.request_measure()

    def _auto_measure(self):
        """启动后自动测一次：车间双击就该直接出结果，不用按键。"""
        if not self._auto_done and self.src.is_ready():
            self._auto_done = True
            self.do_measure()

    def toggle_pause(self):
        self.src.set_paused(not self.src.is_paused())
        self._sync_pause_button()
        if not self.src.is_paused():
            self._frozen = False        # 恢复取景就解冻，否则预览会被冻结的结果图挡住
            self._set_status("取景中", COLOR_BUSY)
            self.canvas.itemconfig(self.canvas_text, text="正在取景…", state="normal")

    def _sync_pause_button(self):
        self.btn_pause.config(text="继续取景 (P)" if self.src.is_paused() else "暂停取景 (P)")

    def open_debug(self):
        if self.result is not None and self.result.debug_jpg:
            _open_file(self.result.debug_jpg)

    def open_report(self):
        if self.result is not None and self.result.report_txt:
            _open_file(self.result.report_txt)

    # ---------------- 轮询：后台线程的东西只在这里落地 ----------------
    def _poll(self):
        try:
            buf = self.src.preview_q.get_nowait()
        except queue.Empty:
            buf = None
        if buf is not None and not self._frozen:
            self._show_image(buf)
            if self.src.fps > 0 and not self.src.is_paused():
                self.lbl_fps.config(text=f"{self.src.fps:.1f} fps")

        while True:
            try:
                kind, payload = self.src.event_q.get_nowait()
            except queue.Empty:
                break
            if kind == "opened":
                self._set_status("取景中", COLOR_BUSY)
                self.canvas.itemconfig(self.canvas_text, text="正在取景…", state="normal")
                self.root.after(AUTO_MEASURE_DELAY_MS, self._auto_measure)
            elif kind == "result":
                self._render_result(*payload)
            else:
                self._render_error(payload)

        self.root.after(30, self._poll)

    # ---------------- 渲染 ----------------
    def _set_status(self, text, color):
        self.lbl_status.config(text=text, bg=color)

    def _show_message(self, text, kind="warn"):
        """提醒区：kind='warn' 黄底提示，kind='error' 红底报错，None 隐藏。"""
        if not text:
            self.msg_frame.pack_forget()
            return
        bg, fg = (COLOR_WARN_BG, COLOR_WARN_FG) if kind == "warn" else (COLOR_ERR_BG, COLOR_ERR_FG)
        self.msg_frame.config(bg=bg)
        self.msg_label.config(text=text, bg=bg, fg=fg)
        self.msg_frame.pack(fill="x", padx=18, pady=(6, 0))

    def _show_image(self, buf):
        try:
            self._img_ref = tk.PhotoImage(data=base64.b64encode(buf))
        except tk.TclError as e:
            write_log(f"[{_now()}] 画面解码失败: {e}")
            return
        self.canvas.delete("img")
        self.canvas.create_image(self.view_w // 2, self.view_h // 2, anchor="center",
                                 image=self._img_ref, tags="img")
        self.canvas.itemconfig(self.canvas_text, state="hidden")

    def _render_result(self, r, thumb):
        self.src.drain_preview()            # 结果图不许被残留预览帧冲掉
        self.result = r
        self.lbl_time.config(text=r.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        self._set_status("已测量", COLOR_BUSY)

        # 没有机位系数时只显示像素：带 (px) 后缀 + 弱化颜色，不会被当成毫米读
        if r.outer_mm is None:
            self.lbl_od.config(text=f"{r.outer_px:.0f}(px)", fg=COLOR_MUTED)
        else:
            self.lbl_od.config(text=f"{r.outer_mm:.3f}", fg=COLOR_TEXT)

        # 清单条数不固定 -> 每次重建（这里就是主线程；tkinter 只能主线程碰）
        for w in self.list_frame.winfo_children():
            w.destroy()
        for e in r.edges[1:]:                       # #1 已经在大字里了
            mm = (f"{e['dia_px'] * r.ratio:9.3f} mm" if r.ratio is not None else "      (px)")
            tag = POLARITY_CN.get(e["pol"], "")
            lvl = ("弱", "中", "强")[e.get("strength", 2)]
            tk.Label(self.list_frame,
                     text=f"#{e['idx']:<3d} {e['dia_px']:8.1f} {mm}  {tag}  "
                          f"{lvl}  落差{e.get('step', 0):.0f}",
                     font=(self.ui, 12), bg=COLOR_BG,
                     fg=COLOR_MUTED if e.get("strength") == 0 else COLOR_TEXT,
                     anchor="w").pack(anchor="w")
        if len(r.edges) <= 1:
            tk.Label(self.list_frame, text="（只找到外缘：没有同心内边）",
                     font=(self.ui, 11), bg=COLOR_BG, fg=COLOR_MUTED).pack(anchor="w")

        msgs = ["⚠ " + w for w in r.warnings]
        if r.ratio is None:
            msgs.append("⚠ 机位没标定过：本次只有像素。用 --ref-mm <外径真值> 标定一次才有毫米。")
        if msgs:
            self._show_message("；\n".join(msgs), "warn")
        else:
            self._show_message(None)

        self.full_img = r.debug_img
        self._frozen = True
        self._zoom_reset()          # 新结果从"整图铺满"开始
        self.canvas.itemconfig(self.canvas_text, state="hidden")

        self.btn_again.config(state="normal")
        self._sync_pause_button()
        self._measuring = False
        coef = (f"换算系数: {r.ratio:.6f} mm/px（{r.ratio_src}）" if r.ratio is not None
                else f"换算系数: 无（{r.ratio_src}）")
        self.lbl_foot.config(
            text=f"图像: {r.src_desc}    {coef}"
                 f"    帧率: {self.src.fps:.1f} fps\n报告: {r.report_txt}")

    def _zoom_by(self, factor):
        """按倍数缩放结果图。只在"冻结着一次结果"时有效（实时取景不缩放）。"""
        if self.full_img is None:
            return
        self._zoom = min(8.0, max(0.2, self._zoom * factor))
        self._render_zoom()

    def _zoom_reset(self):
        if self.full_img is None:
            return
        self._zoom = 1.0
        h, w = self.full_img.shape[:2]
        self._cx, self._cy = w / 2.0, h / 2.0
        self._render_zoom()

    def _on_wheel(self, ev):
        self._zoom_by(1.2 if ev.delta > 0 else 1 / 1.2)

    def _on_drag_press(self, ev):
        self._drag = (ev.x, ev.y)

    def _on_drag_move(self, ev):
        if self._drag is None or self.full_img is None:
            return
        self._cx -= (ev.x - self._drag[0]) / self._zoom
        self._cy -= (ev.y - self._drag[1]) / self._zoom
        self._drag = (ev.x, ev.y)
        self._render_zoom()

    def _render_zoom(self):
        """从全分辨率标注图里裁一块、缩放到画布大小——这就是缩放/平移的全部实现。"""
        if self.full_img is None:
            return
        h, w = self.full_img.shape[:2]
        vw, vh = self.view_w, self.view_h
        cw = min(w, max(32, int(round(vw / self._zoom))))
        ch = min(h, max(32, int(round(vh / self._zoom))))
        x0 = max(0, min(int(round(self._cx - cw / 2)), w - cw))
        y0 = max(0, min(int(round(self._cy - ch / 2)), h - ch))
        self._cx, self._cy = x0 + cw / 2.0, y0 + ch / 2.0
        crop = self.full_img[y0:y0 + ch, x0:x0 + cw]
        interp = cv2.INTER_AREA if cw > vw else cv2.INTER_LINEAR
        view = cv2.resize(crop, (vw, vh), interpolation=interp)
        ok, buf = cv2.imencode(".png", view)
        if ok:
            self._show_image(buf.tobytes())

    def _render_error(self, exc):
        self._set_status("出错", COLOR_NG)
        self.btn_again.config(state="normal")
        self._sync_pause_button()
        self._measuring = False
        # 测失败就别把上一次的读数留在屏幕上误导人
        self.lbl_od.config(text="—", fg=COLOR_MUTED)
        for w in self.list_frame.winfo_children():
            w.destroy()
        brief = friendly_error(exc)
        if not self.src.is_alive():
            # 取图线程已退出（相机掉了 / 取图超时），画面不会再更新
            brief += "\n取景已停止：请关闭本窗口后重新打开。"
            self.btn_again.config(state="disabled")
            self.canvas.itemconfig(self.canvas_text, text="取景已停止", state="normal")
        elif not self.args.image:
            brief += "\n处理完点「再测一次」即可。"
        self._show_message(brief, "error")
        self.lbl_foot.config(text=f"详细日志: {LOG_PATH}")
        # 这里已经不在 except 块里（异常是后台线程捕获后经队列传过来的），
        # traceback.format_exc() 只会返回 "NoneType: None" —— 必须用异常对象本身取堆栈。
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        write_log(f"[{_now()}] {self.args.image or '相机现拍'}\n{tb}" + "-" * 60)


def _directory_arg(argv):
    """返回 argv 里第一个「目录」参数；没有则 None。

    拖拽文件夹到 测量.bat 时，bat 的 `%~x1` 判空分支会把文件夹当普通参数透传，
    而本程序的 argparse 只认 -- 开头的选项，于是抛错退出；又因为走 pythonw
    （无控制台）且 stdout/stderr 已被重定向进日志，用户看到的就是"完全没反应"。
    所以在这里先拦下来，给一句中文提示。
    """
    for a in argv:
        if a.startswith("-"):
            continue
        try:
            if Path(a).is_dir():
                return a
        except OSError:
            continue
    return None


def _warn_directory(arg: str):
    """弹窗提示「要的是照片文件，不是文件夹」，并写日志。"""
    msg = (f"这是文件夹，不是照片文件：\n{arg}\n\n"
           "• 要测已有照片：把 .jpg 照片文件拖到 测量.bat 图标上\n"
           "• 要用相机现拍：直接双击 测量.bat（不要带任何文件）")
    write_log(f"[{_now()}] 参数是文件夹，已拒绝\n  {arg}\n" + "-" * 60)
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning("只能拖照片文件", msg)
        root.destroy()
    except Exception:                    # 连弹窗都失败就只剩日志了
        pass


def main():
    # 先拦「文件夹参数」：argparse 对它的报错在 pythonw 下看不见
    folder = _directory_arg(sys.argv[1:])
    if folder is not None:
        _warn_directory(folder)
        return

    args = build_parser().parse_args()
    if args.image:                       # 先转绝对路径，再切 cwd
        args.image = str(Path(args.image).expanduser().resolve())
    os.chdir(PROJECT_ROOT)               # models/ debug/ data/ 都是相对项目根目录

    root = tk.Tk()
    try:
        win = MeasureWindow(root, args)
    except Exception:
        # 连窗口都没起来：弹个系统对话框，别出现"双击了没反应"
        write_log(f"[{_now()}] 窗口初始化失败\n{traceback.format_exc()}")
        try:
            messagebox.showerror("程序启动失败",
                                 "结果窗口没能打开。\n详细信息已记录到：\n"
                                 f"{LOG_PATH}")
        except Exception:
            pass
        raise
    try:
        root.mainloop()
    finally:
        win.src.stop()


if __name__ == "__main__":
    # pythonw 启动时没有控制台，print 会报错；把输出引到日志文件里
    try:
        RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
        _log = RUN_LOG.open("w", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = _log
    except OSError:
        pass
    main()
