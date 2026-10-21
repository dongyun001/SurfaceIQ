"""相机适配器层：把不同相机抽象成统一接口，方便替换相机品牌。

契约（所有相机必须遵守）：
    capture() 必须返回 BGR 三通道 uint8 图像。黑白相机（如 MV-CU120-10GM）
    必须在适配器内部把单通道灰度图转成 BGR，否则下游 measure.py 的
    COLOR_BGR2GRAY 会崩溃。统一用 BaseCamera._ensure_bgr() 处理。

切换相机：只需修改 src/config.py 的 CAMERA_TYPE / CAMERA_KWARGS。
"""
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

# ---- 自动曝光的高光约束 ----
# 亮金属被 255 削顶会把边缘位置往内推：同一零件、只改曝光，8~32ms 孔壁比稳定在
# 0.630，105ms（亮环整片顶到 255）掉到 0.618。所以按均值标定完还要压一下削顶占比；
# 但压得太狠会把画面弄暗、检测先坏，所以同时有均值下限。
AUTO_SAT_LEVEL = 250      # 视为"削顶"的灰度阈值
AUTO_MIN_MEAN = 40.0      # 为压高光最多把均值降到这儿（再低就得不偿失）
AUTO_SAT_STEPS = 5        # 最多往下收几档（每档 ×0.75）


def _add_mvs_runtime_to_dll_path():
    """把 MVS 运行时 DLL 目录补进 DLL 搜索路径。

    MVS 5.x 把 MvCameraControl.dll 装到 Common Files 并写入系统 PATH，
    但安装前已打开的终端/进程拿不到新 PATH，这里显式补一份，保证
    任何时刻启动的进程都能加载 SDK。
    """
    candidates = [
        os.environ.get("MVCAM_COMMON_RUNENV"),
        r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
        r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64",
    ]
    for d in candidates:
        if not d or not os.path.isdir(d):
            continue
        if d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(d)
        except (OSError, AttributeError):
            pass  # Python<3.8 无此接口，PATH 方案已足够


class BaseCamera(ABC):
    """相机抽象基类。所有具体相机都需要实现 capture 和 release。"""

    @abstractmethod
    def capture(self) -> np.ndarray:
        """捕获一帧图像，返回 BGR 三通道 uint8 格式的 numpy 数组。"""
        pass

    @abstractmethod
    def release(self):
        """释放相机资源。必须可重复调用（幂等）。"""
        pass

    @staticmethod
    def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
        """把灰度图/BGRA 图统一转换为 BGR 三通道，保证下游代码无需关心相机类型。"""
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class WebcamCamera(BaseCamera):
    """电脑摄像头实现（基于 OpenCV VideoCapture）。

    用法：
        with WebcamCamera(device_id=0) as cam:
            frame = cam.capture()
    """

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._cap = None
        self._open()

    def _open(self):
        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 device_id={self.device_id}，请检查驱动或换一个编号")
        # 等待几帧，让摄像头自动曝光稳定
        for _ in range(3):
            self._cap.read()

    def capture(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("摄像头采图失败，请检查连接")
        return self._ensure_bgr(frame)

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class FileCamera(BaseCamera):
    """文件相机：每次 capture 都返回同一张本地图片。

    用于无摄像头时的单元测试或服务调试。
    """

    def __init__(self, image_path: str):
        self.image_path = Path(image_path)
        self._frame = cv2.imread(str(self.image_path))
        if self._frame is None:
            raise ValueError(f"无法读取图片: {self.image_path}")

    def capture(self) -> np.ndarray:
        return self._ensure_bgr(self._frame.copy())

    def release(self):
        pass


class HikvisionCamera(BaseCamera):
    """海康机器人 GigE 工业相机（基于 MVS Python SDK，如 MV-CU120-10GM）。

    前置条件：
        1. 安装海康 MVS 软件（含 Python SDK 组件）；
        2. 相机已通过网线与电脑连接并上电，MVS 客户端能枚举到相机。

    用法（config.py 中配置）：
        CAMERA_TYPE = "hikvision"
        CAMERA_KWARGS = {"sdk_path": None, "exposure_us": 20000, "gain": 0}

    注意：
        - MV-CU120-10GM 是黑白相机，SDK 输出 Mono8 单通道，capture() 内部
          已转换为 BGR 三通道，下游代码无需改动；
        - 曝光默认按现场亮度**自动标定一次再锁定**（见 auto_exposure）：现场光不足时
          固定曝光会把整帧打成全黑（实测均值 16/255，外圈检测全废），那比"亮度漂移"
          严重得多。标定完曝光值即固定，同一次开机内读数依然可复现。
          要严格复现历史读数，传 auto_exposure=False 并用固定 exposure_us。

          （历史注：这里原本写"固定曝光，避免自动曝光导致亮度漂移——纹理特征对亮度
           敏感"。那条理由属于 commit 7375601 已删除的 EDM 粗糙度纹理分类线；现在
           只做尺寸测量，靠边缘不靠纹理。见 AGENTS.md §1。）
    """

    DEFAULT_SDK_PATH = r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"

    def __init__(self, sdk_path: str = None, exposure_us: float = None,
                 gain: float = None, auto_exposure: bool = True,
                 auto_target: float = 110.0, auto_max_us: float = 200000.0,
                 auto_sat_max: float = 0.01,
                 timeout_ms: int = 3000, packet_size=1500):
        self._timeout_ms = timeout_ms
        self._packet_size = packet_size
        self._cam = None
        self._grabbing = False
        self._auto_exposure = auto_exposure
        self._auto_target = auto_target
        self._auto_max_us = auto_max_us
        self._auto_sat_max = float(auto_sat_max)
        self.exposure_us = exposure_us          # 最终生效值（自动标定后会更新）
        self._mv = self._import_sdk(sdk_path or self.DEFAULT_SDK_PATH)
        self._open(exposure_us, gain)

    @staticmethod
    def _import_sdk(sdk_path: str):
        """导入海康 MVS Python SDK；失败时给出操作指引。"""
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
        _add_mvs_runtime_to_dll_path()
        try:
            import MvCameraControl_class as mv
        except ImportError as e:
            raise RuntimeError(
                f"找不到海康 MVS Python SDK（{e}）\n"
                "请按以下步骤处理：\n"
                "  1. 安装海康机器人 MVS 软件（安装时勾选 Python SDK）；\n"
                "  2. 确认 MvCameraControl_class.py 的实际位置（通常在\n"
                f"     {HikvisionCamera.DEFAULT_SDK_PATH}）；\n"
                "  3. 若路径不同，在 config.py 的 CAMERA_KWARGS 中设置 sdk_path。"
            )
        return mv

    def _check(self, ret: int, action: str):
        """检查 SDK 返回码，非 0 抛出带中文说明的异常。"""
        if ret != 0:
            raise RuntimeError(f"海康相机操作失败: {action} (错误码 0x{ret:08x})")

    def _open(self, exposure_us, gain):
        """枚举 GigE 相机，逐个尝试打开并试取一帧，直到成功。

        同一台相机经多个网卡（或与其他设备 IP 冲突）会被枚举成多项，
        其中可能只有部分枚举项能真正收到视频流——"打开成功"不算数，
        必须试取一帧验证。
        """
        from ctypes import POINTER, cast

        mv = self._mv
        device_list = mv.MV_CC_DEVICE_INFO_LIST()
        ret = mv.MvCamera.MV_CC_EnumDevices(mv.MV_GIGE_DEVICE, device_list)
        self._check(ret, "枚举设备")
        if device_list.nDeviceNum == 0:
            raise RuntimeError(
                "找不到海康相机，请检查：\n"
                "  1. 网线是否连接相机与电脑，相机是否上电（指示灯）；\n"
                "  2. 电脑网卡 IP 是否与相机同一网段（用 MVS 的 IP 配置工具检查）；\n"
                "  3. MVS 客户端里能否枚举到相机。"
            )

        errors = []
        for i in range(device_list.nDeviceNum):
            gige = cast(device_list.pDeviceInfo[i], POINTER(mv.MV_GIGE_DEVICE_INFO)).contents
            model = bytes(gige.chModelName).split(b"\x00")[0].decode("ascii", "ignore")
            ip = gige.nCurrentIp
            ip_str = ".".join(str((ip >> s) & 0xFF) for s in (24, 16, 8, 0))
            print(f"[hikvision] 尝试相机{i}: {model} @ {ip_str}")
            try:
                self._try_open_one(device_list, i, exposure_us, gain)
                print(f"[hikvision] 相机{i} 取图验证通过，已就绪")
                if self._auto_exposure:
                    try:
                        self._calibrate_exposure()
                    except Exception as e:      # 标定失败不该让整个相机判为不可用
                        print(f"[hikvision] 自动曝光标定失败，沿用当前曝光: {e}")
                return
            except Exception as e:
                errors.append(f"  相机{i}({model} @ {ip_str}): {e}")

        raise RuntimeError(
            "枚举到的海康相机全部无法取图：\n" + "\n".join(errors) +
            "\n请检查：网线是否插在有相机的网口、相机是否被 MVS 客户端占用、"
            "是否存在 IP 冲突（两台设备同 IP）。"
        )

    def _set_exposure(self, exposure_us: float):
        """设置曝光时间——只走手动曝光通道，不碰自动曝光节点。"""
        self._check(self._cam.MV_CC_SetEnumValue("ExposureAuto", 0), "关闭自动曝光")
        self._check(self._cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_us)),
                    "设置曝光时间")
        self.exposure_us = float(exposure_us)

    def _grab_stats(self, settle: int = 2):
        """取一帧，返回 (平均亮度, 削顶像素占比)；先丢 settle 帧等新曝光值生效。

        改完曝光，流里还在传改之前那一帧；不丢帧就会把旧亮度当成新亮度的结果，
        二分搜索会被带偏。
        """
        for _ in range(settle):
            self.capture()
        g = self.capture()
        return float(g.mean()), float((g >= AUTO_SAT_LEVEL).mean())

    def _calibrate_exposure(self, iters: int = 6):
        """按现场亮度自动标定曝光，标定完即锁定（auto_exposure=True 时开机跑一次）。

        现场光不足时固定曝光会把整帧打成全黑（实测均值 16/255），外圈检测全废——
        这比"亮度漂移"严重得多。标定只跑一次，之后曝光值固定，同一次开机内读数
        仍可复现。

        两步：
        1. 按**对数**二分把画面均值调到 `auto_target`（可用跨度 0.2~200ms，线性二分
           在暗端分辨率不够）。只用手动曝光通道（ExposureAuto=0 + ExposureTime，
           本机型已验证可用），不依赖 Brightness / AutoExposureTime* / ExposureAuto=Once
           这些各机型命名不一的节点，免得出错在节点名上反而把相机弄成打不开。
        2. **高光兜底**：削顶像素（≥`AUTO_SAT_LEVEL`）超过 `auto_sat_max` 时，按 0.75 倍
           往下收曝光。亮金属削顶会移动边缘位置（实测孔壁比 0.630→0.618）；但最多收
           `AUTO_SAT_STEPS` 档，且均值不得低于 `AUTO_MIN_MEAN`——再低画面就太暗，
           检测会先坏，那种情况宁可保留一点高光并明确告警。
        """
        lo, hi = np.log(200.0), np.log(float(self._auto_max_us))
        target = float(self._auto_target)
        best = None
        for _ in range(iters):
            us = float(np.exp((lo + hi) / 2))
            self._set_exposure(us)
            mean, _ = self._grab_stats()
            if best is None or abs(mean - target) < best[0]:
                best = (abs(mean - target), us, mean)
            if mean < target:       # 太暗 -> 加曝光
                lo = (lo + hi) / 2
            else:                   # 太亮 -> 减曝光
                hi = (lo + hi) / 2
        # 一轮都没接近目标 -> 现场太暗：直接顶到曝光上限，至少把图拍亮，并明确告警。
        # （判断看在"有没有到过目标亮度"，不看"曝光是否等于上限"——对数二分收敛后
        #   落在下限附近而不是上限值本身，拿等于上限去判会漏报。）
        dark = best[2] < target * 0.9
        chosen = float(self._auto_max_us) if dark else best[1]
        self._set_exposure(chosen)
        mean, sat = self._grab_stats()

        # ---- 高光削顶兜底 ----
        for _ in range(AUTO_SAT_STEPS):
            if sat <= self._auto_sat_max:
                break
            nxt = max(self.exposure_us * 0.75, 200.0)
            if nxt >= self.exposure_us:
                break
            prev = (self.exposure_us, mean, sat)
            self._set_exposure(nxt)
            m2, s2 = self._grab_stats()
            if m2 < AUTO_MIN_MEAN:      # 再收就太暗 -> 回退，保留一点高光
                self._set_exposure(prev[0])
                mean, sat = prev[1], prev[2]
                break
            mean, sat = m2, s2

        warn = "  ⚠️ 现场偏暗，已顶到曝光上限，考虑补光" if dark else ""
        if sat > self._auto_sat_max:
            warn += (f"  ⚠️ 仍有 {sat * 100:.1f}% 高光削顶（≥{AUTO_SAT_LEVEL}）："
                     "亮金属过曝会让边缘内偏，考虑补光/减弱直射反光")
        print(f"[hikvision] 自动曝光: {self.exposure_us:.0f}us"
              f"（均值 {mean:.1f}，目标 {target:.0f}；削顶 {sat * 100:.2f}%）{warn}")

    def _try_open_one(self, device_list, idx, exposure_us, gain):
        """打开指定枚举项、配置参数、开始采集并试取一帧；失败抛异常并清理。"""
        from ctypes import POINTER, byref, cast, memset, sizeof

        mv = self._mv
        cam = mv.MvCamera()
        device_info = cast(device_list.pDeviceInfo[idx], POINTER(mv.MV_CC_DEVICE_INFO)).contents
        self._check(cam.MV_CC_CreateHandle(device_info), "创建设备句柄")
        try:
            self._check(cam.MV_CC_OpenDevice(mv.MV_ACCESS_Exclusive, 0), "打开设备")

            # GigE 网口包大小：默认 1500（标准以太网，任何网卡都稳定）；
            # 现场网卡开启巨型帧(Jumbo Frame)后可调大（如 8164）提升帧率，
            # 或传 packet_size="auto" 用 SDK 推荐值（部分网卡虚报巨型帧支持，会丢包）
            pkt_size = self._packet_size
            if pkt_size == "auto":
                pkt = cam.MV_CC_GetOptimalPacketSize()
                pkt_size = int(pkt) if 0 < pkt < 0x80000000 else 1500
            self._check(cam.MV_CC_SetIntValue("GevSCPSPacketSize", int(pkt_size)),
                        "设置网口包大小")

            # 连续采集模式（关闭触发；现场若用光电触发，在这里改 TriggerMode/TriggerSource）
            self._check(cam.MV_CC_SetEnumValue("TriggerMode", mv.MV_TRIGGER_MODE_OFF),
                        "设置触发模式")

            # 手动曝光/增益：锁定成像参数，避免亮度漂移影响纹理特征
            if exposure_us is not None:
                self._check(cam.MV_CC_SetEnumValue("ExposureAuto", 0), "关闭自动曝光")
                self._check(cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_us)),
                            "设置曝光时间")
            if gain is not None:
                self._check(cam.MV_CC_SetEnumValue("GainAuto", 0), "关闭自动增益")
                self._check(cam.MV_CC_SetFloatValue("Gain", float(gain)), "设置增益")

            self._check(cam.MV_CC_StartGrabbing(), "开始采集")

            # 试取一帧：枚举项可能对应收不到视频流的网卡，打开成功≠能出图
            frame = mv.MV_FRAME_OUT()
            memset(byref(frame), 0, sizeof(frame))
            try:
                self._check(cam.MV_CC_GetImageBuffer(frame, self._timeout_ms),
                            "试取一帧")
            finally:
                if frame.pBufAddr:
                    cam.MV_CC_FreeImageBuffer(frame)
        except Exception:
            try:
                cam.MV_CC_StopGrabbing()
            except Exception:
                pass
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            raise
        self._cam = cam
        self._grabbing = True

    def capture(self) -> np.ndarray:
        """取一帧图像，返回 BGR 三通道数组；超时抛出带排查指引的异常。"""
        from ctypes import byref, memset, sizeof, string_at

        if self._cam is None or not self._grabbing:
            raise RuntimeError("海康相机未在采集状态，请重新创建相机实例")

        mv = self._mv
        frame_out = mv.MV_FRAME_OUT()
        memset(byref(frame_out), 0, sizeof(frame_out))

        ret = self._cam.MV_CC_GetImageBuffer(frame_out, self._timeout_ms)
        if ret != 0:
            raise RuntimeError(
                f"海康相机取图超时/失败 (错误码 0x{ret:08x})，请检查：\n"
                "  1. 网线是否松动（GigE 链路灯是否亮）；\n"
                "  2. 相机是否被其他程序（如 MVS 客户端）占用。"
            )

        try:
            info = frame_out.stFrameInfo
            data = string_at(frame_out.pBufAddr, info.nFrameLen)

            if info.enPixelType == mv.PixelType_Gvsp_Mono8:
                # 黑白相机：单通道灰度图
                img = np.frombuffer(data, dtype=np.uint8).reshape(info.nHeight, info.nWidth)
            elif info.enPixelType == mv.PixelType_Gvsp_BGR8_Packed:
                img = np.frombuffer(data, dtype=np.uint8).reshape(info.nHeight, info.nWidth, 3)
            elif info.enPixelType == mv.PixelType_Gvsp_RGB8_Packed:
                img = np.frombuffer(data, dtype=np.uint8).reshape(info.nHeight, info.nWidth, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                raise RuntimeError(
                    f"暂不支持的海康相机像素格式: 0x{info.enPixelType:08x}\n"
                    "请在 MVS 客户端把 Pixel Format 改为 Mono8（黑白）或 BGR8/RGB8（彩色）。"
                )
            return self._ensure_bgr(img.copy())
        finally:
            self._cam.MV_CC_FreeImageBuffer(frame_out)

    def release(self):
        """停止采集并释放设备，可重复调用。"""
        cam, self._cam = self._cam, None
        if cam is None:
            return
        try:
            if self._grabbing:
                cam.MV_CC_StopGrabbing()
        finally:
            self._grabbing = False
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()


def create_camera(camera_type: str = "webcam", **kwargs) -> BaseCamera:
    """工厂函数：根据配置创建对应相机实例。"""
    camera_type = camera_type.lower()
    if camera_type == "webcam":
        return WebcamCamera(**kwargs)
    if camera_type == "file":
        return FileCamera(**kwargs)
    if camera_type == "hikvision":
        return HikvisionCamera(**kwargs)
    raise ValueError(f"不支持的相机类型: {camera_type}")
