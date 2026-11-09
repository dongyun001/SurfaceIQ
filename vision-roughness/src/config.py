"""全局配置：换相机只需改这个文件，业务代码不用动。

换相机流程：
    1. 改下面的 CAMERA_TYPE / CAMERA_KWARGS；
    2. 用卡尺量出零件外径真值，重建 mm/px 换算系数：
           python src\\measure_ring.py --ref-mm <外径真值mm>
       系数跟着"相机+镜头+拍摄距离"走，三者任一变化都要重建。
"""

# ========== 相机配置 ==========
# 可选类型: "webcam"（USB/笔记本摄像头）| "hikvision"（海康工业相机）| "file"（调试用图片）
CAMERA_TYPE = "hikvision"
CAMERA_KWARGS = {
    "sdk_path": r"E:\海康sadp\MVS\Development\Samples\Python\MvImport",
    "exposure_us": 30000.0,      # 手动模式的固定曝光；自动标定只拿它当搜索起点
    "gain": 0,
    # 开机时按现场亮度自动标定曝光、标定完锁定（2026-10-07 加）。原因：固定 30ms
    # 在现场光不足时会把整帧打成全黑（实测均值 16/255），外圈检测全废——那比
    # "亮度漂移"严重得多。auto_target=110 取自唯一那张能稳定测出的标定图（实测
    # 均值 109.7）。要严格复现历史读数，把 auto_exposure 改成 False。
    "auto_exposure": True,
    "auto_target": 110.0,
    "auto_max_us": 200000.0,     # 曝光上限 200ms：再长实时取景就只剩几帧
    # 削顶像素（≥250）占比上限。亮金属过曝会把边缘往内推（实测孔壁比 0.630→0.618），
    # 超过这个比例就自动把曝光往下收；但不会收到把画面压暗（见 AUTO_MIN_MEAN）。
    "auto_sat_max": 0.01,
}

# 海康相机示例（MVS 装好后切换为下面几行）:
# CAMERA_TYPE = "hikvision"
# CAMERA_KWARGS = {"sdk_path": None, "exposure_us": 20000, "gain": 0,
#                  "auto_exposure": True, "auto_target": 110.0}


# ========== 侧视厚度（同一台相机换 0.3x 远心镜头） ==========
# 流程：装上 0.3x 远心 + 背光 → --mode thickness 测厚度 → 换回顶视镜头 →
# 外径测量读这次缓存做高度补偿。相机是独占资源，不要试图同时开两个视角。
THICKNESS = {
    # 外径测量时厚度的来源：
    #   "latest" = 读 debug/thickness_latest.json（厚度模式刚测的那次，带时间戳）
    #   "manual" = 只认 --thickness / 界面输入
    "source": "latest",
    "max_age_min": 30,        # 缓存有效期：超过就拒绝补偿（宁可少补，不套错零件）

    # ---- 侧视几何（换镜头 / 换工装后按现场复核） ----
    "roi": None,              # (x0, x1) 列范围；None = 全宽
    # dark_part = 亮背景、暗零件/台面（背光）；bright_part = 暗背景、亮零件
    "polarity": "dark_part",
    "smooth_sigma": 1.5,
    "min_grad": 0.0,          # 0 = 按图自适应；背光很弱时可手填
    "min_profiles": 20,       # 顶面/台面各自至少这么多列
    "min_gap_px": 80.0,       # 两条边至少差这么多像素（0.0115mm/px 约 0.9mm）
    "edge_search_px": 24,     # 首次越阈后找梯度峰值的窗口（离焦时峰值才无偏）
    "tilt_max_deg": 0.5,      # 水平线允许的倾斜；超了告警
    "max_mad_px": 5.0,        # 每条边的离散度上限（MAD）
}
