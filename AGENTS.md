# AGENTS.md —— 给 AI 助手的规则清单

**本文件只放两样东西：必须遵守的规则、改了会静默出错的坑。** 事实、数据、设计依据不写在这里，
按下面这张表去对应的地方看（各自只有一个真相源，别在两边都写一遍）：

| 想知道什么 | 去哪看 |
|---|---|
| 怎么用：快速开始 / 命令行参数 / 排错速查 / **读数怎么解读** / 实测数据 | `README.md` |
| 为什么这么设计、参数依据、试过并否决的做法、回归用例的期望值 | `docs/自适应定位改造计划.md` |
| 已知缺陷（现象 / 证据 / 影响 / 修复方向）、**停手与重新开工的条件** | `docs/已知缺陷.md` |

**本文件不写行号。** 定位一律用「文件名 + 符号名」，没有名字的行就写一小段**可 grep 的原文**
（例如 `os.chdir(PROJECT_ROOT)` 那一行）。行号会在每次改动后静默失效、把人带到"看着合理但是错"
的位置；历史上被重新对齐过四次、四次都很快又过期。

---

## 0. 开工前先问（硬约束，优先级高于本文件其它条目）

**加任何代码之前，先回答：「我要加的东西，是不是某个现有文件本来就该有的？」**
答案是「是」就改那个文件——但**改动方案要先报给用户，等他确认再动手**。

- **不许自己另起新文件/新脚本/新窗口**。需要时先把方案和理由说给用户，同意了再写；
  确有必要也要说清**为什么不能改现有的那个文件**。
- **「现有文件是车间在用的入口」不是免改理由**——该改就改，但改法要先让用户拍板。
- **动手前先读现有实现**，确认要加的能力是不是已经被别的文件实现了（`gui_ring.py` /
  `capture_demo.py` 这类并列工具很容易功能重叠）。
- 用户否决、或你没把握时：**停下来问，不要先写完再解释**。

两条真实教训（细节见 git history）：① 曾经为了"不动车间在用的 `gui_ring.py`"另写了`gui_live.py`，
结果两份数字渲染各写一遍、依赖方向还反了，最后被要求合并回去、白做一轮；
② 曾经用 `feature: countersink|core_hole` 一个枚举同时决定"怎么定位孔"和"量哪条边界"，
型号一多就组合爆炸——**新增形态应当是"加一个策略函数"，不是"加一个枚举值"**。

## 1. 项目与边界

金属圆环/法兰零件的视觉尺寸测量：相机拍图 → 亚像素射线边缘 + 椭圆鲁棒拟合 →
输出外径 / 中心孔 / 各螺栓孔的毫米读数。

- **唯一在跑的业务线就是圆环尺寸测量**。commit `7375601` 已删除 EDM 粗糙度分类线与旧棋盘格
  标定路线，**不要以任何理由重新引入**（README 标题里的「锐感」、仓库名里的 "Roughness"、
  `**/models/*.pth` 的 gitignore 条目都是历史残留，不是待恢复的功能）。
- 实际依赖只有 `opencv-python` + `numpy`（`vision-roughness/requirements.txt`）。

## 2. 运行环境硬约束

- **解释器**：必须 `vision-roughness\venv\Scripts\python.exe`（3.11.9 / cv2 5.0.0 / numpy 2.4.6）。
  系统 Python 没有 cv2。
- **cwd 必须是 `vision-roughness/`**：`models/`、`debug/`、`data/captures` 全是相对 cwd 的
  `Path`（`measure_ring.py` 顶部的 `PROFILE_PATH` / `DEBUG_DIR`），且该文件用
  `sys.path.insert(0, "src")` 找同目录模块。
- **cwd 报错的正确读法**：在仓库根跑 `vision-roughness\src\measure_ring.py` 会报「找不到型号档案」——
  **档案其实存在**，先查 cwd。GUI 不受影响（`gui_ring.py` 的 `main()` 里 `os.chdir(PROJECT_ROOT)`）。
- **GUI 需要显示器与相机**：它会**阻塞在 tkinter `mainloop`**（得关窗口才返回）且**独占相机**
  （与别的程序同时开会撞 `0x80000203`）。CLI 不阻塞、也能走相机现拍（不传 `--image`），
  所以**能用 CLI 验证的就别开 GUI**。

## 3. 验收合约（回归）

**项目没有测试框架**（无 pytest / unittest / `def test_`）。改动检测相关代码后，
**下面两条都要跑**（cwd 一律 `vision-roughness/`）：

```bash
venv\Scripts\python.exe src\measure_ring.py --image ..\data\captures\capture_20260908_115557.jpg
venv\Scripts\python.exe src\measure_ring.py --image data\captures\capture_single_20261007_203729.jpg
```

**一条用例 = 一张参考图 + 它对应的档案**（不是"一个型号一条"：同型号换机位再拍一张，像素读数就变）。
两条用例覆盖**两条互不相干的检测路径**（`template_ngon`+`countersink_rim` / `profile_angles`+`dark_core`），
**缺一条就等于漏一半**。两张夹具图都随仓库提交，新克隆能跑全。

**判据只有两条**：

- **px 变了 = 检测被改坏**（这是唯一的回归信号：`外径:` / `中心孔:` / `螺栓孔N:` 各行里的
  **px 与椭圆轴长**必须逐字符不变）；
- **px 没变、mm 变了 = 系数被动过**（mm = px × 该型号档案的当前系数；`mm_per_px` 是**机位**的属性，
  换机位重标后 mm 必然整体变，这不是回归失败）。

`换算系数:` 那行的来源标签、`型号:` / `定位=` / `边界=` 这几行是可追溯信息，不参与比对。
**期望值（各条用例的 px）与"怎么判"的完整说明在 `docs/自适应定位改造计划.md`**，这里不重复。

**重标系数**用 `--profile <型号> --image <该机位的图> --ref-mm <真值>` 让工具自己算，**别手算**：
手算容易用四舍五入后的 px 而不是内部值（正是 `fe33d3a` 那条"别让第三位小数漂移"）。

## 4. 不可动 / 不入库 / 明确禁止

- **不许为了让读数对上标称值去改 `od_mm` / `mm_per_px`**（`models/part_profiles.json`）。
  `od_mm` 是卡尺真值、`mm_per_px` 由它标定出来；**那是伪造数据**。改正真值要用户给新数据。
- **`models/mm_per_pixel.json`（旧的全局缓存）已删除，别复活**：它是同一个数的第二份真相源，
  而且属于"上次标定的某个型号"。现在系数只存在型号档案里；查不到就**只报像素、不给毫米**。
- **不入库**（`.gitignore`）：`debug/`、`vision-roughness/data/captures/*`、`venv/`、`.zcode/`、
  `**/temp/*`（仅保留 `hik_debug.py`）、`**/models/*.pth`。运行时产物别 `git add`。
- **入库的参考数据**：`models/part_profiles.json`、`models/fingerprints/*.npz`（+ `.png` 预览）。
  指纹**必须入库**——它的参考图不入库，指纹是自动识别的唯一凭据。
- **`data/captures/` 的例外只有回归夹具**（§3 那两张，`.gitignore` 里用 `!` 单列）。
  以后再加夹具图照此办理：**在 `.gitignore` 里显式 `!` 出来，不要用 `git add -f`**（后者会留下
  "已跟踪但仍被忽略"的状态，后人 grep 时看不出是故意的）。
- **`测量.bat` 内容必须 ASCII + CRLF**（内部走 `pythonw` + `gui_ring.py`），存成 UTF-8 或 LF 会让 cmd 报错。
- 不要重新引入 `7375601` 删掉的粗糙度分类线 / 棋盘格标定路线；**不要新增第三方依赖**（tkinter 是标准库）。

## 5. 改了会静默出错的坑

只列"改错了不报错、只是数悄悄变了"的那类。**读数怎么解读**（螺栓孔量的是沉孔口、型号2 的孔是
紧固件暗芯、精度天花板、没档案不许编毫米……）在 `README.md` 的「限制」一节，不在这里重复。

- **`cv2.imread` / `cv2.imwrite` 遇中文路径不报错**：`imread` 返回 `None`、`imwrite` 返回 `False`
  （只往 stderr 打一行 WARN）。本项目报告/指纹预览的文件名就是中文——**读要判空、写要用
  `cv2.imencode` + `tofile`**。
- **`find_edge_points` 的梯度算在未平滑的原始径向剖面上**。要加平滑就是**改所有读数**
  （堆 `(n_ray,L)` 配 `(1,7)` 是沿剖面平滑、配 `(7,1)` 是跨射线平滑，两者都会动外径/孔径），
  属于测量定义变更，**不要顺手混进清理提交**。
- **mm/px 是"机位"的属性**：换相机高度/距离/倾角后 mm 会整体偏，必须对该型号重标；
  像素读数不受影响。（实测同一天内三套机位，外圈长轴 2826.7 / 2650 / 2868px。）
- **中心孔的偏差是无量纲比值，改系数救不了**：若检出的是"外圈半径的某个比例"变了，
  调 `mm_per_px` 只会等比缩放，改不了比例。遇到"比例不对"要先怀疑**锁错了哪条圆**，
  不是标定。详见 `docs/已知缺陷.md`。
- **外缘是一条"有厚度的带"**（倒角/圆角 + 斜视下有的角度亮、有的暗）：所以**同一个零件转个角度，
  检出的外径会变**（实测 47° 时 +2.5%），拟合椭圆会落到带的不同位置。任何"以检出外圈为基准"的
  东西（含指纹）都要考虑这件事。
- **改指纹的"展开"参数必须重建全部 `.npz`**：指纹存档里就是按这些参数算出来的数组。
  只改判定阈值（`FP_ACCEPT`/`FP_GAP`）不用重建；改 `FP_N_ANG`/`FP_N_RAD`/`FP_DEPTH`/
  `FP_ANCHOR_*`/`FP_SIGMA_RAD`/`FP_HARMONICS` 之后**每个型号都要 `--save-fingerprint` 重跑**。
  重建是确定性的（同样输入 → 逐位相同的数组）。
- **`--spec-*` 一个都不给时"综合判定: OK"不是合格判定**，只是"未判定"的默认串
  （`overall_judged=False`，GUI 显示「已测量」）。现在**判定只认命令行传的 `--spec-*`**，
  档案里的 `od_mm`/`id_mm` 还没接进判定——别误以为档案填了标称值就会自动判 OK/NG。

## 6. 代码地图（改什么去哪改）

- **换相机 / 曝光 / 增益**：只改 `src/config.py` 的 `CAMERA_TYPE` + `CAMERA_KWARGS`——
  这是换相机的唯一改动点，业务代码不许动。曝光默认自动标定（开机二分几帧把亮度调到
  `auto_target`，**开机因此多等 1~3 秒**；要严格复现历史读数就 `auto_exposure=False`）。
- **相机接入抽象层**：`src/camera_adapter.py`（`webcam` / `file` / `hikvision` 三实现，
  工厂函数 `create_camera()`），海康 DLL 搜索路径由它自动补。
- **检测管线核心**：`src/fit_ellipse_ring.py`。几何算法都改这里。
  - `find_outer(gray, init_c, init_r)` → `((椭圆, 点数) | None, σ=5 模糊图)`：外圈定位，
    **识别与检测共用**（识别也要外圈椭圆；识别若复用 `detect()` 会把整条螺栓孔管线白跑一遍）。
  - `detect(img, init_c, init_r, profile)` → `{"outer","center_hole","bolts","bolts_locate",
    "bolts_boundary","bolts_unverified"}`。
  - **外圈定位是自适应的**：`_outer_candidates` 给候选（旧先验排第一保基线，之后是
    「候选圆心 × 径向峰半径 × 两种极性」），由 `_outer_ok` 判可信。**判据只用点数/轴比/画幅，
    不能用残差 RMS**（基线自身 41px、错误极性 40.6px，区分不开）。
  - **中心孔定位也是自适应的**：`_inner_candidates` 给"多个比例带 × 两种极性"的候选，
    `_inner_ok` 用点数+轴比+在外圈内+**与外圈同心**判可信。**旧档位排第一，且它的 r0 必须
    严格用 0.70**（不是波段中点 0.69——圆等价滤点边界跟着 r0 走，差这一点读数就变；
    这个坑是基线回归门抓出来的）。**档案有 `id_mm`/`od_mm` 时**，先用真值孔径比
    （`id_mm/od_mm`）给一组窄带候选，带内取**最陡梯度**，再在"实测比落在 `INNER_RATIO_TOL` 内"
    的候选里挑最接近的；没有真值才只走旧档位。**别改成"取缓坡最内侧像素"**——孔壁是约 0.02r
    宽的缓坡，取坡脚会随窗口位置漂（实测 0.620~0.629 vs 最陡 0.630）。型号1 内圈有三圈
    （55.9 / ~60 / ~78mm），只量最内侧；残差与依据见 `docs/已知缺陷.md` 缺陷 1。
  - **螺栓孔是"两个正交维度"的派发**（`LOCATE_FNS` / `BOUNDARY_FNS`），**所有型号平级**：
    `locate`（怎么定位孔）= `profile_angles` | `template_ngon`；
    `boundary`（量哪条边界）= `countersink_rim` | `dark_core`。
    **新增形态 = 加一个策略函数（约 15 行），不碰 `detect()`**；
    原来那条 `feature=countersink|core_hole` 一维枚举**已删除，别加回来**。
  - **`ngon_slots` 的孔数来自档案 `bolt_count`**（缺省才从候选峰推断）；槽位没匹配到峰时按
    等角假设**合成**坐标并标 `src="synthesized"`，**报告和 GUI 都必须标出来**（以前它和实测孔
    长得一模一样）。
  - **`bolts_unverified=True`** 表示档案缺 `locate`/`boundary`（或整个没给）、走了兜底：
    此时**不许给毫米数**、结果要标「未经验证」。
  - **中心孔没检出时螺栓孔照样跑**（环带用外圈椭圆，有中心孔就挖掉，没有就用整个盘面）。
- **外圈定位失败时要说清原因**：`measure_ring.locate_fail_hint` 区分「画面太暗」/「画面过曝」/
  「其他」，别退回成一句笼统的「检查对焦、光照、零件是否在画面中心」。
- **结果唯一来源**：`measure_ring.measure()` 产出 `RingResult`，CLI 打印它、GUI 显示它——
  **不要在 `gui_ring.py` 里自己再算一遍尺寸**。
- **型号档案是唯一入口**：`models/part_profiles.json`。字段：`od_mm` / `id_mm` /
  `has_center_hole` / `bolt_mm` / `bolt_count` / `bolt_angles_rel`（相对角度，支持非等分）/
  `bolt_circle_ratio` / `locate` / `boundary` / `mm_per_px` / `fingerprint_src`。
- **型号识别 = 外圈环带的极坐标指纹**（`polar_fingerprint` / `match_fingerprint`）。不传
  `--profile` 时：`find_outer` → 环带按归一化半径展开 → 与各型号指纹做**沿角度的循环互相关** →
  最高分过 `FP_ACCEPT` 且与次高分拉开 `FP_GAP` 才认；否则**不许猜**，只给像素 + 提示人工选型号
  （界面顶部「型号」下拉框 / `--profile`）。**参数依据、被否决的做法、现场实测数据**见
  `docs/自适应定位改造计划.md` 的「型号识别」一节与 `docs/已知缺陷.md`。
  生成/更新指纹：`--image <参考图> --profile <型号名> --save-fingerprint`（**必须显式带
  `--profile`**，同 `--ref-mm` 的理由：否则会悄悄覆盖某个型号的指纹）。
- **显示层**：`src/gui_ring.py` 只做显示/交互，不改数值。它负责**实时取景**，所以有一条硬约束：
  取帧与测量**只在同一个后台线程**（`LiveSource`）里做，**绝不能让两个线程同时 `capture()`**
  （相机是独占资源）；tkinter 只能在主线程碰，后台线程只往队列塞「编码好的 PNG 字节」和事件。
  螺栓标签写死 6 个（`CIRCLED`），孔数 >6 时会**显式提示**「只显示前 6 个」，不许静默丢。
  型号下拉框把选中项写进 `self.args.profile`（与命令行同一套参数），**`takefocus=0` 是必须的**
  （否则点过它之后空格会去开下拉菜单而不是测量）。
- **调参前必须先看数据**，按此顺序跑诊断脚本（cwd 同上，传图片路径）：
  `explore_ring.py` → `profile_ring.py` → `ray_ring.py` → `edge_ring.py`。**不要凭直觉改阈值。**
- 采集：`capture_single.py`（无窗口连拍）/ `capture_demo.py`（带预览手动采图）。

## 7. 约定与提交风格

- 中文注释 / docstring / 用户可见输出；报告文件按 UTF-8 写。
- `measure_ring.MeasureError` 表示**「可预期、需原样提示给用户」的失败**（读图失败、外圈拟合失败等），
  `main()` 里会把它转成一句中文退出。**代码缺陷不要用 MeasureError 包**，要让真实 traceback 抛出来。
- 提交信息：**中文 + `type:` 前缀**（`feat` / `fix` / `refactor` / `docs` / `chore`）。**一次提交只做一件事。**
- **本文件不出现行号**（见文件开头）：要指某个位置，用符号名或可 grep 的原文片段。
