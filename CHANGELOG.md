# Changelog

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased]

### 新增

- **M1 · 体模与投影几何 + 正向投影器**（`src/ct_forward.py`）。这是交接手册 §8 的
  **第一个待建模块**，也是 §1.2 列出的第一个缺口（"无正向投影器 → 无法生成过采样数据"）。
  M4 的 K 帧过采样、M5 的重建、M7 的尺度验证都要建在它上面。

  - 解析体模：圆柱 / Shepp-Logan / 分辨率卡（椭圆叠加表）
  - **解析正向投影**：椭圆弦长闭式解。注意常见写法 `2√((1−s²Q)/P)` **只对圆截面或
    未旋转椭圆成立**，旋转椭圆下交叉项不抵消，本实现用完整二次式
    `√(B²−4PC)/P`，`B = 2s·sinθcosθ·(1/b²−1/a²)`
  - **数值正向投影**：射线驱动 + 双线性采样，体模栅格化支持抗锯齿子采样
  - 投影几何输出：`M`、`p_eff = p/M`、样品面视场与采样网格、Nyquist
  - 验收（手册判据 < 1%）：圆柱 0.030%、Shepp-Logan 0.803%（n=2048）；
    误差随网格**一阶收敛**（体素减半、误差减半）

- 新增 `scripts/verify_m1.py` 与 `scripts/verify_sync.py`：把两条技术线的验收做成
  可复现脚本（**8/8** 与 **24/24** 通过），对应手册 §9 的"复现与验证清单"。

- **同步辐射仿真模式**（第 6 种架构 `synchrotron`）—— 以**单元光子计数**架构改造，
  把「无像素约束 PCD + 多次过采样能否逼近同步辐射 CT」这条技术线从定性讨论变成
  **可复算的整链模型**。新增 `src/ct_synchrotron.py`（纯 numpy、无 Qt 依赖）。

  硬件侧参数化：源类型（波荡器 / 弯铁 / 液态金属射流靶 / 微焦点 / 纳米焦点 /
  旋转阳极，各带亮度档与相干性）、焦点 `f`、探测器材料（CdTe / CZT / Si / GaAs，
  各带 K 边与 Kα 与 `λ_K`）、像素 `p`、电荷云 `σ_c`、整形时间 `τ`、过采样帧数 `K`、
  能量箱与阈值、入射通量 `φ`。

  算法侧可复算：MTF 级联（圆盘焦点 `2J₁(x)/x` · 孔径 `|sinc(πpν)|` · 电荷云高斯）
  与 10%/50% 截止；`p_eff = p/M`、半影 `b = f(M−1)/M`、`r(M)`、
  **`M* = 1+(p/f)²`**、**`r* = pf/√(p²+f²)`** 与实用放大率；
  过采样 `f_N(K) = K/(2p)` 与 **K ≥ 2 饱和**结论；电荷共享判据 **`p ≳ 2σ_c`**；
  K 荧光逃逸长度；脉冲堆积对照 `1/τ`；VMI 权重（光电 + Klein–Nishina 两基最小二乘，
  回传残差）；Paganin 单距离相衬核；剂量幂律 `N³~N⁴`、源亮度差、Swank 因子、
  半 Nyquist 处 DQE/NEQ 与香农容量。共 **62 项 `sync_*` 派生量**。

- 新增 `_ci/verify_sync.py`：以交接手册的数值表为判据，**24/24 通过**
  （`M*`/`r*` 五行、`f₀`/`f_N(K)`、剂量幂律、`λ_K`、电荷共享阈值，以及 §5 手算的
  `p/M = 13.75` / `b = 7.5` / `r ≈ 15.7 µm`）。

- **FBP Process 页右侧新增「同步辐射仿真路径」面板**（仅该架构可见）：把手册的级联
  模型逐级作用在**刚重建出的真实切片**上，同一标尺并排显示——① 基准 ② 微焦点半影
  ③ 单帧采样 ④ K 帧过采样 ⑤ 电荷共享 ⑥ K 荧光逃逸 ⑦ 反卷积收口 ⑧ 相衬；下方给出
  MTF10 / r(M) / r* / DQE / 香农 / FOV 与「体素 vs PSF」的反卷积有效性判据。

- 控制桥新增 `tab` op，可切换重建分析页签（Agent 需要先把页签切到前台才能截图核对）。

### 修复

- **架构切换匹配错误**：`_on_arch` 原用"关键字包含"匹配，而同步辐射的标签是
  「同步辐射 (Synchrotron · 单元光子计数改造)」，其中含"光子计数"四字，
  会被 `pcct` 抢先命中，导致切不到同步辐射。改为对选项文本尾部做**前缀**匹配，
  并把「同步辐射」排在「光子计数」之前。
- **同步辐射的 FBP 基线被门禁误拦**：原逻辑把"能量维 > 1"一律当成需要 Fermi 分箱。
  但同步辐射是**准单色高通量**源，FBP 基线本就该按单色等效（VMI）跑，能量箱属于
  下游材料分解。现已单独豁免并给出明确口径提示。
- **同步辐射的重建体素取错**：原借用临床 `iso_sampling`（625 µm，机架通道采样），
  与样品面尺度差约 40 倍，导致"反卷积是否有效"的判断完全相反。现用
  `p/M/K`（默认 3.44 µm），结论正确。
- 本模块 `simulate_paths` 两处物理错误：单帧采样漏乘孔径 MTF（导致单帧反而比过采样
  亮）、K 荧光按整幅模糊（严重夸大，实为部分串扰）。均已修正。

## [1.0] — 2026-09-22

首个正式版本。此前只有零散提交，没有任何 tag、Release 或版本号；
本版把项目整理成可分发、可复现、可引用的状态。

### 新增

- **五种 CT 架构统一建模**：双源 `dual_source`、单源宽体 `single_wide`、
  双层能谱 `dual_layer`、光子计数 `pcct`（8 能量箱）、静态多源 `static_multi`
  （24 源环形阵列，不旋转）
- **几何内核**（`ct_geometry.py`）：扇形角 / 最小间距（150 mm）/ 180° 弧差（120°）
  安全校验、**探测器阵列整数化自动匹配**（先取整再反解真实像素间距）、
  Bowtie 滤过器与真正视野、非对称准直、扫描协议闭环（剂量指数 / 时间分辨率 / 螺距上限）
- **LEAP-CT 1.26 重建后端**（`ct_leap.py`）：扇束 FBP、锥束 FBP、螺旋
  （360LI / 180LI / 180MF z 插值）、模块束 + SART / ASDPOCS 迭代、短扫描 Parker 加权
- **Fermi 半导体探测器响应**（`ct_fermi.py` + `Fermi/` 子项目）：深度加权 CCE、
  能量箱匹配、双层叠加、电场与输运剖面
- **严格多材料谱分解**（`ct_spectral.py`）：逐能量正投影 + 非线性牛顿法，
  碘浓度回收误差由线性化的 57.5% 降到 **6.1%**（逐像素 Lᵢ/L_w 比值标定后）
- **AI Agent 控制桥**：应用内 TCP 桥（`ct6_bridge.py`）+ MCP stdio 服务器
  （`ct6_mcp_server.py`，5 个工具：`ct6_call` / `ct6_set_geometry` /
  `ct6_switch_arch` / `ct6_run_fbp` / `ct6_screenshot`）。写变量**无白名单**，
  Agent 可直接改 `alpha` / `RA` / `RB` / `FDD` 等几何主参数
- **一键构建与发布**：`scripts/build_windows.ps1` 负责全部构建参数
  （workflow 只调用它），产物为 Windows onedir 包

### 修复（自早期原型以来）

打包与 CI 曾长期存在"构建绿色但产物不可用"的问题，本版逐一修掉：

- **CI 构建失败**：`pyinstaller` 用了 cmd 的 `^` 续行符，而 GitHub Windows runner
  的 `run:` 跑在 PowerShell 下 → `ParserError: Missing expression after unary operator '--'`
- **exe 一启动即崩**：`frozen_entry` 用 `runpy.run_path` 动态加载主程序，
  PyInstaller 静态分析看不到 → 从未打包进去。现由
  `--add-data "src/simulate_ct6.py;."` 随产物分发
- **LEAP-CT 显示"不可用"**：`imageio/__init__.py` 在 import 期需读取自己的
  `dist-info` 版本号，而 PyInstaller 默认不打包 `.dist-info`
  → `import leapctype` 抛 `PackageNotFoundError`。
  现由 `--copy-metadata imageio --copy-metadata lazy_loader` 解决
- **exe 图标不是 LOGO**：从未传 `--icon`，PyInstaller 嵌了自己的默认图标。
  现每次构建由 `scripts/make_icon.py` 从 `assets/LOGO.jpg` 现生成 `logo.ico` 并传入
- **Fermi 静默失效**：`ct_fermi` 的 `FERMI_ROOT` 用 `dirname(dirname(__file__))`
  推断，冻结后指向错目录。改为候选目录探测
- **产物虚胖**：`libleapct.dll` 的 PE 导入表实际只需 `cufft64_11.dll`
  （+ nvJitLink / nvrtc），并不需要 cublas / cusparse。产物由 981.5 MB 降到 **905.5 MB**
- **旧工作流假绿灯**：`build.yml` 编译的是已废弃的 PyQt5 原型 `simulate_qt.py`，
  且 PyQt5 不在依赖里，每次构建都产出缺模块的废 exe。该工作流已删除
- 另修：`.ps1` 缺 UTF-8 BOM 导致 PowerShell 5.1 解析失败、
  PowerShell 把 PyInstaller 的 stderr 当终止错误、脚本中文输出撞 CI 的 cp1252 控制台

### 工程与合规

- **冻结环境自检**（`DSW_CT_SELFTEST`）：在**打包产物内部**真的 import 全部关键模块
  并真的构造一次 LEAP 引擎（CPU），失败即构建失败。"文件都在"≠"import 得动"，
  这类问题从此挡在 CI 里
- **许可**：项目自有代码以 **MIT** 发布；随附 `THIRD_PARTY_NOTICES.md` 说明
  LEAP-CT（MIT）、xraylib（BSD-3-Clause）、Qt/PySide6（**LGPL-3.0**）等义务
- **许可原文自动化**：`scripts/collect_licenses.py` 从已安装分发的元数据收集
  19 个依赖的真实许可文件进产物，不靠手抄
- 打包固定使用 `--onedir`：Qt 的 `*.dll` / `*.pyd` 保持为独立文件，
  接收者可自行替换 —— 这是 LGPL-3.0 合规的要求，**请勿改成 `--onefile`**

### 已知限制

- 螺旋 `helicalPitch` 单位为 **mm/rad**（LEAP 约定），代码内已按 `每圈进 mm / 2π` 换算
- 谱分解碘浓度误差约 **6%**，主要残差来自 FBP 域边缘伪影
- 静态多源提供"等效旋转锥束"与"模块束 + SART"两条路径；
  xraylib 化合物表缺失，改用元素加权构造 μ
- `src/legacy/` 的 PyQt5 版本仅作历史参考，不再构建、不再修复
- `PyOpenGL` 的 wheel 未自带许可文件，`THIRD_PARTY_NOTICES.md` 中已单独注明

[1.0]: https://github.com/hg3992260/digital_ct_simulate/releases/tag/v1.0
