# Changelog

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

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
