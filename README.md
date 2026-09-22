# digital_ct_simulate

**双源 CT 真实几何模拟器 · Native Edition（CT6）**

[![windows-build](https://github.com/hg3992260/digital_ct_simulate/actions/workflows/windows-build.yml/badge.svg)](https://github.com/hg3992260/digital_ct_simulate/actions/workflows/windows-build.yml)
[![release](https://img.shields.io/github/v/release/hg3992260/digital_ct_simulate?label=release&color=3070B3)](https://github.com/hg3992260/digital_ct_simulate/releases/latest)
[![license](https://img.shields.io/github/license/hg3992260/digital_ct_simulate?color=3070B3)](LICENSE)

> 当前版本 **v1.0** —— 变更见 [`CHANGELOG.md`](CHANGELOG.md)。

> 双源宽体 CT 的**几何 / 数据 / 重建数字孪生**：6 种 CT 架构（含**同步辐射仿真模式**）×
> LEAP-CT 物理重建 × 半导体探测器响应（Fermi）× 严格多材料谱分解，
> 并内置**面向 AI Agent 的控制桥（MCP）**，让 Agent 能直接读写几何主参数、
> 切换架构、跑重建、取截图。

不是"画个好看的示意图"，而是**从几何恒等式一路接到真实重建算子**的研发台：
左边栏改一个 `alpha`，扇形角、最小间距、螺旋螺距上限、探测器阵列整数化匹配、
剂量指数、时间分辨率会同时重算；点一下 FBP，走的是 LEAP-CT 的锥束/螺旋正投影与反投影。

- **程序入口**：`src/simulate_ct6.py`（界面标题：双源 CT 真实几何模拟器 · Native Edition）
- **Windows 产物**：`DSW_CT.exe` —— 取自 GitHub Actions 的 `DSW_CT_Windows` artifact
- **Agent 入口**：`src/ct6_mcp_server.py`（MCP stdio）+ `src/ct6_bridge.py`（应用内 TCP 控制桥）

---

## 界面预览

> 以下截图均由**程序自身**渲染（构造真实 `MainWindow` 后调用 `win.grab()`，
> 与 Agent 控制桥的 `shot` 能力同源），拍的是**当前版本**界面，不是早期 PyQt5 原型。

**主界面** —— 左栏「系统几何参数 / 扫描方案 / 计算结果」（含 Bowtie 真正视野与边缘剂量比），
右栏 `System Geometry` 与 `FOV Impact Analysis` 双 3D 视口：

![主界面](docs/ct6_main.png)

| FBP 重建流程（LEAP 四联图） | 光子计数架构 · Fermi 探测器响应 |
| :---: | :---: |
| ![FBP](docs/ct6_fbp_process.png) | ![Fermi](docs/ct6_pcct_fermi.png) |

**静态多源 24 源环形阵列** —— 架构切到 `static_multi` 时，3D 视口中可见整圈源环：

![静态多源](docs/ct6_static_multi.png)

---

## 核心能力

- **五种 CT 架构**统一建模：双源、单源宽体、双层能谱、光子计数、静态多源阵列
- **同步辐射仿真模式**：以单元光子计数架构改造，把「PCCT 能否逼近同步辐射」这条
  技术线从定性讨论变成可计算的整链（源亮度 → 焦点半影 → 孔径 MTF → 电荷共享 /
  K 荧光 → VMI / 相衬 → 剂量与信息量）
- **几何内核**：扇形角 / 最小间距 / 180° 弧差安全校验、探测器阵列**整数化自动匹配**、
  Bowtie 滤过器、非对称准直、扫描协议闭环（剂量 / 时间分辨率 / 螺距上限）
- **真实重建**：LEAP-CT 1.26 —— 扇束 FBP、锥束 FBP、螺旋（360LI / 180LI / 180MF 插值）、
  模块束（静态阵列）+ SART / ASDPOCS 迭代、短扫描 Parker 加权
- **探测器物理**：CZT / TlBr 等半导体响应的深度加权 CCE、能量箱匹配、双层叠加
- **谱分解**：逐能量正投影 + 非线性牛顿法，输出基材料分解与有效 μ
- **Agent 原生**：应用内 TCP 控制桥 + MCP stdio 服务器，Agent 可脚本化一切

---

## 六种 CT 架构

| 架构 key | 源数 | 能量维 | 旋转 | 说明 |
| :--- | ---: | ---: | :---: | :--- |
| `dual_source` | 2 | 1 | ✅ | 双源，两套系统按 `alpha` 夹角同时采样 |
| `single_wide` | 1 | 1 | ✅ | 单源宽体，靠大锥角覆盖 |
| `dual_layer` | 1 | 2 | ✅ | 双层探测器能谱（2 层） |
| `pcct` | 1 | 8 | ✅ | 光子计数（8 能量箱），配合 Fermi 响应 |
| `static_multi` | 24 | 1 | ❌ | 静态多源环形阵列，每个源一个视角 |
| `synchrotron` | 1 | 可配（默认 8） | ✅ | **同步辐射仿真模式**，以单元光子计数架构改造 |

架构不只是"标签"：它直接决定采样几何。旋转架构按 `views_per_rot × n_src` 计视角数；
静态架构不旋转，`180°` 对应的是**一半源数**（`static_src_180 = round(n_src/2)`），
并额外给出 `static_src_short / static_src_used / static_span_deg / static_ok_180`
用于判断短扫描可用性。

### 五架构三维几何

> 均为当前程序自身渲染的 `SYSTEM GEOMETRY` 视口实拍（`win.grab()`）。
> 左上角是实时参数浮层，其中 `System Load` 为机架离心力 —— 静态架构不旋转，故为 `0 N`。

| 双源 Dual-Source（两套系统 α=95°） | 单源宽体 Single-Source Wide-Body |
| :---: | :---: |
| ![双源](docs/arch_dual_source.png) | ![单源宽体](docs/arch_single_wide.png) |

| 双层能谱 Dual-Layer（2 层） | 光子计数 Photon-Counting（8 能量箱） |
| :---: | :---: |
| ![双层能谱](docs/arch_dual_layer.png) | ![光子计数](docs/arch_pcct.png) |

**静态多源 Stationary 24-Source** —— 不旋转，24 个源/探测器模块均布整圈：

![静态多源](docs/arch_static_multi.png)

---

## 同步辐射仿真模式

以**单元光子计数**架构改造的第 6 种架构（`arch='synchrotron'`），把
「无像素约束 PCD + 多次过采样能否逼近同步辐射 CT」这条技术线从定性讨论
变成**可复算的整链模型**。实现在 [`src/ct_synchrotron.py`](src/ct_synchrotron.py)，
纯 numpy、无 Qt 依赖。

### 级联模型

```
源能谱 S(E) ─▶ 焦点模糊 ─▶ 样品 μ(x,y,E) ─▶ 像素孔径 ─▶ 电荷云/串扰 ─▶ 计数分 bin
   亮度 B         MTF_f        投影 p(θ,t)       MTF_ap        MTF_ch         NPS(E)
```

### 硬件参数（GUI 中仅该架构可见）

| 组 | 参数 |
| :--- | :--- |
| **源** | 类型（波荡器 / 弯铁 / 液态金属射流靶 / 微焦点 / 纳米焦点 / 旋转阳极，各带亮度档与相干性）、焦点尺寸 `f` |
| **几何** | 复用 `RA`（= SOD）与 `FDD`（源-探测器），故 `M = (SOD+ODD)/SOD = FDD/RA` ——"把样品推向源"就是调小 `RA` |
| **探测器** | 材料（CdTe / CZT / Si / GaAs，各带 K 边与 Kα）、像素 `p`、电荷云 `σ_c`、整形时间 `τ`、阵列尺寸 |
| **采集** | 子像素过采样帧数 `K`、能量箱数与阈值、标称能量、入射通量 `φ` |

### 算法（全部可复算）

| 环节 | 公式 / 判据 |
| :--- | :--- |
| 采样与过采样 | `f₀ = 1/p`（孔径零点）、`f_N(K) = K/(2p)`；**K ≥ 2 即把可恢复频带填满到 `f₀`**，K > 2 无新增信息 |
| MTF 级联 | `MTF_sys = MTF_focus(圆盘 2J₁(x)/x) · MTF_aperture(\|sinc(πpν)\|) · MTF_charge(高斯)`，输出 10% / 50% 截止 |
| 几何与最优放大 | `p_eff = p/M`、半影 `b = f(M−1)/M`、`r(M) = √(f²(M−1)² + p²)/M`、**`M* = 1+(p/f)²`**、**`r* = pf/√(p²+f²)`** |
| 电荷共享 | 能谱可用判据 **`p ≳ 2σ_c`**；低于阈值时给出能谱保持度的经验衰减 |
| K 荧光逃逸 | `λ_K`（CdTe/CZT 249 µm、Si 15 µm、GaAs 40 µm），以像素为单位给出污染范围 |
| 脉冲堆积 | 计数率对照 `1/τ`，给出堆积率与是否可忽略 |
| 谱与 VMI | bin 阈值/中心/宽度；VMI 权重由**光电 + Klein–Nishina 两基最小二乘**解出，并回传残差 |
| 相衬 | Paganin 单距离相位恢复核参数 `δ/β · λR/4π`；源无相干性时明确警告收益按 0 计 |
| 剂量与信息量 | 剂量幂律 `N³`（体素）～`N⁴`（角度同步加密）、源亮度差、Swank 因子、半 Nyquist 处 `DQE`/`NEQ`、香农容量 |

> **报告频点说明**：DQE/NEQ 取**半 Nyquist**。有效 Nyquist 恰好落在 `f₀ = 1/p` 上，
> 那里 `MTF_ap` 为首零点恒等于 0，在该处报 DQE 会得到 0 而无意义。

### 公式验收

模块的数值与交接手册逐条对齐（`_ci/verify_sync.py`，**24/24 通过**）：

| 判据来源 | 内容 | 结果 |
| :--- | :--- | :--- |
| 手册 §4.3 表 | `M*`、`r*` 五行（f = 10/5/5/1/0.5 µm × p = 55/55/200/55/55 µm） | 全部精确复现 |
| 手册 §5 | `f₀`、`f_N(1/2/4)`、K=2 饱和 | 全部一致 |
| 手册 §5 | 剂量幂律 N = 2/4/10 → ×8/×64/×1000 | 全部一致 |
| 手册 §4.6 | Cd Te K 荧光逃逸长度 | 一致 |
| 手册 §4.5 | `p ≳ 2σ_c` 能谱失效阈值 | 一致 |
| 手册 §5 手算 | 默认配置（f = 10 µm、p = 55 µm、M = 4）→ `p/M = 13.75`、`b = 7.5`、`r ≈ 15.7` µm | 精确复现 |

### 诚实边界

`ct_synchrotron.py` 输出的是**模型计算值，不是实测标定值**。交接手册 §1.2 明确指出
原技术线只有可行性分析、没有仿真器实现，且 MTF / NPS / 剂量曲线原为"示意构造"；
本模块把那些示意曲线换成了可复算的解析模型，**但仍不等于实测**，
引用时请按手册 §6.2 的口径注明来源。

---

## 快速开始

### 方式 A：下载 Windows 可执行文件（推荐）

前往 **[Releases](https://github.com/hg3992260/digital_ct_simulate/releases/latest)**，
下载 `DSW_CT_Windows_v1.0.zip`，解压后直接运行 `DSW_CT.exe`（`_internal` 需与它同级）。

若想拿最新的开发版构建，也可以走
**Actions → [`windows-build`](https://github.com/hg3992260/digital_ct_simulate/actions/workflows/windows-build.yml)
→ 最新一次成功 run → Artifacts → `DSW_CT_Windows`**。

> 产物已内含 CUDA 运行时，且构建流程带**冻结环境自检**（真的 import 全部模块并构造一次
> LEAP 引擎），所以不会出现"构建绿了但一跑就崩"的包。

### 方式 B：源码运行

```bat
:: 需要 Python 3.11
pip install -r requirements.txt
run.bat
```

或

```bat
python src\simulate_ct6.py
```

`run.bat` 会把 `src/`、`vendor/leapct`、`vendor/xraylib`、`vendor/cuda` 一起加进 `PYTHONPATH`。

---

## 目录结构

```
src/
  simulate_ct6.py      主程序（PySide6 + PyCt6 界面，3285 行）
  ct_geometry.py       几何内核：5 架构 / 阵列整数化匹配 / Bowtie / 协议闭环
  ct_leap.py           LEAP-CT 引擎：扇束 / 锥束 / 模块束+迭代 / 螺旋 / 短扫描
  ct_helical.py        螺旋 z 插值（360LI / 180LI / 180MF）
  ct_index.py          sinogram ↔ 探测器单元 ↔ 几何 解析索引 + z 重排
  ct_fermi.py          Fermi 半导体响应桥接（深度加权 CCE / 能量箱 / 双层）
  ct_spectral.py       严格多材料谱分解（逐能量正投影 + 非线性牛顿）
  ct_scene.py          3D 场景与伪影可视化
  ct6_bridge.py        应用内 TCP 控制桥（Agent 入口）
  ct6_mcp_server.py    MCP stdio 服务器（5 个工具）
  frozen_entry.py      PyInstaller 打包入口 + 冻结环境自检
  legacy/              早期 PyQt5 版本（保留参考，不再维护）
scripts/
  build_windows.ps1    Windows 打包脚本（workflow 只负责调用它）
  collect_cuda.py      从 nvidia wheel 收集 CUDA 运行时 DLL
  make_icon.py         由 assets/LOGO.jpg 生成 assets/logo.ico
assets/                logo / 图标 / 启动图
docs/                  界面截图
Fermi/                 半导体探测器响应模拟子项目（ct_fermi 依赖其 src/）
vendor/leapct/         LEAP-CT 1.26（leapctype.py + libleapct.dll 18MB）
vendor/xraylib/        xraylib 4.3.0（xraylib.py + 两个 pyd，29MB）
vendor/cuda/           仓库只带 cudart64_12.dll，其余由构建脚本补齐
```

---

## 几何内核要点

### 探测器阵列整数化自动匹配

物理上通道数与排数必须是整数，但由弧长/覆盖反推往往得到小数。内核的做法是
**先算出理想值，再取整，再用取整结果反解真实像素间距**，保证几何自洽：

```
n_ch_ideal = arc / pixel_xy          n_ch   = round(n_ch_ideal)
n_rows     = round(Z_coverage / pixel_z)

col_pitch   = arc / n_ch                          # 反解，替代名义 pixel_xy
row_pitch_iso = Z_coverage / n_rows
row_pitch_det = row_pitch_iso × FDD / RA           # 投影到探测器平面
array_total   = n_ch × n_rows
```

默认参数（`alpha=95°`、`RA=RB=600`、`FDD=1100`、`Z=160`、`pixel_z=0.625`）下：

```
n_ch = 825        col_pitch = 1.1461 mm      array_residual_px = (0.169, 0.0)
n_rows = 256      row_pitch_iso = 0.625 mm   array_ok = True
array_total = 211200
```

`array_residual_px` 是取整残差（单位：像素），`array_ok` 是"整数化后仍满足视场要求"的判据。
也可用 `n_ch_set` 手动钉死通道数（`n_ch_manual`）。

### 其它同步回算量

`alpha / RA / RB / FDD / SFOV_A / SFOV_B` 一改，下列量全部重算：
扇形角 A/B、最小间距 `min_dist`（安全阈值 150 mm）、180° 弧差 `arc_diff`（阈值 120°）、
`is_safe` / `is_arc_ok`、几何效率 `beta_A/beta_B`、`kappa`、探测器高度 `H_det_A/B`、
螺距上限（双源/单源分别一个）、`ctdi_rel` 剂量指数、`t_res_arch_ms` 时间分辨率、
`slices` / `views_total` / `data_cells` 等。

---

## 重建后端：LEAP-CT

`ct_leap.py` 把 LEAP-CT 1.26 的 C++ 算子接成 Python 侧的统一接口：

- `FanGeometry` 直接采用内核算出的 `n_ch / n_rows / col_pitch / row_pitch_det / array_total`
- 锥束：`set_conebeam(...)` + 曲面探测器；螺旋用 `set_helicalPitch()`
- 静态阵列：`set_modularbeam(...)`（模块束）→ 解析重建或 SART / ASDPOCS
- 短扫描：< 360° 时走手工 FBP 链（`preRampFiltering → rampFilterProjection →
  postRampFiltering → weightedBackproject`）并叠 Parker 加权

> ⚠️ **`helicalPitch` 的单位是 mm/弧度**（LEAP 约定），不是"每圈进给比"。
> 代码内已按 `pitch × Z_cov / (2π)` 换算；这是最容易静默出错的一处。

---

## 探测器物理：Fermi 子项目

`Fermi/` 是一个独立的半导体探测器响应模拟器（CZT / TlBr / CdTe / Si / GaAs …）。
`ct_fermi.py` 负责桥接，向主程序提供：

- 材料参数与载流子输运剖面（`get_transport_profile()`）
- 深度加权 CCE（电荷收集效率）与权重场
- 能量箱响应矩阵 / 源谱 / 箱权重 → 光子计数与双层架构的分箱展开
- 电场剖面（`get_electric_field_profile()`）

Fermi 的 `src/` 内部使用顶层导入（`from src.material import ...`），因此
`ct_fermi` 需要把 `Fermi/` 与 `Fermi/src/` 都加进 `sys.path` —— 打包时也必须把
`Fermi/` 作为**真实目录**带上，不能只丢进 PYZ。

---

## 谱分解

`ct_spectral.py` 做严格多材料分解：

- μ 表优先用 **xraylib**；水等材料改用**元素加权**构造（pip wheel 常缺化合物表，
  `CS_Total_CP('Water', E)` 会直接报错）
- `per_energy_sinograms` 逐能量正投影，`bin_from_energy_maps` 按箱聚合
- `decompose`（线性化）与 `decompose_newton`（逐射线 2 参数牛顿，显式 2×2 正规方程，
  回传残差）
- 谱与响应在能量网格上做**单元积分**（`spectrum_on_grid` / `response_on_grid`）

精度演进（碘浓度回收，真值 10 mg/mL 量级）：

| 方法 | 相对误差 |
| :--- | ---: |
| 线性化分解 | 57.5% |
| 非线性牛顿 | 10.4%（残差 4.9e-07） |
| 牛顿 + 逐像素 Lᵢ/L_w 比值标定 | **6.1%** |

---

## AI Agent 控制（MCP）

### 控制链路

```mermaid
flowchart LR
    A["AI Agent<br/>(Claude / DSH / …)"] -->|"MCP · stdio · JSON-RPC 2.0"| B["ct6_mcp_server.py"]
    B -->|"TCP · 换行分隔 JSON"| C["ct6_bridge<br/>应用内 TCP 服务"]
    C -->|"QObject 队列信号<br/>(跨线程安全)"| D["PySide6 主线程 / GUI"]
    D --- E["ct_geometry<br/>几何内核"]
    D --- F["ct_leap<br/>LEAP-CT"]
    D --- G["ct_fermi<br/>ct_spectral"]
```

程序启动时把监听端口写进发现文件 `ct6_bridge.json`，Agent 侧读它即可连上 ——
所以 **Agent 不需要知道端口号**，只要程序在跑。

### 工具集（1 通用 + 4 高频）

| 工具 | 作用 |
| :--- | :--- |
| `ct6_call(op, params)` | 通用调用，`op ∈ {ping, list, get, set, arch, scan, fbp, fermi, shot, view}` |
| `ct6_set_geometry(**params)` | 写几何/扫描方案参数（`alpha, RA, RB, FDD, SFOV_A/B, Z_coverage, rotation_time, sampling_rate, pixel_xy, pixel_z, n_ch_set, bowtie_sfov, bowtie_edge, pitch, scan_length, slice_thickness, slice_interval, shots_per_source, nch_auto_sw, asymmetric_cb, scan_mode`） |
| `ct6_switch_arch(arch)` | 切换架构：`dual_source \| single_wide \| dual_layer \| pcct \| static_multi` |
| `ct6_run_fbp(settle_ms)` | 执行 FBP 重建并等待落定 |
| `ct6_screenshot()` | 回传窗口截图（MCP image 内容） |

`list` 列出全部可写输入变量，`get` 额外回传上百项派生结果（数量随架构与页签可见性变化）。
**写变量没有白名单**——包括 `alpha / RA / RB / FDD` 这些几何主参数，Agent 都能直接改。

### 接入配置

```json
{
  "mcpServers": {
    "ct6": {
      "command": "D:\\python\\envs\\mar\\python.exe",
      "args": ["I:\\dsw\\digital_ct_simulate\\src\\ct6_mcp_server.py"]
    }
  }
}
```

前提：仿真程序已启动（否则工具会返回"未连接：仿真程序未运行或发现文件缺失"）。

### 自检

```bat
echo {"jsonrpc":"2.0","id":1,"method":"tools/list"} | python src\ct6_mcp_server.py
```

---

## 构建与发布

CI 只做三件事：装依赖 → 调 `scripts/build_windows.ps1` → 上传产物。
**所有构建参数都在仓库脚本里**（不是 workflow 里），改构建不用动 workflow。

```bat
:: 本地复现同一套构建 + 自检
pwsh -File scripts\build_windows.ps1
```

脚本会依次：

1. `scripts/collect_cuda.py` —— 从 nvidia wheel 收集 CUDA 运行时到 `vendor/cuda/`
2. `scripts/make_icon.py` —— 由 `assets/LOGO.jpg` 现生成 `assets/logo.ico`
3. PyInstaller `--onedir` 打包（含 `--icon`、`--copy-metadata`、`--add-data vendor/*`）
4. **冻结环境自检**：在产物内部 import 全部关键模块并构造一次 LEAP 引擎，失败即构建失败

> 本仓库的 `.ps1` 必须保存为 **UTF-8 with BOM**：Windows PowerShell 5.1 会把无 BOM
> 文件按 ANSI(GBK) 解码，中文注释会变成乱码并抛 `ParserError`。

---

## 依赖与 CUDA

- **Python 3.11**（`libleapct` / `xraylib` 的扩展都是 `cp311`）
- `pip install -r requirements.txt`
- **LEAP-CT / xraylib 不走 pip**：已在 `vendor/` 内附带，启动脚本会加进 `sys.path`
- **CUDA**：`libleapct.dll` 是 CUDA 版。按 PE 导入表实测，它直接依赖
  **`cufft64_11.dll`**（后者又依赖 `nvJitLink_120_0.dll` / `nvrtc64_120_0.dll`）
  与显卡驱动自带的 `nvcuda.dll`；**并不需要 cublas / cusparse**。
  这些库体积过大（cufft 单个约 274 MB，超过 GitHub 单文件 100 MB 限制），
  故仓库只带一个 `cudart64_12.dll`：
  - 源码运行：安装 CUDA 12 运行时并加入 `PATH`，或在程序中切 `set_GPU(-1)` 走 CPU
  - 打包：`scripts/collect_cuda.py` 会从 nvidia wheel 自动取齐，本机无需装 CUDA

---

## 排错

| 症状 | 原因 / 处理 |
| :--- | :--- |
| `ModuleNotFoundError: No module named 'PyQt5'` | 你拿到的是**旧原型**产物。本程序的界面是 PySide6 + PyCt6；PyQt5 版本已移到 `src/legacy/` 且不再构建 |
| 程序能启动，但显示 **"LEAP-CT 不可用"** | 打包时漏了 `--copy-metadata imageio --copy-metadata lazy_loader`。`imageio/__init__.py` 在 import 期读自己的 dist-info 取版本号，PyInstaller 默认不打 `.dist-info`，于是 `import leapctype` 抛 `PackageNotFoundError` |
| exe 图标是 PyInstaller 的默认图标 | 打包时没传 `--icon`。见 `scripts/make_icon.py` |
| CI：`ParserError: Missing expression after unary operator '--'` | 在 GitHub Windows runner 的 `run:`（默认 **PowerShell**）里用了 cmd 的 `^` 续行符。参数应写成数组或整行 |
| CI：`NativeCommandError` 紧跟 PyInstaller 第一行 INFO | PowerShell 5.1 在 `$ErrorActionPreference='Stop'` 下会把原生命令的 stderr 当终止错误。见脚本里的 `Invoke-Native` |
| 本地跑 `.ps1`：`Unexpected token '}'` | `.ps1` 丢了 UTF-8 BOM，被按 GBK 解码 |
| `import leapctype` 失败但文件都在 | 检查 `vendor/leapct` 是否作为**真实目录**打进了产物（`leapctype` 靠 `__file__` 定位 `libleapct.dll`，不能只在 PYZ 里） |
| Fermi 相关功能静默失效 | `Fermi/` 需作为真实目录随产物发布；`ct_fermi` 要把 `Fermi/` 与 `Fermi/src/` 加进 `sys.path` |

---

## 已知限制

- 螺旋 `helicalPitch` 为 **mm/rad**（LEAP 约定），代码内已按 `每圈进 mm / 2π` 换算
- 谱分解的碘浓度误差约 **6%**（逐像素比值标定后）；主要残差来自 FBP 域边缘伪影
- 静态多源提供"等效旋转锥束"与"模块束 + SART"两条路径；xraylib 化合物表缺失，
  改用元素加权构造 μ
- `src/legacy/` 的 PyQt5 版本仅作历史参考，不再构建、不再修复

---

## 许可

本项目自有代码以 **[MIT](LICENSE)** 发布，Copyright (c) 2026 hg3992260。

但本程序**分发了若干第三方二进制**，因此还需履行它们的许可义务，完整声明见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)：

| 组件 | 许可 | 关键义务 |
| :--- | :--- | :--- |
| **LEAP-CT** 1.26（`vendor/leapct/`） | MIT | 附版权与许可声明 |
| **xraylib** 4.3.0（`vendor/xraylib/`） | BSD-3-Clause | 附声明；**不得**以作者名义背书 |
| **Qt / PySide6** | LGPL-3.0 | 附声明；允许替换 / 重新链接 —— 本项目用 `--onedir` 打包以满足 |
| CUDA 运行时 | NVIDIA EULA | 构建时从官方 wheel 取，仓库不含该二进制（超过 GitHub 单文件 100 MB 限制） |
| 其余 Python 依赖 | MIT / BSD / PSF 系 | 附各自声明 |

> ⚠️ 因此**不要**把打包方式改成 `--onefile`：`--onedir` 让 Qt 的 `*.dll` / `*.pyd`
> 保持为 `_internal/` 下的独立文件，接收者可直接替换 —— 这是 LGPL-3.0 合规的关键。

打包产物内会一并附带 `LICENSE`、`THIRD_PARTY_NOTICES.md` 与 `licenses/` 目录
（各依赖自带的许可原文，由 `scripts/collect_licenses.py` 从已安装分发的元数据收集，
不靠手抄，保证与实际打包版本对应）。

---

## 致谢

- **[LEAP-CT](https://github.com/LLNL/LEAP)**（LLNL，MIT）—— 锥束 / 螺旋 / 模块束投影与重建算子
- **[xraylib](https://github.com/tschoonj/xraylib)**（BSD-3-Clause）—— 元素与化合物的 X 射线衰减截面
- **[PyCt6](https://pypi.org/project/PyCt6/)**（MIT）—— 主题化 PySide6 组件库
- **[Qt for Python (PySide6)](https://www.qt.io/)**（LGPL-3.0）—— 界面与 OpenGL 视口运行时
