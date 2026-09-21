# 双源 CT 真实几何模拟器 (Cone Beam) — Native Edition

双源宽体 CT 的几何 / 数据 / 重建数字孪生。支持 5 种 CT 架构、LEAP-CT 物理重建、
半导体探测器响应（Fermi）与严格多材料谱分解，并提供 Agent 控制桥（MCP）。

## 目录结构

```
src/                   Python 源码
  simulate_ct6.py      主程序（PySide6 + PyCt6 界面）
  ct_geometry.py       几何内核（5 架构 / 探测器阵列整数化匹配 / Bowtie / 协议闭环）
  ct_leap.py           LEAP-CT 引擎（扇束 / 锥束 / 模块束+迭代 / 螺旋）
  ct_helical.py        螺旋 z 插值（360LI / 180LI / 180MF）
  ct_index.py          sinogram ↔ 探测器单元 ↔ 几何 解析索引 + z 重排
  ct_fermi.py          Fermi 半导体响应桥接（深度加权 CCE / 能量箱 / 双层）
  ct_spectral.py       严格多材料谱分解（逐能量正投影 + 非线性牛顿）
  ct6_bridge.py        应用内 TCP 控制桥
  ct6_mcp_server.py    MCP stdio 服务器（5 个工具）
  legacy/              早期 PyQt5 版本
assets/                logo / 图标 / 启动图
Fermi/                 半导体探测器响应模拟子项目（ct_fermi 依赖）
vendor/leapct/         LEAP-CT 1.26（leapctype.py + libleapct.dll 18MB）
vendor/xraylib/        xraylib 4.3.0（xraylib.py + 两个 pyd，29MB）
vendor/cuda/           CUDA 运行时（仅 cudart64_12.dll）
docs/                  界面截图
```

## 环境要求

- Python **3.11**（LEAP/xraylib 的 pyd 为 cp311）
- 依赖：`pip install -r requirements.txt`
- **LEAP-CT / xraylib 不走 pip**：已在 `vendor/` 内附带，启动脚本会加入 `sys.path`
- **CUDA**：`libleapct.dll` 为 CUDA 版。按 PE 导入表实测，它直接依赖 **`cufft64_11.dll`**
  （后者又依赖 `nvJitLink_120_0.dll` / `nvrtc64_120_0.dll`）与显卡驱动自带的 `nvcuda.dll`；
  **并不需要** cublas / cusparse。这些库体积过大（cufft 单个约 274MB），故未纳入仓库，
  `vendor/cuda/` 只随仓库带一个 `cudart64_12.dll`。
  - 源码运行：安装 CUDA 12 运行时并加入 PATH，或在程序中切 `set_GPU(-1)` 走 CPU。
  - GitHub Actions 打包：`windows-build.yml` 会自动从 nvidia wheel 取
    cufft / cudart / nvJitLink / nvrtc 打进产物，本机无需装 CUDA。

## 下载可执行文件

GitHub **Actions → `windows-build` → 最新一次 run → Artifacts → `DSW_CT_Windows`**，
解压后直接运行 `DSW_CT.exe`。

> ⚠️ 不要使用 `DualSourceCT_Simulator.exe`。那个产物来自已删除的 `build.yml`，
> 编译的是 `simulate_qt.py`（PyQt5 旧原型，现存于 `src/legacy/`）；而且 PyQt5
> 并不在 `requirements.txt` 里，所以它每次构建都是"绿色但缺模块"的废产物。

## 运行

```bat
run.bat
```
或
```bat
python src\simulate_ct6.py
```

## Agent 控制（MCP）

```json
"ct6": { "command": "<python>", "args": ["<repo>/src/ct6_mcp_server.py"] }
```
工具：`ct6_call` / `ct6_set_geometry` / `ct6_switch_arch` / `ct6_run_fbp` / `ct6_screenshot`。
前提：程序已启动（桥会写发现文件）。

## 已知限制

- 螺旋 `helicalPitch` 单位为 **mm/rad**（LEAP 约定），代码内已按 `每圈进 mm / 2π` 换算
- 谱分解的碘浓度误差 ~6%（逐像素比值标定后）；主要残差来自 FBP 域边缘伪影
- 静态多源采用等效旋转锥束 + SART 两条路径；xraylib 化合物表缺失，改用元素加权
