# 第三方组件与许可声明

本仓库自身以 **MIT** 许可发布（见 [`LICENSE`](LICENSE)）。

但本程序**分发（redistribute）了若干第三方二进制**，因此除自有代码的 MIT 之外，
还必须履行下列许可义务。本文件即为这些声明；打包产物内也会带上一份
（`dist/DSW_CT/licenses/`，由 `scripts/collect_licenses.py` 从各包真实元数据收集，
不靠手抄）。

---

## 1. 直接随仓库/产物分发的二进制

这两项是本项目**实际打包进去的第三方二进制**，其许可要求随发行物提供完整声明。

### 1.1 LEAP-CT (`vendor/leapct/`)

- 组件：`libleapct.dll`（约 18 MB）、`leapctype.py`、`leap_filter_sequence.py`、
  `leap_preprocessing_algorithms.py`、`leaptorch.py`
- 版本：1.26
- 上游：<https://github.com/LLNL/LEAP>
- 许可：**MIT**
- 用途：锥束 / 扇束 / 螺旋 / 模块束的正投影与重建算子（本项目的物理引擎）

```
MIT License

Copyright (c) 2013-2023 LLNS, LLC and other LEAP Project Developers.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 1.2 xraylib (`vendor/xraylib/`)

- 组件：`_xraylib.cp311-win_amd64.pyd`、`xraylib_np.cp311-win_amd64.pyd`、
  `xraylib.py`（SWIG 生成）
- 版本：4.3.0
- 上游：<https://github.com/tschoonj/xraylib>
- 许可：**BSD 3-Clause**
- 用途：元素/化合物的 X 射线衰减截面（谱分解的 μ 表；缺失时回退内置 NIST 量级表）

> **注意 BSD-3 的第三条**：不得以原作者或贡献者的名义为衍生品背书或推广。
> 本项目及任何基于本项目的产品，均不得声称获得 xraylib 作者团队的认可。

```
Copyright (c) 2009, Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio,
Tom Schoonjans and Teemu Ikonen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote
      products derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez
del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom
Schoonjans and Teemu Ikonen BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.
```

### 1.3 CUDA 运行时（构建时收集，不随仓库分发）

`scripts/collect_cuda.py` 在构建时从 NVIDIA 官方 wheel
（`nvidia-cufft-cu12` / `nvidia-cuda-runtime-cu12` / `nvidia-nvjitlink-cu12` /
`nvidia-cuda-nvrtc-cu12`）取出 `cufft64_11.dll` 等运行时并打入产物。
它们受 **NVIDIA CUDA Toolkit End User License Agreement** 约束，
详见 <https://docs.nvidia.com/cuda/eula/index.html>。
仓库内不含这些文件（单个文件即超过 GitHub 的 100 MB 限制）。

---

## 2. Qt / PySide6 —— LGPL-3.0（需要特别留意）

### 2.1 声明

本程序使用 **Qt for Python (PySide6)**，其许可为
**LGPL-3.0-only（或 GPL-2.0-only / GPL-3.0-only）**。
本项目以 **LGPL-3.0** 选项使用之。

- 组件：`PySide6`（`Qt6Core.dll`、`Qt6Gui.dll`、`Qt6Widgets.dll`、
  `Qt6OpenGL*.dll`、`pyside6.abi3.dll`、各 `*.pyd` 与 Qt 插件）
- 版权：Copyright (C) The Qt Company Ltd. 及贡献者
- 上游：<https://www.qt.io/> / <https://code.qt.io/cgit/pyside/pyside-setup.git/>
- LGPL-3.0 全文：<https://www.gnu.org/licenses/lgpl-3.0.txt>

### 2.2 本项目的合规做法

LGPL-3.0 允许以本项目的 MIT 许可发布，前提是接收者能够**替换 / 重新链接**
其中的 LGPL 组件。本项目据此采取：

1. **打包使用 `--onedir`，不使用 `--onefile`。**
   产物为目录结构，Qt 的 `*.dll` / `*.pyd` 是 `_internal/` 下的**独立文件**，
   接收者可直接替换为自己编译或从上游取得的版本。

   > ⚠️ 因此**请勿把构建改成 `--onefile`**。单文件模式会把 Qt 全部塞进一个 exe，
    > 使"能否替换 LGPL 组件"变得含混，实质性提高合规风险。

2. 产物内随附 Qt 的版权与许可声明（`licenses/` 目录）。

3. 不对 Qt 做任何阻止替换的技术措施（无签名校验、无完整性自检、无加密加载）。

### 2.3 商业许可选项

若你希望规避 LGPL 义务（例如要静态链接 Qt、或法律上不接受 LGPL 条款），
可向 The Qt Company 购买商业许可。

---

## 3. 其余随产物打包的第三方 Python 包

以下均为**宽松许可**（MIT / BSD / PSF 系），与 MIT 兼容。各自完整的许可文本由
`scripts/collect_licenses.py` 在构建时从已安装包的元数据中收集并放入产物的
`licenses/` 目录。

| 包 | 用途 | 许可 |
| :--- | :--- | :--- |
| `PyCt6` | 主题化 PySide6 组件库（界面） | MIT |
| `pyqtgraph` | 二维/三维绘图与 GL 视口 | MIT |
| `PyOpenGL` | OpenGL 绑定 | BSD |
| `numpy` | 数组与数值基础 | BSD 3-Clause |
| `scipy` | 科学计算（滤波、优化、插值） | BSD 3-Clause |
| `scikit-image` | Radon / 图像变换 | BSD 3-Clause |
| `matplotlib` | 曲线绘制 | PSF-based（BSD 兼容） |
| `imageio` | 图像读写（LEAP 依赖） | BSD 2-Clause |
| `Pillow` | 图像编解码 | MIT-CMU |
| `lazy_loader` | scikit-image 的延迟导入 | BSD 3-Clause |
| `networkx` / `tifffile` / `packaging` / `python-dateutil` / `contourpy` / `fonttools` / `kiwisolver` / `cycler` / `pyparsing` | 上述包的传递依赖 | 各自 BSD/MIT/Apache 系 |

> **关于 PyOpenGL**：其 wheel 元数据只声明了
> `License :: OSI Approved :: BSD License`，**不自带许可文件**，所以
> `collect_licenses.py` 无法从元数据中提取到原文。上游项目页：
> <https://mcfletch.github.io/pyopengl/>（BSD）。

Fermi 子项目（`Fermi/`）自带其独立的 `requirements.txt`，其中 `xraylib` 与
`matplotlib` 的许可同上；其 GUI 部分依赖 PyQt6，**不参与本程序打包**
（`ct_fermi` 只使用其中的 `simulator` / `material` / `physics` 三个模块）。

---

## 4. 汇总

| 类别 | 许可 | 义务 |
| :--- | :--- | :--- |
| 本项目自有代码 | MIT | 附版权与许可声明 |
| LEAP-CT | MIT | 附版权与许可声明 |
| xraylib | BSD-3-Clause | 附声明；**不得**以作者名义背书 |
| Qt / PySide6 | LGPL-3.0 | 附声明；允许替换/重链接（本项目用 `--onedir` 满足） |
| CUDA 运行时 | NVIDIA EULA | 附 EULA 指引；仓库不含该二进制 |
| 其余 Python 依赖 | MIT / BSD / PSF 系 | 附各自声明 |
