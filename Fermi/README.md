# Fermi

## 本地运行

```powershell
conda run -n Fermi python I:\Fermi\src\gui.py
```

## GitHub 编译（Windows exe）

本仓库内置 GitHub Actions 工作流，会在每次 push 或手动触发时，自动用 PyInstaller 生成 Windows 可执行文件并作为 Artifact 上传。

下载路径：
GitHub 仓库页面 → Actions → 选择一次运行 → Artifacts → Fermi-windows

## GitHub 编译（macOS DMG）

工作流会在 macOS Runner 上打包 `Fermi.app` 并生成 `Fermi.dmg`，作为 Artifact 上传。

下载路径：
GitHub 仓库页面 → Actions → 选择一次运行 → Artifacts → Fermi-macos-dmg

macOS 运行提示：
未做 Apple 签名/公证（Notarization）的 DMG/App 可能会被 Gatekeeper 拦截。通常可通过“右键打开”放行；或在终端对解压后的 App 执行：

```bash
xattr -dr com.apple.quarantine /Applications/Fermi.app
```

## 图标

- 运行时窗口/任务栏图标：使用 `resources/Atom.jpeg`（程序内加载）
- exe 文件图标：使用 `resources/Atom.ico`（PyInstaller `--icon`）
- 如需重新生成 ico：

```powershell
conda run -n Fermi python tools\make_ico.py
```
