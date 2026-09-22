#Requires -Version 5.1
<#
  DSW_CT Windows 打包脚本（onedir）。

  ** 本文件必须保存为 UTF-8 with BOM **
  Windows PowerShell 5.1 会把无 BOM 的文件按 ANSI(GBK) 解码，下面的中文注释会变成
  乱码并直接抛 ParserError: Unexpected token '}'。PowerShell 7 / GitHub runner 不受
  影响，但本地 5.1 会挂，所以务必保留 BOM。

  为什么构建逻辑在仓库脚本里而不在 workflow 里：
  workflow 文件受 GitHub 'workflow' scope 保护，改一行都要有人去网页端操作；
  scripts/ 下的普通文件 push 即可生效。workflow 只负责
  "装依赖 -> 调本脚本 -> 传产物"。

  本地复现：
      pwsh -File scripts/build_windows.ps1
      pwsh -File scripts/build_windows.ps1 -SkipCuda     # 跳过拉取 CUDA wheel
#>
param(
    [switch]$SkipCuda
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
Write-Host "=== repo root: $Root ===" -ForegroundColor Cyan


function Invoke-Native {
    <#
      运行原生命令，并把 stderr 当普通输出处理。

      坑：PowerShell 5.1 在 $ErrorActionPreference='Stop' 下，会把原生命令写到
      stderr 的**每一行**当成终止错误。PyInstaller 的 INFO 日志恰好走 stderr，
      于是构建还没开始就 NativeCommandError 挂掉。所以这里必须临时降级，
      并显式回传退出码。
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$Arguments = @()
    )
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Exe @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
    }
    finally {
        $ErrorActionPreference = $old
    }
    return $LASTEXITCODE
}


# ---------------------------------------------------------------------------
# 1. CUDA 12 运行时
# ---------------------------------------------------------------------------
if (-not $SkipCuda) {
    Write-Host "`n=== [1/4] 安装 CUDA 12 wheel ===" -ForegroundColor Cyan
    $code = Invoke-Native 'python' @(
        '-m', 'pip', 'install', '--quiet',
        'nvidia-cufft-cu12', 'nvidia-cuda-runtime-cu12',
        'nvidia-nvjitlink-cu12', 'nvidia-cuda-nvrtc-cu12'
    )
    if ($code -ne 0) { exit $code }
}

Write-Host "`n=== [1/4] 收集 CUDA 运行时到 vendor/cuda ===" -ForegroundColor Cyan
$code = Invoke-Native 'python' @('scripts/collect_cuda.py')
if ($code -ne 0) { exit $code }

# ---------------------------------------------------------------------------
# 2. 图标：LOGO.jpg -> LOGO.ico（exe 文件图标 + 窗口图标）
# ---------------------------------------------------------------------------
Write-Host "`n=== [2/4] 生成 assets/logo.ico ===" -ForegroundColor Cyan
$code = Invoke-Native 'python' @('scripts/make_icon.py')
if ($code -ne 0) { exit $code }
if (-not (Test-Path 'assets/logo.ico')) { throw 'assets/logo.ico 生成失败，无法设置 exe 图标' }

# ---------------------------------------------------------------------------
# 3. PyInstaller
#    参数写成数组，彻底避免 cmd 的 ^ / PowerShell 反引号续行问题
#    （最初 CI 就是死在 pwsh 里用了 cmd 的 ^ -> ParserError）。
#
#    几个"文件在但 import 不动"的坑，都靠下面的参数兜住：
#      --add-data src/simulate_ct6.py
#          frozen_entry 用 runpy.run_path 动态加载主程序，PyInstaller 静态分析看不到它，
#          不打包就会运行即 FileNotFoundError。
#      --add-data vendor/leapct | vendor/xraylib | vendor/cuda
#          leapctype 靠 __file__ 定位同目录的 libleapct.dll，所以必须是**真实文件**，
#          不能只塞进 PYZ；而且 leapct / xraylib 都不来自 requirements.txt。
#      --copy-metadata imageio | lazy_loader
#          imageio 在 __init__ 里读自己的 dist-info 取版本号，lazy_loader 同理
#          （skimage 依赖它）。漏掉的表现就是"exe 能启动，但 LEAP-CT 不可用"：
#          文件明明都在，import leapctype 却抛 PackageNotFoundError。
#      --exclude-module torch
#          leapctype 的可选后端，约 2GB，不需要。
#      --icon assets/logo.ico
#          不传就会把 PyInstaller 的默认图标嵌进 exe。logo.ico 由 scripts/make_icon.py
#          从 assets/LOGO.jpg 现生成，保证 exe 图标永远和 LOGO.jpg 一致。
# ---------------------------------------------------------------------------
Write-Host "`n=== [3/4] PyInstaller 打包 ===" -ForegroundColor Cyan
$pyiArgs = @(
    '--noconfirm', '--clean', '--windowed', '--onedir',
    '--name', 'DSW_CT',
    '--icon', 'assets/logo.ico',
    '--paths', 'src',
    '--add-data', 'assets;assets',
    '--add-data', 'Fermi;Fermi',
    '--add-data', 'src/simulate_ct6.py;.',
    '--add-data', 'vendor/leapct;vendor/leapct',
    '--add-data', 'vendor/xraylib;vendor/xraylib',
    '--add-data', 'vendor/cuda;vendor/cuda',
    '--collect-all', 'PyCt6',
    '--copy-metadata', 'imageio',
    '--copy-metadata', 'lazy_loader',
    '--exclude-module', 'torch',
    '--hidden-import', 'simulate_ct6',
    '--hidden-import', 'ct_geometry',
    '--hidden-import', 'ct_leap',
    '--hidden-import', 'ct_scene',
    '--hidden-import', 'ct_helical',
    '--hidden-import', 'ct_index',
    '--hidden-import', 'ct_fermi',
    '--hidden-import', 'ct_spectral',
    '--hidden-import', 'ct6_bridge',
    '--hidden-import', 'ct6_mcp_server',
    '--hidden-import', 'pyqtgraph',
    '--hidden-import', 'skimage',
    '--hidden-import', 'OpenGL',
    '--hidden-import', 'matplotlib',
    '--hidden-import', 'imageio',
    '--hidden-import', 'scipy',
    '--hidden-import', 'scipy.ndimage',
    'src/frozen_entry.py'
)

$code = Invoke-Native 'pyinstaller' $pyiArgs
if ($code -ne 0) { exit $code }

# CUDA DLL 同时放 exe 同级目录（Windows 优先搜索 exe 所在目录）
# 与 _internal/vendor/cuda（frozen_entry 会用 os.add_dll_directory 加进去），双保险。
if (Test-Path 'vendor/cuda') {
    Copy-Item vendor/cuda/*.dll dist/DSW_CT/ -Force -ErrorAction SilentlyContinue
}
Copy-Item src/simulate_ct6.py dist/DSW_CT/ -Force

Write-Host "`n--- dist/DSW_CT ---"
Get-ChildItem dist/DSW_CT | Select-Object Name, Length | Format-Table -AutoSize

# ---------------------------------------------------------------------------
# 4. 冻结环境自检
#    在**打包产物内部**真的 import 一遍关键模块，并真的构造一次 LEAP 引擎。
#    "文件都在"不等于"import 得动"——imageio 元数据那类问题只有这里挡得住。
# ---------------------------------------------------------------------------
Write-Host "`n=== [4/4] 冻结环境自检 ===" -ForegroundColor Cyan
$exe = Join-Path $Root 'dist/DSW_CT/DSW_CT.exe'
if (-not (Test-Path $exe)) { throw "构建产物缺少 $exe" }

$env:DSW_CT_SELFTEST = '1'
$env:DSW_CT_SELFTEST_OUT = Join-Path $Root 'selftest.json'
$env:QT_QPA_PLATFORM = 'offscreen'
$code = Invoke-Native $exe @()
Remove-Item Env:\DSW_CT_SELFTEST -ErrorAction SilentlyContinue

if (Test-Path $env:DSW_CT_SELFTEST_OUT) {
    Get-Content $env:DSW_CT_SELFTEST_OUT -Raw | Write-Host
}
else {
    Write-Host '!! 自检没有产出报告文件' -ForegroundColor Red
    exit 1
}

if ($code -ne 0) {
    Write-Host "!! 冻结自检失败 (exit $code) —— 产物不可用，中止" -ForegroundColor Red
    exit $code
}
Write-Host "`n=== 构建完成，冻结自检通过 ===" -ForegroundColor Green
