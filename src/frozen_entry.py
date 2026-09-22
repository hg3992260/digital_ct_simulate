# -*- coding: utf-8 -*-
"""PyInstaller 打包入口：先把 vendor/_MEIPASS 加入 DLL 搜索路径，再启动主程序。

冻结运行(libleapct.dll / xraylib / CUDA 运行时)必须能被 ctypes/CDLL 找到；
Windows 下除了 PATH 还需 os.add_dll_directory。

注意：主程序是用 runpy.run_path **动态**加载的，PyInstaller 的静态分析看不到它。
因此构建时必须加 --add-data "src/simulate_ct6.py;." 把源文件一起打进去，
否则运行时会 FileNotFoundError。本文件会在多个候选目录里查找它。
"""
import os
import sys


def _candidate_dirs():
    """所有可能的资源根目录：打包 _MEIPASS / exe 同级 / 模块目录 / 上一级。"""
    dirs = []
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        dirs.append(meipass)
    if getattr(sys, 'frozen', False):
        dirs.append(os.path.dirname(os.path.abspath(sys.executable)))
    here = os.path.dirname(os.path.abspath(__file__))
    dirs.append(here)
    dirs.append(os.path.dirname(here))

    out = []
    for d in dirs:
        if d and os.path.isdir(d) and d not in out:
            out.append(d)
    return out


def _setup_dll_paths():
    for root in _candidate_dirs():
        for p in (root,
                  os.path.join(root, 'vendor'),
                  os.path.join(root, 'vendor', 'leapct'),
                  os.path.join(root, 'vendor', 'xraylib'),
                  os.path.join(root, 'vendor', 'cuda')):
            if os.path.isdir(p):
                try:
                    os.add_dll_directory(p)
                except Exception:
                    pass
                if p not in sys.path:
                    sys.path.insert(0, p)
    real = [p for p in _candidate_dirs()]
    os.environ['PATH'] = os.pathsep.join(real + [os.environ.get('PATH', '')])


_setup_dll_paths()


def _find_main_script():
    for root in _candidate_dirs():
        p = os.path.join(root, 'simulate_ct6.py')
        if os.path.isfile(p):
            return p
    return None


# --------------------------------------------------------------------------
# 冻结环境自检（CI 用）：DSW_CT_SELFTEST=1 时不启动 GUI，只在**打包产物内部**
# 逐个 import 关键模块并真正构造一次 LEAP 引擎，把结果写成 JSON，用退出码表态。
#
# 存在的理由：PyInstaller 只保证"文件在"，不保证"import 得动"。例如 imageio 与
# lazy_loader 在 import 期要读自己的 dist-info 元数据，若构建时漏了
# --copy-metadata，文件明明都在、exe 也能启动，但 leapctype 一导入就抛
# PackageNotFoundError，LEAP-CT 直接不可用。只有真的在冻结环境里 import 才测得出来。
# --------------------------------------------------------------------------
_SELFTEST_MODULES = [
    'numpy', 'scipy', 'imageio', 'skimage', 'matplotlib', 'pyqtgraph', 'OpenGL',
    'PySide6.QtWidgets', 'PyCt6',
    'leapctype', 'xraylib',
    'ct_geometry', 'ct_leap', 'ct_scene', 'ct_helical', 'ct_index',
    'ct_fermi', 'ct_spectral', 'ct6_bridge',
]


def _selftest():
    import json
    import traceback

    report = {
        'frozen': getattr(sys, 'frozen', False),
        'meipass': getattr(sys, '_MEIPASS', None),
        'imports': {},
        'failures': {},
    }
    ok = True
    for name in _SELFTEST_MODULES:
        try:
            __import__(name)
            report['imports'][name] = 'OK'
        except Exception as exc:
            ok = False
            report['imports'][name] = '%s: %s' % (type(exc).__name__, exc)
            report['failures'][name] = traceback.format_exc()

    # LEAP：不只看 import，还要真的加载 libleapct.dll 并构造引擎（走 CPU，避免依赖显卡）
    try:
        import ct_leap as CL
        report['leap_available'] = bool(CL.LEAP_AVAILABLE)
        report['leap_import_error'] = CL.LEAP_IMPORT_ERROR
        if CL.LEAP_AVAILABLE:
            CL.LeapEngine(gpu_index=-1)
            report['leap_engine'] = 'constructed on CPU'
        else:
            ok = False
            report['leap_engine'] = 'unavailable: %s' % CL.LEAP_IMPORT_ERROR
    except Exception as exc:
        ok = False
        report['leap_engine'] = '%s: %s' % (type(exc).__name__, exc)
        report['failures']['leap_engine'] = traceback.format_exc()

    report['ok'] = ok
    out = os.environ.get('DSW_CT_SELFTEST_OUT')
    if out:
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
    return 0 if ok else 1


if __name__ == '__main__':
    if os.environ.get('DSW_CT_SELFTEST'):
        raise SystemExit(_selftest())

    import runpy

    _main = _find_main_script()
    if _main is None:
        raise SystemExit(
            '[FATAL] 找不到 simulate_ct6.py，无法启动。\n'
            '        打包构建请确认已加 --add-data "src/simulate_ct6.py;."\n'
            '        搜索过的目录：%s' % (_candidate_dirs(),))
    runpy.run_path(_main, run_name='__main__')
