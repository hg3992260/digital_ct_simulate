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


if __name__ == '__main__':
    import runpy

    _main = _find_main_script()
    if _main is None:
        raise SystemExit(
            '[FATAL] 找不到 simulate_ct6.py，无法启动。\n'
            '        打包构建请确认已加 --add-data "src/simulate_ct6.py;."\n'
            '        搜索过的目录：%s' % (_candidate_dirs(),))
    runpy.run_path(_main, run_name='__main__')
