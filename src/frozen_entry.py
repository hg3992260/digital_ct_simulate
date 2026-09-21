# -*- coding: utf-8 -*-
"""PyInstaller 打包入口：先把 vendor/_MEIPASS 加入 DLL 搜索路径，再启动主程序。

冻结运行(libreapct.dll / xraylib / CUDA 运行时)必须能被 ctypes/CDLL 找到；
Windows 下除了 PATH 还需 os.add_dll_directory。
"""
import os
import sys


def _setup_dll_paths():
    base = getattr(sys, '_MEIPASS', None) or os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    cands = [base,
             os.path.join(base, 'vendor'),
             os.path.join(base, 'vendor', 'leapct'),
             os.path.join(base, 'vendor', 'xraylib'),
             os.path.join(base, 'vendor', 'cuda')]
    for p in cands:
        if os.path.isdir(p):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass
            if p not in sys.path:
                sys.path.insert(0, p)
    os.environ['PATH'] = os.pathsep.join(
        [p for p in cands if os.path.isdir(p)] + [os.environ.get('PATH', '')])


_setup_dll_paths()

if __name__ == '__main__':
    import runpy
    here = os.path.dirname(os.path.abspath(__file__))
    runpy.run_path(os.path.join(here, 'simulate_ct6.py'), run_name='__main__')
