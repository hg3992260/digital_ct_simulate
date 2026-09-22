# -*- coding: utf-8 -*-
"""从 assets/LOGO.jpg 生成多尺寸 assets/logo.ico。

每次构建都重新生成，保证 logo.ico 永远和 LOGO.jpg 一致 —— 这正是"图标没同步为
LOGO.jpg"那类问题的根治办法（改了 LOGO.jpg 却忘了换 .ico）。

产物有两个用途，缺一不可：
  1. PyInstaller --icon  —— 决定 **exe 文件图标**（资源管理器 / 任务栏 / 开始菜单）。
     不传 --icon 时 PyInstaller 会嵌它自己的默认图标（icon-windowed.ico），
     这是"程序图标不是 LOGO"的真正原因。
  2. 程序运行时 setWindowIcon —— 窗口 / 任务栏图标。

注意：Windows 文件名大小写不敏感，所以这里写 logo.ico 而不是 LOGO.ico，
否则会和仓库里已有的 logo.ico 撞成同一个文件，徒增困惑。

LOGO.jpg 是 1024x1024 无透明通道的 RGB 图，转成 RGBA 后各尺寸重采样。
"""
import pathlib
import sys

# 输出一律用 ASCII：GitHub Windows runner 的控制台编码是 cp1252，
# 打印中文会抛 UnicodeEncodeError 把构建搞挂（注释保留中文没问题）。
try:
    sys.stdout.reconfigure(errors='replace')
except Exception:
    pass

SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def main():
    try:
        from PIL import Image
    except ImportError:
        print('[make_icon] requires Pillow: pip install pillow')
        return 1

    src = pathlib.Path('assets/LOGO.jpg')
    out = pathlib.Path('assets/logo.ico')
    if not src.exists():
        print('[make_icon] not found: %s' % src)
        return 1

    im = Image.open(src)
    print('[make_icon] source %s %s %s' % (src.name, im.mode, im.size))
    im.convert('RGBA').save(out, format='ICO', sizes=SIZES)

    chk = Image.open(out)
    print('[make_icon] wrote %s (%d bytes) sizes=%s'
          % (out.name, out.stat().st_size, sorted(chk.info.get('sizes', []))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
