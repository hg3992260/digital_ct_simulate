# -*- coding: utf-8 -*-
"""把各第三方包**真实的**许可文件收集进打包产物。

为什么不手抄全文：numpy/scipy/scikit-image 等的 LICENSE 长度可观且随版本变化，
手抄必然过期或出错。这里直接从已安装分发的 dist-info 元数据里取原始文件，
保证与产物内实际打包的版本严格对应。

输出：
    dist/DSW_CT/LICENSE                    本项目自己的 MIT
    dist/DSW_CT/THIRD_PARTY_NOTICES.md     汇总声明（含 LEAP-CT / xraylib / Qt 全文）
    dist/DSW_CT/licenses/<包名>/...        各依赖自带的许可原文
"""
import importlib.metadata as md
import pathlib
import shutil
import sys

# 需要随产物附带许可声明的第三方分发（用 PyPI 分发名）
DISTS = [
    'PySide6', 'PyCt6', 'pyqtgraph', 'PyOpenGL',
    'numpy', 'scipy', 'scikit-image', 'matplotlib', 'imageio', 'pillow',
    'lazy_loader', 'networkx', 'tifffile', 'packaging', 'python-dateutil',
    'contourpy', 'fonttools', 'kiwisolver', 'cycler', 'pyparsing',
]

LICENSE_GLOBS = ('LICENSE*', 'LICENCE*', 'COPYING*', 'NOTICE*', 'AUTHORS*')

REPO_FILES = ('LICENSE', 'THIRD_PARTY_NOTICES.md')


def main():
    repo = pathlib.Path('.')
    out = pathlib.Path('dist/DSW_CT')
    if not out.is_dir():
        print('[collect_licenses] 产物目录不存在：%s（请先跑 PyInstaller）' % out)
        return 1

    lic_root = out / 'licenses'
    lic_root.mkdir(parents=True, exist_ok=True)

    # 1. 本项目自己的许可与声明
    for name in REPO_FILES:
        src = repo / name
        if src.exists():
            shutil.copy2(src, out / name)
            print('[collect_licenses] + %s' % name)

    # 2. 各依赖的原始许可文件
    got, missing = [], []
    for dist_name in DISTS:
        try:
            dist = md.distribution(dist_name)
        except md.PackageNotFoundError:
            missing.append(dist_name)
            continue

        di = pathlib.Path(getattr(dist, '_path', '') or '')
        if not di.is_dir():
            missing.append(dist_name)
            continue

        dest = lic_root / dist_name
        dest.mkdir(parents=True, exist_ok=True)
        n = 0

        candidates = []
        for g in LICENSE_GLOBS:
            candidates += list(di.glob(g))
        sub = di / 'licenses'
        if sub.is_dir():
            candidates += [p for p in sub.rglob('*') if p.is_file()]

        for f in candidates:
            if not f.is_file():
                continue
            try:
                rel = f.relative_to(di)
            except ValueError:
                rel = pathlib.Path(f.name)
            target = dest / str(rel).replace('\\', '_').replace('/', '_')
            if not target.exists():
                shutil.copy2(f, target)
                n += 1

        ver = dist.version
        if n:
            got.append('%s %s (%d files)' % (dist_name, ver, n))
        else:
            missing.append('%s %s (无许可文件)' % (dist_name, ver))

    print('[collect_licenses] 已收集 %d 个分发：' % len(got))
    for g in got:
        print('    + %s' % g)
    if missing:
        print('[collect_licenses] 跳过 %d 个：%s' % (len(missing), ', '.join(missing)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
