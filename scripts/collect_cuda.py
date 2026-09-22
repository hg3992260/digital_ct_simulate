# -*- coding: utf-8 -*-
"""把 nvidia wheel 里的 CUDA 运行时 DLL 收集到 vendor/cuda。

为什么需要这些库：
    libleapct.dll 的 PE 导入表只直接依赖 cufft64_11.dll，而 cufft 又依赖
    nvJitLink_120_0.dll / nvrtc64_120_0.dll（以及显卡驱动自带的 nvcuda.dll）。
    cublas / cusparse / cusolver 完全用不到，但单个文件就有数百 MB，故不收集。

为什么放仓库而不是 workflow：workflow 受 GitHub 'workflow' scope 保护，
改一行都要网页端操作；普通文件 push 即可生效。
"""
import pathlib
import shutil
import sys

# 输出一律用 ASCII：GitHub Windows runner 的控制台编码是 cp1252，
# 打印中文会抛 UnicodeEncodeError 把构建搞挂（注释保留中文没问题）。
try:
    sys.stdout.reconfigure(errors='replace')
except Exception:
    pass

# 前缀匹配；注意 'cufftw' 必须在 'cufft' 之前不冲突（这里用 startswith 逐个判断，无顺序问题）
PREFIXES = ('cufft', 'cufftw', 'cudart', 'nvjitlink', 'nvrtc')

# nvrtc wheel 里带的备用副本，用不到，排除掉省 ~86MB
SKIP = {'nvrtc64_120_0.alt.dll'}


def main():
    src = pathlib.Path(sys.prefix) / 'Lib' / 'site-packages' / 'nvidia'
    dst = pathlib.Path('vendor/cuda')
    dst.mkdir(parents=True, exist_ok=True)

    if not src.is_dir():
        print('[collect_cuda] not found: %s -- pip install nvidia-cufft-cu12 etc. first' % src)
        return 1

    copied = []
    for f in src.rglob('*.dll'):
        low = f.name.lower()
        if low in SKIP:
            continue
        if any(low.startswith(p) for p in PREFIXES):
            shutil.copy2(f, dst / f.name)
            copied.append(f.name)

    have = sorted(x.name for x in dst.glob('*.dll'))
    print('[collect_cuda] %d gathered, vendor/cuda now: %s' % (len(copied), have))

    missing = [n for n in ('cufft64_11.dll',) if n not in have]
    if missing:
        print('[collect_cuda] missing required library: %s -- libleapct.dll will fail to load'
              % missing)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
