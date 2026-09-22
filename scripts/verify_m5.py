# -*- coding: utf-8 -*-
"""M5 验收：重建与去卷积链（手册 §8 M5 判据）。

  判据 1（基准一致性）：K 帧路径的重建与"解析直接密采样后重建"的基准一致
  判据 2（分辨率）：端到端重建体的 MTF 10% 截止，与**同口径**的理论系统 MTF
        （焦点圆盘 × 像素孔径）一致

判据 2 务必同口径：r(M)=√(f²(M−1)²+p²)/M 是 PSF 宽度类量，直接与频域 MTF 截止
相比会得到一个恒定比例（实测约 0.42×），看着像误差其实只是定义不同。
"""
import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / 'src'))
_sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np                                            # noqa: E402

import ct_forward as F                                       # noqa: E402

ok = fail = 0


def check(tag, good, detail=''):
    global ok, fail
    ok += good
    fail += (not good)
    print('  [%s] %-40s %s' % ('OK ' if good else 'FAIL', tag, detail))


print('=== 重建栅格随过采样细化 ===')
g = F.m1_geometry(20.0, 780.0, 55.0, 256, 1, 180, 180.0)
p_eff = g['p_eff_mm']
ang = np.linspace(0.0, 180.0, 180, endpoint=False)
ell = F.slit_phantom(p_eff / 6.0)
vox = {}
for K in (1, 2, 4):
    acq = F.acquire_k_frames(ell, ang, 256, p_eff, K=K, scale_mm=1.0, mu_scale=1.0, n_sub=16)
    rec, dense = F.reconstruct_k(acq, ang, deconvolve=False, focus_um=3.0, M=g['M'],
                                 pixel_um=55.0, focus_blur=True)
    vox[K] = (acq['p_dense_mm'] * 1000.0, rec.shape)
    print('  K=%d  密栅距 %6.3f µm  重建尺寸 %s' % (K, vox[K][0], vox[K][1]))
check('K=1 重建栅距 = p_eff', abs(vox[1][0] - p_eff * 1000) < 1e-6, '%.3f µm' % vox[1][0])
check('K=4 重建栅距 = p_eff/4', abs(vox[4][0] - p_eff * 1000 / 4) < 1e-6, '%.3f µm' % vox[4][0])

print()
print('=== 判据 1 / 判据 2（随焦点扫描）===')
print('  %-6s %6s %11s %11s %10s %8s' % ('f µm', 'M', '理论MTF10', '实测MTF10', '去卷积后', 'corr'))
for f in (1.0, 2.0, 3.0, 5.0, 10.0):
    r = F.validate_m5(focus_um=f, pixel_um=55.0, K=4, n_cols=256, n_angles=180)
    print('  %-6.1f %6.1f %11.2f %11.2f %10.2f %8.4f'
          % (f, r['M'], r['res_theory_mtf10_um'], r['res_raw_um'],
             r['res_deconv_um'], r['corr_vs_reference']))
    check('f=%.0f µm 基准一致' % f, r['crit1_baseline_match'],
          'corr %.4f' % r['corr_vs_reference'])
    check('f=%.0f µm 分辨率同口径吻合' % f, r['crit2_resolution_in_band'],
          '%.2f vs 理论 %.2f µm' % (r['res_deconv_um'], r['res_theory_mtf10_um']))

print()
print('=== 分辨率随焦点单调 ===')
res = [F.validate_m5(focus_um=f, pixel_um=55.0, K=4, n_cols=256, n_angles=180)['res_deconv_um']
       for f in (1.0, 3.0, 5.0, 10.0)]
check('f 增大 → 分辨率变差（单调）', all(res[i] < res[i + 1] for i in range(len(res) - 1)),
      ' → '.join('%.2f' % x for x in res) + ' µm')

print()
print('通过 %d / 失败 %d' % (ok, fail))
_sys.exit(1 if fail else 0)
