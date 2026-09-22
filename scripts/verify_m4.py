# -*- coding: utf-8 -*-
"""M4 验收：K 帧亚像素过采样采集（手册 §8 M4 判据）。

  判据 1：K=1/2/4 的 MTF 主瓣零点 = min(f_N(K), f₀)
  判据 2：K=2 与 K=4 的 MTF 差异落在噪声内（验证 §4.1 的饱和结论）

判据 2 用**严格带限**物体，并同时给出宽带锐边物体的对照值 —— 后者大了一个
数量级，但那是孔径在 f₀ 以上的残余传递在 K=2 的 Nyquist(=f₀) 处混叠，
不是"K=2 未饱和"。
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
    print('  [%s] %-44s %s' % ('OK ' if good else 'FAIL', tag, detail))


print('=== 亚像素位移模式 δ_k ===')
off = F.subpixel_offsets(4)
check('linear: δ_k = k/K', np.allclose(off, [0.0, 0.25, 0.5, 0.75]), str(off))
oc = F.subpixel_offsets(4, 'centered')
check('centered 以 0 对称', abs(oc.sum()) < 1e-12, str(np.round(oc, 3)))

print()
print('=== K 帧投影栈 ===')
g = F.m1_geometry(100.0, 300.0, 55.0, 256, 1, 16, 180.0)
p = g['p_eff_mm']
ang = np.linspace(0.0, 180.0, 16, endpoint=False)
acq = F.acquire_k_frames(F.slit_phantom(p / 8.0), ang, 256, p, K=4,
                         scale_mm=1.0, mu_scale=1.0, n_sub=16)
check('frames 形状 (K, n_ang, n_cols)', acq['frames'].shape == (4, 16, 256),
      str(acq['frames'].shape))
check('合成栅距 = p_eff/K', abs(acq['p_dense_mm'] - p / 4) < 1e-15,
      '%.4f µm' % (acq['p_dense_mm'] * 1000))
check('f_N(K=4) = 2f₀', abs(acq['f_nyq_K'] - 2.0 / p) < 1e-9, '%.2f' % acq['f_nyq_K'])

print()
print('=== 判据 1：主瓣零点 = min(f_N(K), f₀) ===')
r = F.validate_m4(n_cols=256, pixel_um=55.0, sod_mm=100.0, odd_mm=300.0,
                  n_angles=16, Ks=(1, 2, 4), fine=32)
print('  %-3s %9s %11s %13s %9s' % ('K', 'f_N(K)', '理论 f_eff', '实测零点', '相对差'))
for K in r['K_list']:
    d = r['per_K'][K]
    print('  %-3d %9.2f %11.2f %13.2f %8.1f%%'
          % (K, d['f_nyq_K'], d['f_theory_cyc_mm'], d['f_meas_null_cyc_mm'],
             100 * abs(d['f_meas_null_cyc_mm'] - d['f_theory_cyc_mm']) / d['f_theory_cyc_mm']))
    check('K=%d 零点吻合' % K,
          abs(d['f_meas_null_cyc_mm'] - d['f_theory_cyc_mm']) <= 0.15 * d['f_theory_cyc_mm'],
          '%.2f vs %.2f cyc/mm' % (d['f_meas_null_cyc_mm'], d['f_theory_cyc_mm']))

print()
print('=== 判据 2：K=2 与 K=4 饱和 ===')
check('带限物体上 K=2≈K=4', r['crit2_saturated_bandlimited'],
      '最大差 %.6f' % r['k2_vs_k4_bandlimited'])
print('       （对照）宽带锐边物体: %.6f —— 这是孔径 f₀ 以上残余传递的混叠，'
      '非未饱和' % r['k2_vs_k4_broadband'])

print()
print('通过 %d / 失败 %d' % (ok, fail))
_sys.exit(1 if fail else 0)
