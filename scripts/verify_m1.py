# -*- coding: utf-8 -*-
"""M1 验收：正向投影器 vs 解析解（手册 §8 判据：误差 < 1%）。

判据分三层：
  1. 解析弦长本身要对（圆截面 vs 教科书 2√(R²−s²)；旋转椭圆 vs 数值积分）
  2. 数值正向投影 vs 解析正弦图，误差随网格**一阶收敛**
  3. 手册点名要求的圆柱 / Shepp-Logan 在合理网格下 < 1%
"""
import sys
import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / 'src'))
_sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np


import ct_forward as F                                        # noqa: E402

ok = fail = 0


def check(tag, good, detail=''):
    global ok, fail
    ok += good
    fail += (not good)
    print('  [%s] %-46s %s' % ('OK ' if good else 'FAIL', tag, detail))


print('=== 1. 解析弦长自检 ===')
R = 5.0
s = np.linspace(-R * 0.98, R * 0.98, 21)
mine = F.ellipse_chord(s, R, R, 0.0)
text = 2.0 * np.sqrt(np.maximum(R ** 2 - s ** 2, 0.0))
check('圆截面 vs 2√(R²−s²)', np.abs(mine - text).max() < 1e-9,
      '最大差 %.2e' % np.abs(mine - text).max())

# 旋转椭圆：闭式 vs 直接沿射线密集采样的数值弦长
import math
worst = 0.0
for phi in (0.0, 30.0, -18.0, 75.0):
    a, b, ss, th = 3.0, 1.0, 0.4, 0.0
    ch = float(F.ellipse_chord(np.array([ss]), a, b, th - phi)[0])
    t = np.linspace(-50, 50, 2000001)
    ct, st = math.cos(math.radians(th)), math.sin(math.radians(th))
    nx, ny = -st, ct
    x, y = ss * nx + t * ct, ss * ny + t * st
    cp, sp = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    u, v = x * cp + y * sp, -x * sp + y * cp
    idx = np.nonzero((u / a) ** 2 + (v / b) ** 2 <= 1.0)[0]
    num = (t[idx[-1]] - t[idx[0]]) if len(idx) else 0.0
    worst = max(worst, abs(ch - num))
check('旋转椭圆闭式 vs 数值积分', worst < 5e-3, '最大差 %.2e' % worst)

print()
print('=== 2. 数值正向投影 vs 解析（收敛性）===')
print('  %-12s %6s %9s %9s %10s %10s  %s'
      % ('体模', '网格', '体素µm', '最大%', 'P99%', '平均%', '判定'))
for name in ('cylinder', 'shepp_logan'):
    prev = None
    for ng in (1024, 2048):
        r = F.validate_m1(name=name, n_grid=ng, n_cols=256, pixel_um=55.0,
                          sod_mm=100.0, odd_mm=300.0, n_angles=60, sample_mm=10.0, aa=2)
        print('  %-12s %6d %9.1f %8.3f%% %9.3f%% %9.4f%%  %s'
              % (name, ng, r['voxel_um'], r['rel_err_max'] * 100,
                 r['rel_err_p99'] * 100, r['rel_err_mean'] * 100,
                 'PASS' if r['passed'] else 'FAIL'))
        if prev is not None:
            check('%s 误差随网格一阶收敛' % name,
                  r['rel_err_max'] < prev * 0.65, '%.3f%% → %.3f%%' % (prev * 100, r['rel_err_max'] * 100))
        prev = r['rel_err_max']

print()
print('=== 3. 手册判据（< 1%）===')
for name in ('cylinder', 'shepp_logan'):
    r = F.validate_m1(name=name, n_grid=2048, n_cols=256, pixel_um=55.0,
                      sod_mm=100.0, odd_mm=300.0, n_angles=60, sample_mm=10.0, aa=2)
    check('%s @2048' % name, r['passed'], '最大 %.3f%% / 平均 %.4f%%'
          % (r['rel_err_max'] * 100, r['rel_err_mean'] * 100))

print()
print('=== 4. 投影几何输出（M1 第二项交付）===')
g = F.m1_geometry(100.0, 300.0, 55.0, 256, 1, 180)
for k in ('M', 'p_eff_um', 'fov_mm', 'grid_pitch_um', 'nyquist_cyc_mm', 'n_angles', 'ang_step_deg'):
    print('     %-16s %s' % (k, g[k]))
check('M = (SOD+ODD)/SOD', abs(g['M'] - 4.0) < 1e-9, '%.3f' % g['M'])
check('p_eff = p/M', abs(g['p_eff_um'] - 13.75) < 1e-9, '%.2f µm' % g['p_eff_um'])

print()
print('通过 %d / 失败 %d' % (ok, fail))
sys.exit(1 if fail else 0)
