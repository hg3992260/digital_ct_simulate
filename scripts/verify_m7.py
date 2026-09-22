# -*- coding: utf-8 -*-
"""M7 验收：尺度验证与置信区间（手册 §8 M7）。

  判据 1（尺度）：已知直径的圆盘重建后尺寸误差 < 5%
  判据 2（置信区间）：对位移误差 σ_δ 做蒙特卡洛，分辨率的 95% CI
        应落在手册 §2.2 的可达区间内（1–5 µm 焦点源 → 1–5 µm）

为什么要先验尺度：分辨率数字再漂亮，若几何/单位错了就没有意义。
实测圆盘 200 µm → 200.06 µm（0.03%），说明 M1–M5 的尺度是自洽的。

不确定性的来源不是凭空加的 —— 手册在 M4 的输入规格里就写了"位移误差 σ_δ"。
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
    print('  [%s] %-42s %s' % ('OK ' if good else 'FAIL', tag, detail))


print('=== 判据 1：尺度验证（已知直径圆盘）===')
r = F.validate_m7(focus_um=3.0, pixel_um=55.0, K=4, disk_um=200.0,
                  jitter_frac=0.05, n_trials=10)
print('  真值 %.1f µm  →  重建实测 %.2f µm   相对误差 %.3f%%'
      % (r['disk_true_um'], r['disk_meas_um'], r['scale_rel_err'] * 100))
check('尺度误差 < 5%', r['crit1_scale'], '%.3f%%' % (r['scale_rel_err'] * 100))

print()
print('=== 判据 2：分辨率 95% 置信区间 ===')
print('  %-8s %-22s %10s %10s %20s' % ('σ_δ', '', '均值 µm', '标准差', '95% CI µm'))
for jf in (0.0, 0.05, 0.15):
    rr = F.validate_m7(focus_um=3.0, pixel_um=55.0, K=4, jitter_frac=jf, n_trials=12)
    sig_um = jf * rr['geometry']['p_eff_mm'] * 1000.0
    print('  %-8s %-22s %10.4f %10.4f   [%.4f, %.4f]'
          % ('%.0f%%' % (jf * 100), '%.3f µm' % sig_um, rr['res_mean_um'],
             rr['res_std_um'], rr['res_ci95_um'][0], rr['res_ci95_um'][1]))

r0 = F.validate_m7(focus_um=3.0, pixel_um=55.0, K=4, jitter_frac=0.0, n_trials=4)
r1 = F.validate_m7(focus_um=3.0, pixel_um=55.0, K=4, jitter_frac=0.15, n_trials=12)
check('σ_δ=0 时确定性（std=0）', r0['res_std_um'] == 0.0, 'std %.6f' % r0['res_std_um'])
check('σ_δ 增大 → 不确定度增大', r1['res_std_um'] > r0['res_std_um'],
      '%.4f > %.4f' % (r1['res_std_um'], r0['res_std_um']))
check('σ_δ 增大 → 均值变差', r1['res_mean_um'] > r0['res_mean_um'],
      '%.4f > %.4f' % (r1['res_mean_um'], r0['res_mean_um']))

print()
print('=== 判据 2：置信区间落在手册可达区间（1–5 µm）===')
print('  %-8s %10s %22s %s' % ('焦点 µm', '均值 µm', '95% CI µm', '判定'))
for f in (1.0, 3.0, 5.0):
    rr = F.validate_m7(focus_um=f, pixel_um=55.0, K=4, jitter_frac=0.05, n_trials=10)
    print('  %-8.1f %10.3f   [%.3f, %.3f]      %s'
          % (f, rr['res_mean_um'], rr['res_ci95_um'][0], rr['res_ci95_um'][1],
             'PASS' if rr['crit2_ci_in_band'] else 'FAIL'))
    check('f=%.0f µm 的 CI 在 1–5 µm 带内' % f, rr['crit2_ci_in_band'],
          '[%.2f, %.2f] µm' % (rr['res_ci95_um'][0], rr['res_ci95_um'][1]))

print()
print('通过 %d / 失败 %d' % (ok, fail))
_sys.exit(1 if fail else 0)
