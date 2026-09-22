# -*- coding: utf-8 -*-
"""M3 验收：探测器响应矩阵（手册 §8 M3 判据）。

  判据 1：R 的行和 = 该能量下的总探测效率
  判据 2：K 边处应有跳变

判据 1 的实现要点：行和 + 阈值以下丢失 = 总效率（**概率守恒**）。
早期版本在 R 上做再归一化（R *= eff/s），等于把跌破阈值的丢失补回效率里，
这条检查会假通过 —— 所以守恒式必须显式写出 below_threshold 项。
"""
import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / 'src'))
_sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np                                            # noqa: E402

import ct_detector as D                                      # noqa: E402

ok = fail = 0


def check(tag, good, detail=''):
    global ok, fail
    ok += good
    fail += (not good)
    print('  [%s] %-44s %s' % ('OK ' if good else 'FAIL', tag, detail))


print('=== 判据 1：行和 = 总探测效率（概率守恒）===')
for mat in ('CdTe', 'CZT', 'Si', 'GaAs'):
    r = D.validate_m3(mat, pixel_um=55.0, sigma_c_um=15.0)
    print('  %-5s 守恒误差 %.2e   行和/总效率偏差 均值 %.4f  最大 %.4f'
          % (mat, r['conservation_err_max'], r['rowsum_rel_err_mean'],
             r['rowsum_rel_err_max']))
    check('%s 概率守恒' % mat, r['crit1a_conservation'], '%.1e' % r['conservation_err_max'])
    check('%s 行和不超过吸收效率' % mat, r['crit1b_never_exceeds'])

print()
print('=== 判据 2：K 边跳变 ===')
print('  %-6s %10s %10s %10s' % ('材料', 'K 边 keV', 'μ 跳变比', 'η 跳变比'))
for mat in ('CdTe', 'CZT', 'Si', 'GaAs'):
    r = D.validate_m3(mat)
    print('  %-6s %10.3f %10.2f %10.4f'
          % (mat, r['k_edge_kev'], r['mu_jump_ratio'], r['eff_jump_ratio']))
    check('%s K 边跳变 ≥ 1.5×' % mat, r['crit2_kedge_jump'],
          'μ 跳变 %.2f×' % r['mu_jump_ratio'])

print()
print('=== K 边逃逸缺口（CdTe, E_K = 26.71 keV）===')
r = D.detector_response('CdTe', thresholds_kev=(20.0, 35.0, 50.0, 70.0, 100.0))
E, cnt, bl = r['energies_kev'], r['eff_counted'], r['below_threshold_frac']
ek = r['k_edge_kev']


def at(e):
    i = int(np.argmin(np.abs(E - e)))
    return cnt[i], bl[i]


c_lo, b_lo = at(ek * 0.90)
c_hi, b_hi = at(ek * 1.05)
c_far, b_far = at(60.0)
print('  %.1f keV（K 边以下） 行和 %.4f  阈值损失 %.4f' % (ek * 0.90, c_lo, b_lo))
print('  %.1f keV（刚过 K 边） 行和 %.4f  阈值损失 %.4f' % (ek * 1.05, c_hi, b_hi))
print('  60.0 keV（远高于 K 边）行和 %.4f  阈值损失 %.4f' % (c_far, b_far))
check('K 边处出现逃逸缺口（损失骤增）', b_hi > 3.0 * b_lo,
      '%.4f → %.4f' % (b_lo, b_hi))
check('远高于 K 边后损失回落', b_far < b_hi,
      '%.4f < %.4f' % (b_far, b_hi))
check('K 边以下无 K 逃逸（损失低）', b_lo < 0.15, '%.4f' % b_lo)

print()
print('=== 材料/FOV 依赖 ===')
for mat, p, s in (('CdTe', 55.0, 15.0), ('CdTe', 100.0, 15.0), ('Si', 55.0, 8.0)):
    rr = D.detector_response(mat, pixel_um=p, sigma_c_um=s)
    print('  %-5s p=%5.1f σ_c=%4.1f µm  电荷损失 %.4f  K边逃逸概率 %.3f'
          % (mat, p, s, rr['charge_loss'], rr['escape_prob']))
check('p/σ_c 减小 → 电荷损失增大',
      D.charge_loss(20.0, 15.0) > D.charge_loss(100.0, 15.0),
      '%.4f > %.4f' % (D.charge_loss(20.0, 15.0), D.charge_loss(100.0, 15.0)))

print()
print('通过 %d / 失败 %d' % (ok, fail))
_sys.exit(1 if fail else 0)
