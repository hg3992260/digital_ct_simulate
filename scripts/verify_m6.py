# -*- coding: utf-8 -*-
"""M6 验收：剂量-分辨率曲线（手册 §8 M6）。

  判据 1：量子噪声服从 1/√N                      → **通过**
  判据 2：剂量按手册 §5 的 N³~N⁴ 标度            → **不通过**（实测 α ≈ 0）

本脚本把已确证的部分作为测试，把与手册不符的部分作为**显式记录的差异**输出，
以免把未验证的标度当成已验结论。差异原因见 validate_m6 的 docstring。

另：分辨率-剂量曲线当前不可用 —— MTF-10 指标在噪图上会误判（高剂量处给出
0.713 µm，比光学极限 1.237 µm 还好），需换成对噪声稳健的判据。这一点也一并
如实标记为未完成，不假装通过。
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
    print('  [%s] %-46s %s' % ('OK ' if good else 'FAIL', tag, detail))


print('=== 量子噪声模型 ===')
g = F.m1_geometry(20.0, 780.0, 55.0, 256, 1, 180, 180.0)
ang = np.linspace(0.0, 180.0, 180, endpoint=False)
ell = F.slit_phantom(g['p_eff_mm'] / 6.0)
acq = F.acquire_k_frames(ell, ang, 256, g['p_eff_mm'], K=4, scale_mm=1.0,
                         mu_scale=1.0, n_sub=16)
clean = F.assemble_dense(acq)
d = F.add_poisson_noise(clean, 1e5, seed=1)
check('噪声后的线积分仍有限', np.all(np.isfinite(d)), 'range [%.4f, %.4f]' % (d.min(), d.max()))
check('光子数越多噪声越小', F.recon_noise(
    __import__('skimage.transform', fromlist=['iradon']).iradon(
        F.add_poisson_noise(clean, 1e6, seed=2).T, theta=ang, circle=True,
        preserve_range=True)) <
    F.recon_noise(__import__('skimage.transform', fromlist=['iradon']).iradon(
        F.add_poisson_noise(clean, 1e3, seed=2).T, theta=ang, circle=True,
        preserve_range=True)), '')

print()
print('=== 判据 1：噪声 ∝ 1/√N（差值法隔离纯量子噪声）===')
r = F.validate_m6(n_cols=256, n_angles=180)
for n, v in zip([1e4, 1e5, 1e6], r['noise_x_sqrtN']):
    print('  N=%9.0f   噪声×√N = %.5f' % (n, v))
check('噪声×√N 恒定（四个数量级）', r['crit1_inverse_sqrtN'],
      '离散度 %.4f' % r['noise_law_spread'])

print()
print('=== 判据 2：剂量标度 vs 手册 §5 ===')
print('  达到同一噪声所需光子数：')
for k, v in r['dose_for_target_noise'].items():
    print('    K=%d  →  %.4e' % (k, v))
print('  拟合指数 α = %.3f      手册 §5 称 N³~N⁴ (3~4)' % r['scaling_exponent'])
print('  [差异] 实测 α ≈ 0，与手册不符 —— 已记录，未当作通过')
check('判据 2 与手册一致', r['crit2_manual_scaling'],
      'α = %.3f（手册 3~4）' % r['scaling_exponent'])

print()
print('=== 未完成项（如实标记，不假装通过）===')
print('  分辨率-剂量曲线：MTF-10 在噪图上不可靠（高剂量给出 0.713 µm，')
print('                    比光学极限 1.237 µm 还好，物理上不可能），需换稳健判据。')

print()
print('通过 %d / 失败 %d' % (ok, fail))
_sys.exit(1 if fail else 0)
