# -*- coding: utf-8 -*-
"""用交接手册的数值表验收 ct_synchrotron.py。

判据来自《PCCT 模拟同步辐射 CT · 算法交接手册》§4.3 / §5 / 附录 B。
"""
import sys
import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / 'src'))
_sys.stdout.reconfigure(encoding='utf-8', errors='replace')


import ct_synchrotron as CS

ok = fail = 0


def check(name, got, want, tol=0.02):
    global ok, fail
    good = abs(got - want) <= tol * max(1.0, abs(want))
    ok += good
    fail += (not good)
    print('  [%s] %-44s got=%-12.4g want=%-12.4g' % ('OK ' if good else 'FAIL', name, got, want))


print('=== 手册 §4.3 最优放大率与最优分辨率（精确解）===')
cases = [(10.0, 55.0, 31.25, 9.84), (5.0, 55.0, 122.0, 4.98),
         (5.0, 200.0, 1601.0, 5.00), (1.0, 55.0, 3026.0, 1.00),
         (0.5, 55.0, 12101.0, 0.50)]
for f, p, m_star, r_star in cases:
    res = CS.analyze(source='metaljet', focus_um=f, pixel_um=p, sod_mm=10.0, odd_mm=490.0)
    g = res['geometry']
    check('f=%.1f p=%.0f  M*' % (f, p), g['M_star'], m_star, tol=0.01)
    check('f=%.1f p=%.0f  r* (µm)' % (f, p), g['r_star_um'], r_star, tol=0.02)

print()
print('=== 手册 §5 采样速查（p = 0.2 mm）===')
res = CS.analyze(pixel_um=200.0, oversample=4)
s = res['sampling']
check('f0 = 1/p (cu/mm)', s['f0_cyc_mm'], 5.0, 1e-9)
check('f_N(1)', CS.analyze(pixel_um=200.0, oversample=1)['sampling']['f_nyq_over'], 2.5, 1e-9)
check('f_N(2)', CS.analyze(pixel_um=200.0, oversample=2)['sampling']['f_nyq_over'], 5.0, 1e-9)
check('f_N(4)', CS.analyze(pixel_um=200.0, oversample=4)['sampling']['f_nyq_over'], 10.0, 1e-9)
check('K=2 已饱和 → 有效 Nyquist', s['f_nyq_effective'], 5.0, 1e-9)

print()
print('=== 手册 §5 剂量幂律 ===')
for ref, tgt, want in [(100.0, 50.0, 8.0), (100.0, 25.0, 64.0), (100.0, 10.0, 1000.0)]:
    d = CS.analyze(ref_res_um=ref, target_res_um=tgt)['dose']
    check('N=%.0f → 剂量 ×N³' % (ref / tgt), d['dose_x_n3'], want, tol=0.01)

print()
print('=== 手册 §4.6 K 荧光逃逸长度 ===')
check('CdTe  λ_K (µm)', CS.analyze(detector='CdTe')['detector']['lambda_K_um'], 249.0, 0.01)
check('Si    λ_K (µm)', CS.analyze(detector='Si')['detector']['lambda_K_um'], 15.0, 0.01)

print()
print('=== 手册 §4.13 源亮度差 ===')
d = CS.analyze(source='undulator')['dose']
lo, hi = d['brightness_gap_lo'], d['brightness_gap_hi']
print('  同步辐射 10^18–10^20 vs 转靶 10^7–10^9 → 差值 %.0e .. %.0e（%0.1f 个数量级中值）'
      % (lo, hi, d['brightness_gap_decades']))

print()
print('=== 电荷共享判据 p ≳ 2σ_c ===')
for p, expect in [(55.0, True), (30.0, True), (25.0, False), (15.0, False)]:
    det = CS.analyze(pixel_um=p, sigma_c_um=15.0)['detector']
    good = det['spectral_ok'] == expect
    ok += good
    fail += (not good)
    print('  [%s] p=%.0f µm  νs 2σ_c=%.0f µm → 能谱%s'
          % ('OK ' if good else 'FAIL', p, det['p_crit_um'],
             '可用' if det['spectral_ok'] else '失效'))

print()
print('=== 默认配置（液态金属靶 + 55 µm CdTe + M≈3）===')
res = CS.analyze(source='metaljet', focus_um=10.0, sod_mm=100.0, odd_mm=200.0,
                 pixel_um=55.0, sigma_c_um=15.0, oversample=4, bins=8)
for line in CS.summary_lines(res):
    print('   ' + line)
print()
print('   VMI %g keV 权重: %s (残差 %.2e, ok=%s)'
      % (res['spectrum']['vmi_keV'], ['%.3f' % w for w in res['spectrum']['vmi_weights']],
         res['spectrum']['vmi_residual'], res['spectrum']['vmi_ok']))
print('   DQE@半Nyquist(%.1f cu/mm) = %.3f   NEQ = %.3e   香农容量 = %.1f bit/mm'
      % (res['dose']['dqe_report_freq'], res['dose']['dqe_at_half_nyq'],
         res['dose']['neq_at_half_nyq'], res['dose']['shannon_bits_per_mm']))
print('   分辨率换算自检: MTF10=%0.1f cu/mm → %.1f µm；MTF50 → %.1f µm'
      % (res['mtf']['mtf10_cyc_mm'], res['mtf']['res_at_mtf10_um'], res['mtf']['res_at_mtf50_um']))
print('   FOV = %.1f mm (n_ch=%d, M=%.2f)' % (res['geometry']['fov_mm'],
                                             res['geometry']['n_ch'], res['geometry']['M']))

print()
print('通过 %d / 失败 %d' % (ok, fail))
sys.exit(1 if fail else 0)
