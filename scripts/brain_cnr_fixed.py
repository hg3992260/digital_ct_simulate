# -*- coding: utf-8 -*-
"""脑组织 15 µm 体素 · 检出 15 HU（灰质 vs 白质）的剂量需求。

修正了上一版的两个 bug：
  1. 视场没覆盖体模 —— 现在 n_cols 由 sample_mm 反推（留 30% 余量）
  2. 取样区互相污染 —— 现在用**半径掩膜**严格隔离：
        灰质区 rad < 0.45·R_gm
        白质区 0.6·R_wm < rad < 0.9·R_wm
     两者之间留有间隔，且都在体模内部，不碰边界。

噪声用差值法（两次独立泊松实现求差），因为它已被验证能隔离纯量子噪声，
且直接落在重建值（μ）的量纲上 —— CNR = Δμ/σ_μ 无需任何归一化假设。
"""
import sys
import numpy as np
sys.path.insert(0, r'I:\dsw\_gh_repo\src')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import ct_forward as F
import ct_detector as CD
from skimage.transform import iradon

try:
    import xraylib as xr
    def mu_water(E):
        return (xr.CS_Photo(8, E) + xr.CS_Compt(8, E) + xr.CS_Rayl(8, E)) / 10.0
except Exception:
    def mu_water(E):
        return 0.0327 * (33.0 / E) ** 3


def cnr(p_det_um, K, focus_um, energy_kev, sample_mm, n0, M, n_ang=200, seed=1):
    """返回 (CNR, 重建体素 µm, 视场 mm, Δμ, σ_μ, 需要的 n_cols)。"""
    mu_w = mu_water(energy_kev)
    mu_gm, mu_wm = mu_w * 1.040, mu_w * 1.025          # 40 HU / 25 HU

    # ---- 修正 1：入参是**探测器像素**，样品面像素由几何内核给出 p_eff = 像素/M。
    # 上一版把已经是样品面的值又除了一次 M（双重除法），M=9.13 时列数爆到 6300。
    p_obj_mm = (float(p_det_um) / float(M)) / 1000.0
    n_cols = int(np.ceil(sample_mm * 1.30 / p_obj_mm))
    n_cols = int(np.ceil(n_cols / 16.0) * 16)
    n_cols = min(n_cols, 2048)                           # P10 实配 2048 列

    g = F.m1_geometry(20.0, 20.0 * (M - 1.0), p_det_um, n_cols, 1, n_ang, 180.0)
    ang = np.linspace(0.0, 180.0, n_ang, endpoint=False)

    # 白质背景盘（半径 sample/2）+ 灰质盘（半径 sample/4）
    scale_mm = sample_mm / 2.0 / 0.5
    ell = [(mu_wm, 0.0, 0.0, 0.50, 0.50, 0.0),
           (mu_gm - mu_wm, 0.0, 0.0, 0.25, 0.25, 0.0)]

    acq = F.acquire_k_frames(ell, ang, n_cols, g['p_eff_mm'], K=K,
                             scale_mm=scale_mm, mu_scale=1.0, n_sub=16)
    clean = F.apply_focus_blur(F.assemble_dense(acq), acq['p_dense_mm'], focus_um, M)

    r1 = iradon(F.add_poisson_noise(clean, n0, seed=seed).T, theta=ang,
                circle=True, preserve_range=True).astype(np.float64)
    r2 = iradon(F.add_poisson_noise(clean, n0, seed=seed + 1).T, theta=ang,
                circle=True, preserve_range=True).astype(np.float64)

    # ---- 修正 2：半径掩膜严格隔离取样区 ----
    n = r1.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    rad_mm = np.hypot(xx - c, yy - c) * acq['p_dense_mm']
    R_gm, R_wm = sample_mm / 4.0, sample_mm / 2.0
    gm_mask = rad_mm < 0.45 * R_gm                          # 灰质盘内 45%
    wm_mask = (rad_mm > 0.60 * R_wm) & (rad_mm < 0.90 * R_wm)  # 白质环带
    ov = float((gm_mask & wm_mask).sum())
    gm_mean = float(r1[gm_mask].mean()) if gm_mask.any() else float('nan')
    wm_mean = float(r1[wm_mask].mean()) if wm_mask.any() else float('nan')
    d_mu = abs(gm_mean - wm_mean)
    s_mu = float(np.std(r1 - r2) / np.sqrt(2.0))
    # 诊断：重建后**相对对比度**应保持 0.0146；若明显更小说明被模糊冲淡
    rel_contrast = d_mu / abs(wm_mean) if wm_mean and not np.isnan(wm_mean) else float('nan')
    return (d_mu / s_mu if s_mu > 0 else 0.0), acq['p_dense_mm'] * 1000, \
        n_cols * g['p_eff_mm'], d_mu, s_mu, n_cols, ov, rel_contrast, gm_mean, wm_mean


print('=' * 92)
print('15 µm 体素下检出 15 HU（灰质 40 HU vs 白质 25 HU）所需的剂量')
print('=' * 92)
print('水线衰减：', ', '.join('%d keV %.5f/mm' % (e, mu_water(e)) for e in (33, 70)))
print()
hdr = '%-34s %8s %8s %7s %10s %10s %9s %9s'
print(hdr % ('配置', '体素µm', 'FOVmm', 'n_cols', 'Δμ/μ_WM', '(应=0.0146)', 'σ_μ', 'CNR@1e5'))
print('-' * 104)

CASES = [
    # 名称, p_eff µm, K, 焦点 µm, 能量, 样品 mm, M
    ('体内 70keV M=1    p_eff=137µm K=1',   137.0, 1, 300.0, 70.0, 100.0, 1.00),
    ('体内 70keV M=1    p_eff=137µm K=2',   137.0, 2, 300.0, 70.0, 100.0, 1.00),
    ('离体 33keV M=9.13 p_eff=15µm K=1',    137.0, 1,  10.0, 33.0,   8.0, 9.13),
    ('离体 33keV M=9.13 p_eff=15µm K=2',    137.0, 2,  10.0, 33.0,   8.0, 9.13),
    ('离体 33keV M=9.13 p_eff=15µm K=1 f=1', 137.0, 1,  1.0, 33.0,   8.0, 9.13),
    ('离体 33keV M=40   p_eff=3.4µm K=1',   137.0, 1,   1.0, 33.0,   2.0, 40.0),
]
results = []
for name, pe, K, foc, E, samp, M in CASES:
    (cnr_v, vox, fov, dmu, smu, ncol, ov, rcon, gmm, wmm) = cnr(pe, K, foc, E, samp, 1e5, M)
    need = 1e5 * (3.0 / cnr_v) ** 2 if cnr_v > 0 else float('inf')
    results.append((name, vox, need, cnr_v))
    print(hdr % (name, '%.3f' % vox, '%.1f' % fov, '%d' % ncol,
                 '%.5f' % rcon, '', '%.2e' % smu, '%.3f' % cnr_v)
          + ('   掩膜重叠=%d' % ov if ov else ''))
print()
print('%-40s %10s %14s' % ('配置', '体素 µm', '需光子/像素 (CNR=3)'))
print('-' * 70)
for name, vox, need, c0 in results:
    print('%-40s %10.3f %14.3e' % (name, vox, need))
