# -*- coding: utf-8 -*-
"""任务型检测 · 实算 δ + Paganin 反演版（修正上一轮 90× 的版本）。

与上一轮的三处不同（都是上一轮被指出的不可信来源）：
  1. δ/β 由 **xraylib 实算**（ICRU-44 成分），不再用文献典型值
  2. 增加 **Paganin 反演**路径（`paganin_retrieve`），与弱相位 TIE 对照 ——
     反演把 TIE 的高通反解掉，信号模板由"边缘环"变回"紧凑圆盘"，
     噪声谱也随之改变，匹配滤波的结果会不一样
  3. 对比度用**实算值**：灰质/白质只有 0.97%(δ)/0.46%(μ)，
     而髓鞘/白质有 12.5%(δ)/18.7%(μ) —— 上一轮用的 15 HU 偏乐观

理想观察者：SNR² = Σ_f |S(f)|²/NPS(f)，S 取"无噪声重建之差"（精确模板），
NPS 由独立泊松实现的差值法估计。体模背景半径 = 3× 视场（无外边界 → 无条状伪影）。
"""
import sys
import numpy as np
sys.path.insert(0, r'I:\dsw\_gh_repo\src')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import ct_forward as F
import ct_detector as CD
from skimage.transform import iradon

E_KEV = 33.0
R_MM = 500.0
NPIX = 192


def _rec(sino, ang, n):
    return iradon(sino.T, theta=ang, circle=True, output_size=n,
                  preserve_range=True).astype(np.float64)


def task(cell_um, p_eff_um, K, n0, bg='white_matter', fg='myelin',
         mech='abs', n_ang=360, seed=0):
    """返回 (SNR, 体素 µm, 相对衰减差, 相对 δ 差)。"""
    mu_bg = CD.mu_from_beta(CD.delta_beta(bg, E_KEV)[1], E_KEV)
    mu_fg = CD.mu_from_beta(CD.delta_beta(fg, E_KEV)[1], E_KEV)
    d_bg = CD.delta_beta(bg, E_KEV)[0]
    d_fg = CD.delta_beta(fg, E_KEV)[0]

    r_cell = (cell_um / 2.0) / 1000.0
    p_mm = p_eff_um / 1000.0
    fov_mm = NPIX * p_mm
    bg_r = 3.0 * fov_mm / 2.0
    scale = bg_r / 0.5
    a_cell = r_cell / scale

    abs_bg = [(mu_bg, 0, 0, 0.5, 0.5, 0)]
    abs_sg = [(mu_bg, 0, 0, 0.5, 0.5, 0), (mu_fg - mu_bg, 0, 0, a_cell, a_cell, 0)]
    del_bg = [(d_bg, 0, 0, 0.5, 0.5, 0)]
    del_sg = [(d_bg, 0, 0, 0.5, 0.5, 0), (d_fg - d_bg, 0, 0, a_cell, a_cell, 0)]

    ang = np.linspace(0.0, 180.0, n_ang, endpoint=False)
    d_ret, b_ret = CD.delta_beta(bg, E_KEV)      # 反演按背景 δ/β（单材料近似）

    if mech == 'abs':
        s_bg = F.analytic_sinogram(abs_bg, NPIX, p_mm, ang, scale, 1.0)
        s_sg = F.analytic_sinogram(abs_sg, NPIX, p_mm, ang, scale, 1.0)
    elif mech == 'tie':
        _, s_bg = F.phase_contrast_sinogram(abs_bg, del_bg, ang, NPIX, p_mm, scale, E_KEV, R_MM)
        _, s_sg = F.phase_contrast_sinogram(abs_sg, del_sg, ang, NPIX, p_mm, scale, E_KEV, R_MM)
    else:  # paganin
        _, _, s_bg = F.phase_paganin_sinogram(abs_bg, del_bg, ang, NPIX, p_mm, scale,
                                              E_KEV, R_MM, d_ret, b_ret)
        _, _, s_sg = F.phase_paganin_sinogram(abs_sg, del_sg, ang, NPIX, p_mm, scale,
                                              E_KEV, R_MM, d_ret, b_ret)

    n = NPIX * K
    sig = _rec(s_sg, ang, n) - _rec(s_bg, ang, n)

    acc = np.zeros((n, n))
    for i in range(4):
        r1 = _rec(F.add_poisson_noise(s_bg, n0, seed=seed + 2 * i), ang, n)
        r2 = _rec(F.add_poisson_noise(s_bg, n0, seed=seed + 2 * i + 1), ang, n)
        acc += np.abs(np.fft.fft2(r1 - r2)) ** 2
    nps = acc / 8.0

    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    fr = np.sqrt(fy ** 2 + fx ** 2)
    nb = max(8, n // 24)
    idx = np.clip((fr / 0.5 * nb).astype(int), 0, nb - 1)
    nps_s = np.zeros_like(nps)
    for b in range(nb):
        m = idx == b
        if m.any():
            nps_s[m] = nps[m].mean()
    nps_s = np.maximum(nps_s, 1e-30)

    S2 = np.abs(np.fft.fft2(sig)) ** 2
    snr = float(np.sqrt(max(np.sum(S2 / nps_s), 0.0)))
    return snr, n / float(NPIX) * p_mm * 1000.0, \
        (mu_fg - mu_bg) / mu_bg, (d_fg - d_bg) / d_bg


N0 = 1.0e5
print('=' * 100)
print('任务型检测 · xraylib 实算 δ + Paganin 反演      E=%.0f keV  R=%.0f mm' % (E_KEV, R_MM))
print('=' * 100)
mu_gm = CD.mu_from_beta(CD.delta_beta('gray_matter', E_KEV)[1], E_KEV)
mu_wm = CD.mu_from_beta(CD.delta_beta('white_matter', E_KEV)[1], E_KEV)
mu_my = CD.mu_from_beta(CD.delta_beta('myelin', E_KEV)[1], E_KEV)
print('实算对比度：灰质/白质 μ %+.2f%% (≈%+.1f HU)   δ %+.2f%%'
      % (100 * (mu_gm - mu_wm) / mu_wm, 1000 * (mu_gm - mu_wm) / 0.0330,
         100 * (CD.delta_beta('gray_matter', E_KEV)[0] - CD.delta_beta('white_matter', E_KEV)[0])
         / CD.delta_beta('white_matter', E_KEV)[0]))
print('            髓鞘/白质 μ %+.2f%%              δ %+.2f%%'
      % (100 * (mu_my - mu_wm) / mu_wm,
         100 * (CD.delta_beta('myelin', E_KEV)[0] - CD.delta_beta('white_matter', E_KEV)[0])
         / CD.delta_beta('white_matter', E_KEV)[0]))
print()

hdr = '%-30s %8s %10s %10s %10s %10s'
for label, fg, cell in (('目标 = 髓鞘（细胞膜/髓鞘界面）', 'myelin', 20.0),
                        ('目标 = 灰质（白质背景）', 'gray_matter', 20.0)):
    print('【%s】细胞直径 %.0f µm，N₀=%.0e 光子/像素' % (label, cell, N0))
    print(hdr % ('机制 / 采样', '体素µm', 'SNR', '相对衰减差', '相对δ差', '剂量(SNR=3)'))
    print('-' * 84)
    base = None
    for mech, mname in (('abs', '纯衰减'), ('tie', '相衬·弱相位TIE'), ('paganin', '相衬·Paganin反演')):
        for p_eff, K in ((15.0, 1), (5.0, 1)):
            snr, vox, dm, dd = task(cell, p_eff, K, N0, fg=fg, mech=mech)
            need = N0 * (3.0 / snr) ** 2 if snr > 0 else float('inf')
            if mech == 'abs' and p_eff == 15.0:
                base = snr
            ratio = (snr / base) if base else float('nan')
            print(hdr % ('%s / %4.1fµm K=%d' % (mname, p_eff, K), '%.2f' % vox,
                         '%.4f' % snr, '%+.3f%%' % (100 * dm), '%+.2f%%' % (100 * dd),
                         '%.2e' % need) + ('   ×%.0f' % ratio if ratio == ratio else ''))
    print()
