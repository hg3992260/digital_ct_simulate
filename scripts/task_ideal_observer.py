# -*- coding: utf-8 -*-
"""任务型指标：把"细胞有无"当作检测任务，用**理想观察者**（预白化匹配滤波）
给出 15 µm 下"相衬 vs 纯衰减"的可信倍数与剂量需求。

为什么必须用理想观察者
----------------------
15 µm 采样下，20 µm 细胞只占 0.7 像素，"环带均值差"这类指标毫无意义。
理想观察者 SNR² = Σ_f |S(f)|²/NPS(f) 正确地把已知信号形状与**真实噪声谱**
一起考虑，是处理亚像素信号的标准工具。

关键设计：体模用**远大于视场**的均匀背景
----------------------------------------
背景直径 >> 重建视场 → 每条射线的弦长近似恒定 → 投影近似均匀 → **没有外边界，
就没有 FBP 条状伪影**。上一版把"大盘套小盘"的环带当成信号，量到的其实是伪影，
所以倍数从 1× 散到 363×。这一版从体模设计上消除了该污染源。

信号模板取"无噪声重建(含细胞) − 无噪声重建(纯背景)"，是重建域里**精确**的信号，
不需要假设任何解析形状。
"""
import sys
import numpy as np
sys.path.insert(0, r'I:\dsw\_gh_repo\src')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import ct_forward as F
from skimage.transform import iradon

E_KEV = 33.0
R_MM = 500.0                     # 传播距离 0.5 m
MU_W = 0.03270                   # 1/mm
D_W = 2.10e-7                    # δ of water @33 keV
NPIX = 192


def _recon(sino, ang, n):
    return iradon(sino.T, theta=ang, circle=True, output_size=n,
                  preserve_range=True).astype(np.float64)


def task_snr(cell_um, p_eff_um, K, n0, n_ang=360, mech='abs', seed=0,
             d_mu_frac=0.005, d_delta_frac=0.02):
    """返回 (SNR_ideal, 体素 µm, 信号能量, 噪声方差)。

    mech='abs'   → 纯衰减对比
    mech='phase' → 相衬（同一衰减 + δ 相位）
    """
    r_cell = (cell_um / 2.0) / 1000.0                 # mm
    fov_mm = NPIX * (p_eff_um / 1000.0) / 1.0         # 探测器视场
    # 背景半径 = 视场的 3 倍 → 弦长在视场内近似恒定，无外边界
    bg_r = 3.0 * fov_mm / 2.0
    scale = bg_r / 0.5

    ell_bg = [(MU_W, 0.0, 0.0, 0.5, 0.5, 0.0)]
    ell_sig = [(MU_W, 0.0, 0.0, 0.5, 0.5, 0.0),
               (MU_W * d_mu_frac, 0.0, 0.0, r_cell / scale, r_cell / scale, 0.0)]
    del_bg = [(D_W, 0.0, 0.0, 0.5, 0.5, 0.0)]
    del_sig = [(D_W, 0.0, 0.0, 0.5, 0.5, 0.0),
               (D_W * d_delta_frac, 0.0, 0.0, r_cell / scale, r_cell / scale, 0.0)]

    p_mm = p_eff_um / 1000.0
    n_cols = NPIX
    ang = np.linspace(0.0, 180.0, n_ang, endpoint=False)

    if mech == 'phase':
        _, s_bg = F.phase_contrast_sinogram(ell_bg, del_bg, ang, n_cols, p_mm,
                                            scale, E_KEV, R_MM)
        _, s_sig = F.phase_contrast_sinogram(ell_sig, del_sig, ang, n_cols, p_mm,
                                             scale, E_KEV, R_MM)
    else:
        s_bg = F.analytic_sinogram(ell_bg, n_cols, p_mm, ang, scale, 1.0)
        s_sig = F.analytic_sinogram(ell_sig, n_cols, p_mm, ang, scale, 1.0)

    # ---- 信号模板：无噪声重建之差（重建域里精确的信号形状）----
    n = NPIX * K
    sig = _recon(s_sig, ang, n) - _recon(s_bg, ang, n)

    # ---- NPS：差值法估噪声功率谱 ----
    acc = np.zeros((n, n), dtype=float)
    reps = 4
    for i in range(reps):
        r1 = _recon(F.add_poisson_noise(s_bg, n0, seed=seed + 2 * i), ang, n)
        r2 = _recon(F.add_poisson_noise(s_bg, n0, seed=seed + 2 * i + 1), ang, n)
        acc += np.abs(np.fft.fft2(r1 - r2)) ** 2
    nps = acc / (2.0 * reps)                     # 噪声功率谱（2D 周期图）

    # 径向平均 NPS 以抑制估计方差（对理想观察者是可接受的平滑）
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    fr = np.sqrt(fy ** 2 + fx ** 2)
    nb = max(8, n // 24)
    idx = np.clip((fr / 0.5 * nb).astype(int), 0, nb - 1)
    prof = np.zeros(nb)
    cnt = np.zeros(nb)
    for b in range(nb):
        m = idx == b
        if m.any():
            prof[b] = nps[m].mean()
            cnt[b] = m.sum()
    good = cnt > 0
    nps_s = np.zeros_like(nps)
    for b in range(nb):
        if good[b]:
            nps_s[idx == b] = prof[b]
    nps_s = np.maximum(nps_s, 1e-30)

    S2 = np.abs(np.fft.fft2(sig)) ** 2
    snr2 = float(np.sum(S2 / nps_s))
    return (np.sqrt(max(snr2, 0.0)), n / float(NPIX) * p_mm * 1000.0,
            float(np.sum(sig ** 2)), float(np.mean(np.diag(nps_s))))


print('=' * 96)
print('任务型检测：15 µm 下"相衬 vs 纯衰减"（理想观察者，预白化匹配滤波）')
print('=' * 96)
print('体模：均匀背景（半径 = 3× 视场，无外边界 → 无条状伪影）+ 中心细胞')
print('细胞衰减对比 = 0.5%（≈16 HU @33keV），相位对比 δ = 2%（脂质/水界面的量级）')
print()
hdr = '%-26s %9s %11s %13s %13s %10s'
print(hdr % ('配置', '体素µm', 'SNR_吸收', 'SNR_相衬', '相衬/吸收', '相衬剂量'))
print('-' * 96)

N0 = 1.0e5
rows = []
for cell_um in (20.0, 40.0):
    for p_eff_um, K in ((15.0, 1), (15.0, 2), (5.0, 1)):
        sa, vox, e_sig_a, _ = task_snr(cell_um, p_eff_um, K, N0, mech='abs')
        sp, _, e_sig_p, _ = task_snr(cell_um, p_eff_um, K, N0, mech='phase')
        ratio = sp / sa if sa > 0 else float('inf')
        need_p = N0 * (3.0 / sp) ** 2 if sp > 0 else float('inf')
        rows.append((cell_um, p_eff_um, K, sa, sp, ratio, need_p))
        print(hdr % ('细胞 %2.0fµm 采样%4.1fµm K=%d' % (cell_um, p_eff_um, K),
                     '%.2f' % vox, '%.4f' % sa, '%.4f' % sp,
                     '%.1f×' % ratio, '%.2e' % need_p))

print()
print('注：SNR 在 N₀ = 1e5 光子/像素/帧 下计算；SNR ∝ √N，故剂量可按 (3/SNR)²·N₀ 换算。')
