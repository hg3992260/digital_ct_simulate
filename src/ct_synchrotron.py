# -*- coding: utf-8 -*-
"""同步辐射 CT 仿真模式（以**单元光子计数**架构改造）——纯 numpy，无 Qt 依赖。

设计依据
--------
《PCCT 模拟同步辐射 CT · 算法交接手册》(东观澜, v1.0, 2026-09-10)
以及《PCD CT 细胞级扫描 · 工程交接文档》(v1.0, 2026-09-20)。

一句话定位
----------
无像素约束 PCD + 多次过采样，把实验室 CT 的极限从「采样受限」转移到
「源受限」与「电荷共享受限」：能填平探测器与采样那一半差距，
填不平源亮度与相干性那一半。本模块把这条结论**算出来**。

级联模型（手册 §3.1）
---------------------
    源能谱 S(E) → 焦点模糊 → 样品 μ(x,y,E) → 像素孔径 → 电荷云/串扰 → 计数分 bin
      亮度 B        MTF_f       投影 p(θ,t)       MTF_ap       MTF_ch        NPS(E)

    MTF_sys(f) ≈ MTF_f(f) · MTF_ap(f) · MTF_ch(f) · MTF_rec(f)
    NEQ(f) = φ·|MTF_sys(f)|²/NPS(f)      DQE = NEQ/φ
    C = ∫₀^{f_Nyq} log₂(1+NEQ(f)) df

诚实边界（务必保留）
--------------------
本模块输出的是**模型计算值**，不是实测值。手册 §1.2 明确警告：原技术线只有
可行性分析、没有仿真器实现，且 MTF/NPS/剂量曲线原为"示意构造"。
本模块把那些示意曲线换成可复算的解析模型，但**仍不等于实测标定**；
引用时请按手册 §6.2 的口径说明来源。

单位约定
--------
* 焦点 / 像素 / 电荷云：**µm**
* 源-物 / 物-探距：**mm**
* 空间频率：**cycles/mm**
* 能量：**keV**；时间：**ns**
"""

import numpy as np

try:
    from scipy.special import j1 as _bessel_j1
except Exception:                                    # pragma: no cover
    _bessel_j1 = None

# numpy 2.0 把 trapz 改名为 trapezoid；两者都兼容
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

# ---------------------------------------------------------------------------
# 源：亮度取手册 §4.13 的口径（ph/(s·mm²·mrad²·0.1%BW)），焦点取 §6 表
# ---------------------------------------------------------------------------
SOURCES = {
    'undulator': dict(label='同步辐射 · 波荡器 (Undulator)',
                      brightness=(1e18, 1e20), focus_um=0.5, coherent=True,
                      note='衍射极限光源，空间相干性最好 → 相衬/纳米 CT 的前提'),
    'bending': dict(label='同步辐射 · 弯铁 (Bending Magnet)',
                    brightness=(1e14, 1e16), focus_um=5.0, coherent=True,
                    note='通量高但相干性弱于波荡器；常规 SR-μCT 主力'),
    'metaljet': dict(label='液态金属射流靶 (MetalJet D2+)',
                     brightness=(1e9, 1e11), focus_um=10.0, coherent=False,
                     note='同焦点下实验室源亮度最高，最接近同步辐射（手册 §6 首选）'),
    'microfocus': dict(label='封闭式微焦点管',
                       brightness=(1e7, 1e9), focus_um=20.0, coherent=False,
                       note='固体阳极热限 → 小焦点与功率不可兼得'),
    'nanofocus': dict(label='透射靶纳米焦点管',
                      brightness=(1e6, 1e8), focus_um=1.5, coherent=False,
                      note='亮度极低，视野极小'),
    'rotating': dict(label='旋转阳极 (实验室常规)',
                     brightness=(1e7, 1e9), focus_um=300.0, coherent=False,
                     note='手册 §4.13 中"实验室转靶 10⁷–10⁹"即此档'),
}

# ---------------------------------------------------------------------------
# 探测器材料：K 吸收边与 Kα 荧光能量决定逃逸长度 λ_K = 1/µ(E_K)
# 手册 §4.6 核算：Cd Kα1 23.17 keV → λ_K ≈ 249 µm；Te Kα1 27.47 keV → ≈ 99 µm
# ---------------------------------------------------------------------------
DETECTORS = {
    'CdTe': dict(label='CdTe（碲化镉）', z=48,
                 K_edge_keV=26.71, Ka1_keV=23.17, lambda_K_um=249.0,
                 sigma_c_um=(10.0, 30.0), note='K 荧光逃逸 ~0.25 mm，远大于 55 µm 像素'),
    'CZT': dict(label='CZT（碲锌镉）', z=48,
                K_edge_keV=26.71, Ka1_keV=23.17, lambda_K_um=249.0,
                sigma_c_um=(10.0, 30.0), note='与 CdTe 同族，Cd Kα 主导逃逸'),
    'Si': dict(label='Si（硅）', z=14,
               K_edge_keV=1.84, Ka1_keV=1.74, lambda_K_um=15.0,
               sigma_c_um=(3.0, 8.0), note='低 Z → K 荧光弱，但量子效率低、需更高能量'),
    'GaAs': dict(label='GaAs（砷化镓）', z=33,
                 K_edge_keV=11.87, Ka1_keV=10.54, lambda_K_um=40.0,
                 sigma_c_um=(6.0, 15.0), note='介于 Si 与 CdTe 之间'),
}

# 默认 bin 阈值（keV）：前 4 个沿用 NAEOTOM Alpha 研究模式 20/52/75/82，
# 后面按同步辐射多 bin 需求外推（手册 §4.8）
DEFAULT_BIN_EDGES = (20.0, 35.0, 52.0, 65.0, 75.0, 82.0, 100.0, 120.0)

# 探测器整形时间典型值（ns）：决定脉冲堆积的计数率上限 1/τ
DEFAULT_SHAPING_NS = {'CdTe': 20.0, 'CZT': 20.0, 'Si': 12.0, 'GaAs': 15.0}


# ===========================================================================
# 基础函数
# ===========================================================================

def _sinc(x):
    """归一化 sinc：sin(x)/x，x=0 处取 1。"""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = x != 0
    out[nz] = np.sin(x[nz]) / x[nz]
    return out


def disk_mtf(nu_cyc_per_mm, diameter_mm):
    """圆盘焦点的 MTF：2·J₁(π d ν)/(π d ν)（手册 §8 M2 验收依据）。

    scipy 不可用时退化为高斯近似（首零点位置近似对齐）。
    """
    a = np.pi * float(diameter_mm) * np.asarray(nu_cyc_per_mm, dtype=float)
    if _bessel_j1 is None:                            # pragma: no cover
        return np.exp(-(a ** 2) / 8.0)
    out = np.ones_like(a)
    nz = a != 0
    out[nz] = 2.0 * _bessel_j1(a[nz]) / a[nz]
    return np.abs(out)


def aperture_mtf(nu_cyc_per_mm, pixel_mm):
    """像素孔径 MTF：|sinc(π p ν)|，首零点 f₀ = 1/p（手册 §4.4）。"""
    return np.abs(_sinc(np.pi * float(pixel_mm) * np.asarray(nu_cyc_per_mm, dtype=float)))


def charge_mtf(nu_cyc_per_mm, sigma_c_mm):
    """电荷云扩散的 MTF：高斯核 → exp(−2π²σ²ν²)。"""
    nu = np.asarray(nu_cyc_per_mm, dtype=float)
    return np.exp(-2.0 * (np.pi ** 2) * (float(sigma_c_mm) ** 2) * nu ** 2)


# ---------------------------------------------------------------------------
# 样品面（对象面）系统 MTF 与反卷积
#
# 手册 §5 腿④：过采样只把可恢复频带**推向** f₀，模糊必须靠反卷积收口。
# 这里把三个模糊源都投影到**样品面**再合成，因此可以直接对重建图做反卷积：
#     焦点半影   b      = f·(M−1)/M          （已在样品面）
#     像素孔径   p/M                          （探测器像素投回样品面）
#     电荷云     σ_c/M                        （探测器面扩散投回样品面）
# ---------------------------------------------------------------------------

def object_plane_mtf(nu_cyc_per_mm, focus_um, pixel_um, sigma_c_um, M):
    """样品面的系统 MTF（焦点半影 · 像素孔径 · 电荷云三者乘积）。"""
    m = max(float(M), 1e-9)
    b_mm = float(focus_um) * max(m - 1.0, 0.0) / m / 1000.0
    ap_mm = float(pixel_um) / m / 1000.0
    sg_mm = float(sigma_c_um) / m / 1000.0
    nu = np.asarray(nu_cyc_per_mm, dtype=float)
    return disk_mtf(nu, b_mm) * aperture_mtf(nu, ap_mm) * charge_mtf(nu, sg_mm)


def deconvolve_image(img, voxel_um, focus_um=10.0, pixel_um=55.0, sigma_c_um=15.0,
                     M=4.0, snr=50.0):
    """按**系统自身建模的 MTF** 做维纳反卷积（模型驱动，非盲反卷积）。

    Wiener 权重： W(ν) = H(ν) / (H(ν)² + 1/SNR)

    * H 取样品面系统 MTF，与重建图同一坐标（体素 ν 由 voxel_um 决定）
    * SNR 越大越激进（越接近逆滤波，噪声放大越明显）；缺省 50 是保守值
    * MTF 是径向对称的（各模糊源均各向同性），故只用 |ν|
    * 体素数太少时直接原样返回，避免数值噪声
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 8:
        return img

    v_mm = max(float(voxel_um), 1e-6) / 1000.0
    ny, nx = a.shape
    fy = np.fft.fftfreq(ny, d=v_mm)
    fx = np.fft.fftfreq(nx, d=v_mm)
    NUY, NUX = np.meshgrid(fy, fx, indexing='ij')
    NU = np.sqrt(NUY ** 2 + NUX ** 2)

    H = object_plane_mtf(NU, focus_um, pixel_um, sigma_c_um, M)
    W = H / (H ** 2 + 1.0 / max(float(snr), 1e-6))
    out = np.real(np.fft.ifft2(np.fft.fft2(a) * W))

    # 反卷积会放大直流附近的低频；保持与原图同量纲（按有效增益归一）
    g = float(np.mean(H / (H ** 2 + 1.0 / max(float(snr), 1e-6))))
    if g > 1e-9:
        out = out / g
    return out


def fkn(e_kev):
    """Klein–Nishina 项（Compton 基函数，无单位形状函数）。"""
    e = np.asarray(e_kev, dtype=float) / 511.0
    e = np.maximum(e, 1e-6)
    term = (1.0 + e) / (e ** 2) * (2.0 * (1.0 + e) / (1.0 + 2.0 * e)
                                   - np.log(1.0 + 2.0 * e) / e)
    return term + np.log(1.0 + 2.0 * e) / (2.0 * e) - (1.0 + 3.0 * e) / ((1.0 + 2.0 * e) ** 2)


def mu_basis(e_kev, a_pe=1.0, b_kn=1.0):
    """两基材料衰减模型：μ(E) ≈ a·E⁻³（光电） + b·f_KN(E)（康普顿）。"""
    e = np.maximum(np.asarray(e_kev, dtype=float), 1e-6)
    return a_pe * e ** (-3.0) + b_kn * fkn(e)


# ===========================================================================
# 主分析入口
# ===========================================================================

def analyze(source='metaljet', focus_um=None, sod_mm=200.0, odd_mm=400.0,
            pixel_um=55.0, sigma_c_um=None, detector='CdTe',
            oversample=4, bins=8, energy_kev=60.0, flux_per_mm2=1e8,
            shaping_ns=None, ref_res_um=100.0, target_res_um=15.0,
            phase_enabled=False, delta_beta=100.0, vmi_kev=65.0,
            bin_edges=None, n_ch=825):
    """把整条同步辐射逼近链路算一遍，返回可读的派生量字典。

    参数
    ----
    source        源类型 key（见 SOURCES）
    focus_um      焦点尺寸 f（µm）；None → 取该源典型值
    sod_mm/odd_mm 源-物距 / 物-探距（mm）→ 放大率 M = (SOD+ODD)/SOD
    pixel_um      探测器物理像素 p（µm）
    sigma_c_um    电荷云半径 σ_c（µm）；None → 取材料中值
    detector      探测器材料 key（见 DETECTORS）
    oversample    子像素过采样帧数 K
    bins          能量箱数
    energy_kev    标称能量（用于通量/堆积与 VMI 参考）
    flux_per_mm2  入射光子面通量 φ（photons/mm²）
    shaping_ns    整形时间 τ（ns）；None → 取材料典型值
    ref_res_um    参考分辨率（现状基线，默认 100 µm = 东软 P10 实测）
    target_res_um 目标分辨率（默认 15 µm = 细胞级）
    phase_enabled 是否启用 in-line 相衬
    delta_beta    δ/β 比值（相衬灵敏度指标）
    vmi_kev       虚拟单能目标能量
    bin_edges     能量箱阈值序列（keV）

    返回
    ----
    dict：几何 / 采样 / MTF / 探测器物理 / 谱与相衬 / 剂量与信息量 / 结论 七组
    """
    src = SOURCES.get(str(source), SOURCES['metaljet'])
    det = DETECTORS.get(str(detector), DETECTORS['CdTe'])

    f_um = float(focus_um if focus_um is not None else src['focus_um'])
    sigma_um = float(sigma_c_um if sigma_c_um is not None
                     else 0.5 * (det['sigma_c_um'][0] + det['sigma_c_um'][1]))
    tau_ns = float(shaping_ns if shaping_ns is not None
                   else DEFAULT_SHAPING_NS.get(str(detector), 20.0))
    p_um = float(pixel_um)
    K = max(1, int(oversample))
    n_bins = max(1, int(bins))

    # ---- mm 制（频率、MTF 用）----
    p_mm = p_um / 1000.0
    f_mm = f_um / 1000.0
    sigma_mm = sigma_um / 1000.0

    # =====================================================================
    # 1. 几何（手册 §4.2 / §4.3）
    # =====================================================================
    sod, odd = float(sod_mm), float(odd_mm)
    M = (sod + odd) / sod if sod > 0 else 1.0
    p_eff_um = p_um / M                              # 样品处有效像素
    penumbra_um = f_um * (M - 1.0) / M               # 样品处几何半影 b

    # 总分辨率 r(M) = √( f²(M−1)² + p² ) / M（µm）
    def r_of(m):
        m = max(float(m), 1e-6)
        return float(np.sqrt((f_um ** 2) * ((m - 1.0) ** 2) + p_um ** 2) / m)

    r_cur = r_of(M)
    m_star = 1.0 + (p_um / f_um) ** 2 if f_um > 0 else float('inf')
    r_star = (p_um * f_um / np.sqrt(p_um ** 2 + f_um ** 2)) if (p_um > 0 and f_um > 0) else 0.0

    # r(M) 在 M<M* 单调下降且很快趋近 f：找"实用放大率"= 达到 r* 的 1.05 倍所需 M
    m_practical = None
    if f_um > 0:
        lo, hi = 1.0, max(2.0, min(m_star, 1e6))
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if r_of(mid) > 1.05 * r_star:
                lo = mid
            else:
                hi = mid
        m_practical = 0.5 * (lo + hi)

    # 极限行为：p≫f → 焦点受限 r*→f；p≪f → 像素受限 r*→p
    if p_um > 10.0 * f_um:
        limited_by = 'focus'
    elif f_um > 10.0 * p_um:
        limited_by = 'pixel'
    else:
        limited_by = 'balanced'

    geometry = {
        'M': float(M),
        'sod_mm': sod, 'odd_mm': odd,
        'p_eff_um': float(p_eff_um),
        'penumbra_um': float(penumbra_um),
        'r_um': float(r_cur),
        'M_star': float(m_star),
        'r_star_um': float(r_star),
        'M_practical': float(m_practical) if m_practical else None,
        'limited_by': limited_by,
        # FOV = 探测器有效宽度 / 放大率（手册 §4.2 失效模式：M 过大则 FOV 缩到装不下样品）
        'fov_mm': float(int(n_ch) * p_mm / M) if (n_ch and M > 0) else None,
        'n_ch': int(n_ch) if n_ch else None,
    }

    # =====================================================================
    # 2. 采样与过采样（手册 §4.1 / §4.4）
    # =====================================================================
    f0 = 1.0 / p_mm                                  # 孔径 MTF 首零点 cu/mm
    f_nyq_single = f0 / 2.0                          # 单帧 Nyquist
    f_nyq_over = K / (2.0 * p_mm)                    # 过采样后 Nyquist f_N(K)
    # K=2 即把可恢复频带填满到孔径零点（理想条件）→ K>2 无新增信息
    saturated_at = 2
    oversample_saturated = bool(K >= saturated_at)
    oversample_gain = float(min(f_nyq_over, f0) / f_nyq_single) if f_nyq_single > 0 else 1.0

    # 合成采样网格间距（µm）
    grid_um = p_um / K

    sampling = {
        'f0_cyc_mm': float(f0),
        'f_nyq_single': float(f_nyq_single),
        'f_nyq_over': float(f_nyq_over),
        'f_nyq_effective': float(min(f_nyq_over, f0)),
        'K': int(K),
        'grid_um': float(grid_um),
        'saturated_at_K': int(saturated_at),
        'oversample_saturated': oversample_saturated,
        'oversample_gain': float(oversample_gain),
        'nyquist_um': float(1000.0 / (2.0 * min(f_nyq_over, f0))) if f0 > 0 else None,
        'note': ('K≥2 已把可恢复频带填满到孔径零点 f₀=1/p，K>2 不再带来新增信息；'
                 '工程取 3–4 只为容纳位移误差与去卷积余量'),
    }

    # =====================================================================
    # 3. MTF 级联（手册 §3.1）
    # =====================================================================
    nu = np.linspace(1e-6, f0, 512)
    mtf_f = disk_mtf(nu, f_mm)
    mtf_ap = aperture_mtf(nu, p_mm)
    mtf_ch = charge_mtf(nu, sigma_mm)
    mtf_sys = mtf_f * mtf_ap * mtf_ch
    # 10% MTF 截止频率：工程口径的"可分辨"判据
    def _cutoff(mtf, frac=0.10):
        idx = np.where(mtf < frac)[0]
        return float(nu[idx[0]]) if len(idx) else float(nu[-1])
    mtf10 = _cutoff(mtf_sys, 0.10)
    mtf50 = _cutoff(mtf_sys, 0.50)

    # 频率→分辨率换算：ν [cyc/mm] 的周期 = 1000/ν µm，可分辨特征 ≈ 半周期 = 500/ν µm。
    # 用这个口径校验：333 lp/cm = 33.3 cyc/mm → 500/33.3 = 15.0 µm ✓（手册 §2.1）
    def _res_um(nu_cyc_mm):
        return float(500.0 / nu_cyc_mm) if nu_cyc_mm > 0 else None

    mtf = {
        'mtf_at_f0': float(np.interp(f0, nu, mtf_sys)),      # 孔径零点处，理想情况下为 0
        'mtf10_cyc_mm': mtf10,
        'mtf50_cyc_mm': mtf50,
        'lp_cm_at_mtf10': float(mtf10 * 10.0),               # 1 cyc/mm = 10 lp/cm
        'res_at_mtf10_um': _res_um(mtf10),
        'res_at_mtf50_um': _res_um(mtf50),
        'components': ('MTF_sys = MTF_focus · MTF_aperture · MTF_charge；'
                       '过采样只移动可恢复频带上限，不改变各 MTF 形状'),
    }

    # =====================================================================
    # 4. 探测器物理：电荷共享 / K 荧光 / 脉冲堆积
    # =====================================================================
    # 4.1 电荷共享判据 p ≳ 2σ_c（手册 §4.5）
    spectral_ok = bool(p_um >= 2.0 * sigma_um)
    p_crit_um = 2.0 * sigma_um
    # 经验残差模型：p < 2σ 时能谱判别按偏离度衰减（示意，非实测）
    ratio = p_um / max(p_crit_um, 1e-9)
    spectral_retention = float(min(1.0, ratio ** 2)) if ratio < 1.0 else 1.0

    # 4.2 K 荧光逃逸：λ_K 远大于像素时污染邻域 bin
    lam_um = float(det['lambda_K_um'])
    escape_pixels = lam_um / max(p_um, 1e-9)
    kescape_severe = bool(lam_um > 2.0 * p_um)

    # 4.3 脉冲堆积：计数率 ≪ 1/τ 才可忽略（手册 §4.7）
    tau_s = tau_ns * 1e-9
    max_rate = 1.0 / tau_s if tau_s > 0 else float('inf')
    # 每像素入射光子率估计：φ[ph/mm²] × 像素面积[mm²] × 时间(取 1 s)
    pix_area_mm2 = p_mm ** 2
    rate_per_pixel = float(flux_per_mm2) * pix_area_mm2
    pileup_fraction = float(rate_per_pixel / max_rate) if max_rate else 0.0
    pileup_ok = bool(pileup_fraction < 0.1)

    detector_phys = {
        'material': str(detector),
        'material_label': det['label'],
        'sigma_c_um': sigma_um,
        'p_over_sigma': float(p_um / max(sigma_um, 1e-9)),
        'p_crit_um': float(p_crit_um),
        'spectral_ok': spectral_ok,
        'spectral_retention': spectral_retention,
        'lambda_K_um': lam_um,
        'lambda_Ka1_keV': float(det['Ka1_keV']),
        'escape_pixels': float(escape_pixels),
        'kescape_severe': kescape_severe,
        'shaping_ns': tau_ns,
        'max_rate_cps': float(max_rate),
        'rate_per_pixel_cps': rate_per_pixel,
        'pileup_fraction': pileup_fraction,
        'pileup_ok': pileup_ok,
    }

    # =====================================================================
    # 5. 能谱 bin 与 VMI（手册 §4.8 / §4.9）
    # =====================================================================
    edges = tuple(float(e) for e in (bin_edges or DEFAULT_BIN_EDGES))
    if len(edges) < n_bins + 1:
        # 不够就按标称能量等比外推补齐
        extra = n_bins + 1 - len(edges)
        edges = edges + tuple(edges[-1] + (i + 1) * (edges[-1] - edges[-2] if len(edges) > 1 else 20.0)
                              for i in range(extra))
    edges = edges[:n_bins + 1]
    centers = np.array([0.5 * (edges[i] + edges[i + 1]) for i in range(n_bins)])

    # VMI：求权重 w 使 Σ w_i·basis(Ē_i) ≈ basis(E_vmi)（对光电与康普顿两基同时成立）
    # 这是手册 §4.8 "线性组合合成等效单能"的最小二乘实现
    basis = np.vstack([mu_basis(centers, a_pe=1.0, b_kn=0.0),
                       mu_basis(centers, a_pe=0.0, b_kn=1.0)])       # (2, n_bins)
    target = np.array([[mu_basis(vmi_kev, a_pe=1.0, b_kn=0.0)],
                       [mu_basis(vmi_kev, a_pe=0.0, b_kn=1.0)]])      # (2, 1)
    try:
        w, *_ = np.linalg.lstsq(basis, target, rcond=None)
        w = w.ravel()
        vmi_resid = float(np.linalg.norm(basis @ w - target.ravel()))
        vmi_ok = bool(vmi_resid < 1e-3 * max(1e-9, float(np.linalg.norm(target))))
    except Exception:                                # pragma: no cover
        w = np.full(n_bins, 1.0 / n_bins)
        vmi_resid = float('nan')
        vmi_ok = False

    spectrum = {
        'bins': int(n_bins),
        'bin_edges_keV': list(edges),
        'bin_centers_keV': [float(c) for c in centers],
        'bin_width_keV': [float(edges[i + 1] - edges[i]) for i in range(n_bins)],
        'vmi_keV': float(vmi_kev),
        'vmi_weights': [float(x) for x in w],
        'vmi_residual': vmi_resid,
        'vmi_ok': vmi_ok,
        'note': ('bin 能窗（数 keV 至数十 keV）与真单色光有本质差距；'
                 '电荷共享与 K 荧光会污染 bin 纯净度'),
    }

    # =====================================================================
    # 6. in-line 相衬（手册 §4.10）
    # =====================================================================
    # Paganin 单距离相位恢复：滤波核 H(ν) = 1/(1 + (δ/β)·(λ·R·ν²)/(4π))
    # 波长 λ(mm) = 1.2398e-6 / E(keV)
    lam_mm = 1.23984e-6 / max(energy_kev, 1e-9)
    phase = {
        'enabled': bool(phase_enabled),
        'wavelength_mm': float(lam_mm),
        'delta_over_beta': float(delta_beta),
        'propagation_mm': float(odd),
        'paganin_alpha_mm2': float(delta_beta * lam_mm * odd / (4.0 * np.pi)),
        'coherent': bool(src['coherent']),
        'note': ('需要足够空间相干性：同步辐射远优于实验室微焦点；'
                 '实验室源的通量与几何约束使灵敏度/视场/实用性均低于同步辐射'),
    }
    if not src['coherent'] and phase_enabled:
        phase['warning'] = '当前源相干性不足，相衬收益按 0 计（仅弱吸收对比可用）'

    # =====================================================================
    # 7. 剂量幂律 / 通量 / 信息量（手册 §4.12 / §4.13 / §8 M6）
    # =====================================================================
    N = float(ref_res_um) / max(target_res_um, 1e-9)          # 线性分辨率提升倍数
    dose_n3 = N ** 3
    dose_n4 = N ** 4

    b_src = src['brightness']
    b_lab = SOURCES['rotating']['brightness']
    gap_lo = b_src[0] / b_lab[1]
    gap_hi = b_src[1] / b_lab[0]
    noise_gap = float(np.sqrt(0.5 * (gap_lo + gap_hi)))

    # Swank 因子（能谱响应二阶矩的近似）：电荷共享与 K 荧光都降低它
    swank = float(np.clip(spectral_retention * (0.85 if kescape_severe else 0.97), 0.05, 1.0))
    # 理想探测器下 DQE(ν) = A_S · MTF_sys²(ν)；NEQ = φ·DQE（白噪声归一）
    # 注意：不能在 f₀=1/p 处报 DQE —— 那里 MTF_ap 恰为首零点，MTF≡0 使指标失去意义。
    # 取半 Nyquist 作为报告频点（DQE 的常规报告口径）。
    nu_report = 0.5 * min(f_nyq_over, f0)
    mtf_report = float(np.interp(nu_report, nu, mtf_sys))
    dqe_report = float(swank * (mtf_report ** 2))
    neq_report = float(flux_per_mm2 * dqe_report)
    # 香农容量 C = ∫ log2(1+NEQ) dν，NEQ(ν)=φ·A_S·MTF²(ν)
    neq_curve = float(flux_per_mm2) * swank * (mtf_sys ** 2)
    shannon = float(_trapz(np.log2(1.0 + neq_curve), nu))

    dose = {
        'ref_res_um': float(ref_res_um),
        'target_res_um': float(target_res_um),
        'N': N,
        'dose_x_n3': float(dose_n3),
        'dose_x_n4': float(dose_n4),
        'clinical_tolerance': '1–2×',
        'clinical_verdict': ('人体场景被剂量预算直接否决'
                             if dose_n3 > 2.0 else '剂量预算内'),
        'exvivo_note': '离体场景剂量不再是约束，但换成"通量 × 时间"的约束',
        'brightness_src': list(b_src),
        'brightness_lab': list(b_lab),
        'brightness_gap_lo': float(gap_lo),
        'brightness_gap_hi': float(gap_hi),
        'brightness_gap_decades': float(np.log10(np.sqrt(gap_lo * gap_hi))),
        'noise_gap': noise_gap,
        'swank': swank,
        'dqe_report_freq': float(nu_report),
        'mtf_at_report_freq': mtf_report,
        'dqe_at_half_nyq': dqe_report,
        'neq_at_half_nyq': neq_report,
        'shannon_bits_per_mm': shannon,
        'note': ('DQE ≤ 1 为物理约束；NEQ/DQE 基于"白噪声归一 + Swank 因子"的简化模型，'
                 '不是实测 NPS 标定值。报告频点取半 Nyquist —— f₀=1/p 处 MTF_ap 为首零点，'
                 '在那里报 DQE 会恒为 0，无意义'),
    }

    # =====================================================================
    # 8. 结论：这次配置能到哪一档
    # =====================================================================
    r_best = float(min(r_star, r_cur)) if f_um > 0 else float(r_cur)
    reachable = {
        'r_current_um': float(r_cur),
        'r_best_um': r_best,
        'r_best_at': 'M*' if abs(M - m_star) < 1e-6 else 'current M',
        'target_um': float(target_res_um),
        'target_met': bool(r_best <= target_res_um),
        'lp_cm_at_r_best': float(1e4 / r_best) if r_best > 0 else None,
        'bottleneck': ({'focus': '焦点尺寸（真正的杠杆是 f，不是 M）',
                       'pixel': '像素孔径（f₀=1/p 是过采样的天花板）',
                       'balanced': '焦点与像素量级相当，两者都要压'}[limited_by]),
    }

    return {
        'source': str(source),
        'source_label': src['label'],
        'source_note': src['note'],
        'focus_um': f_um,
        'geometry': geometry,
        'sampling': sampling,
        'mtf': mtf,
        'detector': detector_phys,
        'spectrum': spectrum,
        'phase': phase,
        'dose': dose,
        'reachable': reachable,
    }


def summary_lines(res):
    """把 analyze() 的结果压成几行中文，供 GUI / 日志直接显示。"""
    g, s, m = res['geometry'], res['sampling'], res['mtf']
    d, r = res['detector'], res['reachable']
    out = [
        f"源：{res['source_label']}　焦点 f = {res['focus_um']:.1f} µm",
        f"几何：M = {g['M']:.2f}（M* = {g['M_star']:.0f}）　有效像素 p/M = {g['p_eff_um']:.2f} µm",
        f"　　　半影 b = {g['penumbra_um']:.2f} µm　合成分辨率 r(M) = {g['r_um']:.2f} µm"
        f"（r* = {g['r_star_um']:.2f} µm）",
        f"采样：孔径零点 f₀ = {s['f0_cyc_mm']:.1f} cu/mm　K = {s['K']}"
        f"　有效 Nyquist = {s['f_nyq_effective']:.1f} cu/mm",
        f"MTF：10% 截止 = {m['mtf10_cyc_mm']:.1f} cu/mm（{m['lp_cm_at_mtf10']:.0f} lp/cm"
        f" ≈ {m['res_at_mtf10_um']:.1f} µm）" if m['res_at_mtf10_um'] else "MTF：—",
        f"探测器：{d['material_label']}　p/σ_c = {d['p_over_sigma']:.2f}"
        f"　能谱{'可用' if d['spectral_ok'] else '已失效'}",
        f"　　K 荧光 λ_K = {d['lambda_K_um']:.0f} µm ≈ {d['escape_pixels']:.1f} 像素"
        f"　堆积率 = {d['pileup_fraction']*100:.1f}%",
        f"可达：{r['r_best_um']:.2f} µm"
        f"（{r['lp_cm_at_r_best']:.0f} lp/cm）　目标 {r['target_um']:.0f} µm "
        f"{'达成' if r['target_met'] else '未达成'}　瓶颈：{r['bottleneck']}",
    ]
    return out
