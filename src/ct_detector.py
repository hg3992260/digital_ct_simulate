# -*- coding: utf-8 -*-
"""M3 · 探测器响应矩阵（纯 numpy + xraylib，无 Qt）。

依据《PCCT 模拟同步辐射 CT · 算法交接手册》§8：

    M3 探测器响应（依赖 M2）
      * 输入：探测器材料、像素 p、电荷云 σ_c、能量阈值
      * 输出：**单能响应矩阵 R[E, k]**
      * 验收：R 的行和 = 该能量下的总探测效率；**K 边处应有跳变**

为什么它是最后一块物理缺口
--------------------------
手册 §1.2 把"无可执行电荷共享模型"列为缺口之一。没有 R[E,k]：
  * 箱分辨重建无从谈起（不知道光子落在哪一箱）
  * VMI / K 边成像只能停留在"模型值"层面
  * K 荧光逃逸对能谱的污染无法量化

物理链条（每一环都可独立核对）
------------------------------
    入射能量 E
      → 线衰减 μ(E) = μ_photo + μ_Compt + μ_Rayl        （xraylib 实测截面）
      → 吸收概率 η(E) = 1 − exp(−μ·t)                    ← 行和的上限
      → 作用类型：光电 f_ph / 康普顿 f_co
          · 光电：沉积 E；以 P_K 概率 Kα 逃逸 → 沉积 E − E_Kα
          · 康普顿：按 Klein–Nishina 抽样沉积能量（部分逃逸）
      → 电荷共享：p ≲ 2σ_c 时部分电荷落入邻像素、可能跌破阈值 → 计数损失
      → 按阈值分箱 → R[E, k]

K 边跳变是**实测截面**带来的：xraylib 的 CS_Photo 在 E_K 处有真实的不连续，
不是人为加的台阶。
"""

import numpy as np

try:
    import xraylib as _xr
except Exception:                                    # pragma: no cover
    _xr = None

try:
    import sys as _sys
    _sys.stdout.reconfigure(errors='replace')
except Exception:
    pass


# 材料成分（质量分数按化学计量比换算；元素用 Z 表示）
COMPOSITION = {
    'CdTe': {'Cd': 1.0, 'Te': 1.0},
    'CZT': {'Cd': 0.9, 'Zn': 0.1, 'Te': 1.0},
    'Si': {'Si': 1.0},
    'GaAs': {'Ga': 1.0, 'As': 1.0},
    'Se': {'Se': 1.0},
}

# 与 ct_synchrotron.DETECTORS 保持一致的 K 边 / Kα1（keV）
K_EDGE = {'CdTe': 26.7112, 'CZT': 26.7112, 'GaAs': 10.3671, 'Si': 1.8389, 'Se': 12.6578}
KA1 = {'CdTe': 23.1736, 'CZT': 23.1736, 'GaAs': 9.2517, 'Si': 1.7399, 'Se': 11.2224}

_ATOMIC = {'Cd': 48, 'Te': 52, 'Zn': 30, 'Si': 14, 'Ga': 31, 'As': 33, 'Se': 34}
_ATOMW = {'Cd': 112.414, 'Te': 127.60, 'Zn': 65.38, 'Si': 28.085, 'Ga': 69.723,
          'As': 74.922, 'Se': 78.971}

# K 荧光产额（近似：Z 越大越高；Cd/Te 约 0.83/0.87）
_K_YIELD = {'Cd': 0.83, 'Te': 0.87, 'Zn': 0.48, 'Si': 0.05, 'Ga': 0.55, 'As': 0.58,
            'Se': 0.60}


def _mass_fractions(material):
    comp = COMPOSITION.get(str(material))
    if not comp:
        raise ValueError('unknown detector material: %s' % material)
    tot = sum(_ATOMW[e] * n for e, n in comp.items())
    return {e: (_ATOMW[e] * n) / tot for e, n in comp.items()}, comp


def _xsec(elem, energy_kev, kind):
    """元素截面 (cm²/g)。kind: 'photo' | 'compt' | 'rayl'。"""
    if _xr is None:
        return None
    z = _ATOMIC[elem]
    fn = {'photo': ('CS_Photo', 'CS_Photo_Total'),
          'compt': ('CS_Compt', 'CS_Compton'),
          'rayl': ('CS_Rayl', 'CS_Rayleigh')}[kind]
    for name in fn:
        f = getattr(_xr, name, None)
        if f is None:
            continue
        try:
            return float(f(z, float(energy_kev)))
        except Exception:
            continue
    return None


def _fallback_xsec(elem, energy_kev, kind):
    """xraylib 不可用时的解析近似（仅保证趋势正确，不做定量依据）。"""
    z = _ATOMIC[elem]
    E = max(float(energy_kev), 1e-3)
    if kind == 'photo':
        s = (z ** 4.5) / (E ** 3) * 1.0e-4
        # 人为加 K 边跳变，保证趋势与真实一致
        ek = {'Cd': 26.7112, 'Te': 31.8138, 'Zn': 9.6586, 'Si': 1.8389,
              'Ga': 10.3671, 'As': 11.8637, 'Se': 12.6578}.get(elem)
        if ek and E >= ek:
            s *= 5.0
        return s
    if kind == 'compt':
        return 0.05 * z / max(E, 0.05)
    return 0.01 * (z ** 2) / (E ** 2)


# ---------------------------------------------------------------------------
# 折射率：n = 1 − δ + iβ（xraylib 实算）
#
# 相衬路径需要 δ；衰减路径需要 β（且 μ = 4πβ/λ 必须自洽 —— 这是一条现成的
# 交叉校验，实测水在 33 keV 下 β=9.866e-11 → μ=0.0330/mm，与截面路径一致）。
#
# 组织成分按 ICRU-44 质量分数写成 xraylib 配方串；密度取各组织典型值。
# 之前用的是"文献典型 δ"，量级对但无法区分组织；改成实算后 δ 由元素组成与
# 密度直接决定，髓鞘（脂质富集）与灰质的 δ 差才真正体现出来。
# ---------------------------------------------------------------------------
TISSUE_FORMULA = {
    'water':        ('H2O', 1.000),
    'gray_matter':  ('H10.5C14.5N2.2O71.2P0.4S0.2Cl0.3K0.3Na0.2', 1.040),
    'white_matter': ('H10.5C15.5N2.3O69.0P0.5S0.2Cl0.3K0.3Na0.2', 1.030),
    'myelin':       ('H11.5C19.0N2.0O66.0P0.5S0.5', 0.900),
    'csf':          ('H11.1O88.8Na0.5Cl0.5', 1.007),
    'blood':        ('H10.2C11.0N3.3O74.5Na0.1P0.1S0.2Cl0.3K0.2Fe0.1', 1.060),
}


def formula_of(name):
    """返回 (xraylib 配方串, 密度 g/cm³)。未知名称原样当作配方串。"""
    if str(name) in TISSUE_FORMULA:
        return TISSUE_FORMULA[str(name)]
    return str(name), 1.0


def delta_beta(name, energy_kev, density_gcc=None):
    """返回 (δ, β)：n = 1 − δ + iβ。用 xraylib 实算，非文献典型值。"""
    comp, rho = formula_of(name)
    if density_gcc is not None:
        rho = float(density_gcc)
    if _xr is not None:
        try:
            re = float(_xr.Refractive_Index_Re(comp, float(energy_kev), float(rho)))
            im = float(_xr.Refractive_Index_Im(comp, float(energy_kev), float(rho)))
            # xraylib 的 Refractive_Index_Im 返回的**已经是 +β**（不是 −β）。
            # 早先按 -im 取，得到负 β → Paganin 的 α 为负 → 反演发散到 clip 上限。
            return 1.0 - re, abs(im)
        except Exception:
            pass
    # 回退：δ 用电子密度近似，β 由已知 μ 反推
    lam_mm = 1.23984e-6 / max(float(energy_kev), 1e-9)
    mu_ = mu_linear(name, energy_kev, rho) if str(name) in COMPOSITION else 0.033
    beta = mu_ * lam_mm / (4.0 * np.pi)
    return float(2.12e-7 * (rho) * (33.0 / float(energy_kev)) ** 2), float(beta)


def mu_from_beta(beta, energy_kev):
    """μ = 4πβ/λ（1/mm）—— 用于校验折射率与截面两条路径是否自洽。"""
    return float(4.0 * np.pi * float(beta) / (1.23984e-6 / max(float(energy_kev), 1e-9)))


def mu_over_rho(material, energy_kev):
    """质量衰减系数 (cm²/g)。"""
    w, _ = _mass_fractions(material)
    tot = 0.0
    for e, wi in w.items():
        for kind in ('photo', 'compt', 'rayl'):
            s = _xsec(e, energy_kev, kind)
            if s is None:
                s = _fallback_xsec(e, energy_kev, kind)
            tot += wi * s
    return float(tot)


def mu_linear(material, energy_kev, density_gcc=None):
    """线衰减系数 (1/mm)。"""
    w, _ = _mass_fractions(material)
    if density_gcc is None:
        if _xr is not None:
            try:
                rho = sum(wi * float(_xr.ElementDensity(_ATOMIC[e]))
                          for e, wi in w.items())
            except Exception:
                rho = 5.85
        else:
            rho = 5.85
    else:
        rho = float(density_gcc)
    return float(mu_over_rho(material, energy_kev) * rho / 10.0)   # cm→mm


def absorption_efficiency(material, energy_kev, thickness_um=1000.0):
    """吸收（探测）效率 η(E) = 1 − exp(−μ·t)。"""
    return float(1.0 - np.exp(-mu_linear(material, energy_kev) * thickness_um / 1000.0))


def photo_fraction(material, energy_kev):
    """光电占比 f_ph = μ_photo / μ_total。康普顿占 1−f_ph。"""
    w, _ = _mass_fractions(material)
    ph = co = 0.0
    for e, wi in w.items():
        for kind, acc in (('photo', 'ph'), ('compt', 'co'), ('rayl', 'co')):
            s = _xsec(e, energy_kev, kind)
            if s is None:
                s = _fallback_xsec(e, energy_kev, kind)
            if acc == 'ph':
                ph += wi * s
            else:
                co += wi * s
    tot = ph + co
    return float(ph / tot) if tot > 0 else 0.0


def charge_loss(pixel_um, sigma_c_um, k_sigma=2.0):
    """电荷共享造成的**计数损失**比例。

    判据沿用已验收的 p ≳ 2σ_c：像素明显大于电荷云时几乎无损失；
    像素与电荷云同量级时，事件落在像素边缘会有一部分电荷分给邻像素，
    使总收集电荷跌破最低阈值 → 该事件不被计数。
    """
    p, s = float(pixel_um), max(float(sigma_c_um), 1e-6)
    # 边缘环带（宽度 ~2σ_c）占像素面积的比例，即最可能发生显著共享的事件占比
    frac_edge = min(1.0, 2.0 * k_sigma * s / p)
    if frac_edge >= 1.0:
        return 0.5 * min(1.0, (k_sigma * s) / p)        # 电荷云远大于像素
    # 只有边缘环带中真正跨越边界的那一部分会掉出阈值
    return float(0.5 * frac_edge ** 2)


def total_efficiency(material, energy_kev, thickness_um=1000.0, pixel_um=55.0,
                     sigma_c_um=15.0):
    """总探测效率（含电荷共享损失）—— R 的行和应当等于它。"""
    return float(absorption_efficiency(material, energy_kev, thickness_um)
                 * (1.0 - charge_loss(pixel_um, sigma_c_um)))


def compton_deposit_spectrum(E_kev, n=64):
    """康普顿作用后沉积能量的近似分布（Klein–Nishina 抽样）。

    返回 (沉积能量数组 keV, 权重)。小像素下散射光子大概率逃逸，
    故沉积能量在 0 与康普顿边之间展宽。
    """
    E = float(E_kev)
    mec2 = 510.9989                                    # keV
    alpha = E / mec2
    # 散射角 θ 的 KN 微分截面 ∝ (1+α(1−cosθ))^-2 · (1+cos²θ+α²(1−cosθ)²/(1+α(1−cosθ)))
    th = np.linspace(1e-4, np.pi, int(n))
    c = np.cos(th)
    d = 1.0 + alpha * (1.0 - c)
    kn = (1.0 / d ** 2) * (1.0 + c ** 2 + (alpha ** 2) * (1.0 - c) ** 2 / d) * np.sin(th)
    kn = np.maximum(kn, 0.0)
    w = kn / (kn.sum() or 1.0)
    # 沉积能量 = E − E'，其中 E' = E / (1 + α(1−cosθ))
    e_scatter = E / d
    deposit = E - e_scatter
    return deposit, w


def detector_response(material='CdTe', thickness_um=1000.0, pixel_um=55.0,
                      sigma_c_um=15.0, thresholds_kev=(20.0, 35.0, 50.0, 70.0, 100.0),
                      energies_kev=None, n_e=400, escape_prob=None):
    """M3 核心：单能响应矩阵 R[E, k]。

    参数
    ----
    material      : CdTe / CZT / Si / GaAs / Se
    thickness_um  : 探测器厚度（吸收效率的指数项）
    pixel_um      : 像素尺寸（决定电荷共享损失）
    sigma_c_um    : 电荷云 σ
    thresholds_kev: 能量箱阈值；k 号箱 = [T_k, T_{k+1})，共 len(T)−1 箱
    energies_kev  : 入射能量网格；缺省 5–160 keV
    escape_prob   : Kα 逃逸概率；缺省按荧光产额与像素/逃逸长度比估计

    返回
    ----
    dict: energies / thresholds / R (n_e, n_bins) / eff_total / mu / photo_frac
          / k_edge_kev / ka1_kev / escape_prob / charge_loss
    """
    mat = str(material)
    if energies_kev is None:
        energies_kev = np.linspace(5.0, 160.0, int(n_e))
    E = np.asarray(energies_kev, dtype=float)
    T = np.asarray(thresholds_kev, dtype=float)
    if T.size < 2:
        raise ValueError('至少需要两个阈值才能构成能量箱')
    T = np.sort(T)
    n_bins = T.size - 1

    ek = K_EDGE.get(mat)
    ka = KA1.get(mat, 0.0)
    if escape_prob is None:
        # 逃逸概率：随荧光产额上升，随"逃逸长度/像素"下降（逃逸光子越容易跑出）
        y = _K_YIELD.get('Cd', 0.6)
        if mat in ('GaAs', 'Si', 'Se'):
            y = _K_YIELD.get({'GaAs': 'Ga', 'Si': 'Si', 'Se': 'Se'}[mat], 0.5)
        ratio = min(1.0, 249.0 / max(float(pixel_um), 1.0)) if mat in ('CdTe', 'CZT') \
            else min(1.0, 40.0 / max(float(pixel_um), 1.0))
        escape_prob = float(0.5 * y * ratio)
    pc = float(charge_loss(pixel_um, sigma_c_um))

    R = np.zeros((E.size, n_bins), dtype=float)
    mu_arr = np.zeros(E.size)
    eff = np.zeros(E.size)
    fph = np.zeros(E.size)
    below = np.zeros(E.size)                 # 沉积能量跌破最低阈值而丢失的权重

    for i, e in enumerate(E):
        mu_arr[i] = mu_linear(mat, e)
        eff[i] = total_efficiency(mat, e, thickness_um, pixel_um, sigma_c_um)
        fph[i] = photo_fraction(mat, e)
        if eff[i] <= 0:
            continue

        # --- 光电：全能量沉积；Kα 逃逸时少沉积一个 Kα ---
        # K 荧光只在 E **超过 K 边**时才可能发生（低于 K 边无法电离 K 壳层）。
        # 早先版本无条件施加逃逸，导致 25 keV（< E_K=26.7）就有 38% 的事件
        # 被"逃逸"到 1.83 keV 而跌破阈值 —— 凭空造出巨大的阈值损失。
        esc_ok = bool(ka > 0 and ek and e > ek)
        esc_p = escape_prob if esc_ok else 0.0
        e_ph_full = e
        e_ph_esc = max(e - ka, 0.0) if esc_ok else e
        # --- 康普顿：沉积能量分布 ---
        dep, w = compton_deposit_spectrum(e, n=48)
        w = w * (1.0 - fph[i])
        w = np.append(w, [fph[i] * (1.0 - esc_p), fph[i] * esc_p])
        dep = np.append(dep, [e_ph_full, e_ph_esc])

        lost = 0.0
        for d, wi in zip(dep, w):
            if wi <= 0:
                continue
            k = int(np.searchsorted(T, d, side='right') - 1)
            if 0 <= k < n_bins:
                R[i, k] += wi
            else:
                # 沉积能量落在所有能量箱之外（低于最低阈值）→ 该事件不被计数。
                # 必须如实记账：早先版本在这里做 R[i] *= eff/s 的再归一化，
                # 等于把丢失的事件又补回效率里，把阈值损失整个掩盖掉了。
                lost += wi
        below[i] = lost
        # 按**绝对效率**缩放：权重本身已归一，故行和 = eff × (落在箱内的比例)
        R[i] *= eff[i]

    return {
        'material': mat,
        'energies_kev': E,
        'thresholds_kev': T,
        'R': R,
        'eff_total': eff,
        'below_threshold_frac': below,
        'eff_counted': R.sum(axis=1),
        'mu_per_mm': mu_arr,
        'photo_frac': fph,
        'k_edge_kev': ek,
        'ka1_kev': ka,
        'escape_prob': float(escape_prob),
        'charge_loss': pc,
        'thickness_um': float(thickness_um),
        'pixel_um': float(pixel_um),
        'sigma_c_um': float(sigma_c_um),
    }


def validate_m3(material='CdTe', thickness_um=1000.0, pixel_um=55.0, sigma_c_um=15.0,
                thresholds_kev=(20.0, 35.0, 50.0, 70.0, 100.0), n_e=400,
                tol_rowsum=1e-9, jump_min=1.5):
    """手册 §8 M3 验收。

    判据 1：R 的行和 = 该能量下的总探测效率（逐能量点核对）
    判据 2：K 边处应有跳变 —— 线衰减系数在 E_K 处的不连续比 ≥ jump_min
    """
    r = detector_response(material, thickness_um, pixel_um, sigma_c_um,
                          thresholds_kev, n_e=n_e)
    R, eff = r['R'], r['eff_total']
    below = r['below_threshold_frac']
    row = R.sum(axis=1)

    # 判据 1a：行和 + 阈值以下丢失 = 总探测效率（**概率守恒**）。
    # 早先版本在 R 上做了再归一化，等于把丢失补回效率里，这条检查会假通过。
    cons = np.abs(row + eff * below - eff)
    ok1a = bool(cons.max() < 1e-9)
    # 判据 1b：任何能量下都不得"计得比吸收的多"
    ok1b = bool(np.all(row <= eff + 1e-12))
    # 行和相对总效率的偏差（含阈值损失，物理上应 ≤ 1）
    rel = np.abs(row - eff) / np.maximum(np.abs(eff), 1e-12)

    # K 边跳变：在 E_K 两侧各取一点量 μ 的比值
    ek = r['k_edge_kev']
    jump = None
    ok2 = None
    if ek:
        e_lo = ek * 0.995
        e_hi = ek * 1.005
        m_lo = mu_linear(material, e_lo)
        m_hi = mu_linear(material, e_hi)
        jump = float(m_hi / m_lo) if m_lo > 0 else None
        ok2 = bool(jump is not None and jump >= float(jump_min))

    # 附带：K 边上下吸收效率的跳变
    eff_jump = None
    if ek:
        eff_jump = float(absorption_efficiency(material, ek * 1.005, thickness_um)
                         / max(absorption_efficiency(material, ek * 0.995, thickness_um), 1e-12))

    return {
        'response': r,
        'rowsum_rel_err_max': float(rel.max()),
        'rowsum_rel_err_mean': float(rel.mean()),
        'conservation_err_max': float(cons.max()),
        'below_threshold_max': float(np.max(below)),
        'k_edge_kev': ek,
        'mu_jump_ratio': jump,
        'eff_jump_ratio': eff_jump,
        'crit1a_conservation': ok1a,
        'crit1b_never_exceeds': ok1b,
        'crit2_kedge_jump': bool(ok2) if ok2 is not None else False,
        'passed': bool(ok1a and ok1b and (ok2 is None or ok2)),
    }
