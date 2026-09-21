# -*- coding: utf-8 -*-
"""Fermi ↔ 光子计数 CT 的物理桥接层。

把 `I:\\dsw\\Fermi`（半导体探测器响应模拟）封装成光子计数架构可直接调用的形式：

    FermiModel.spectrum(...)        单点能谱响应（材料/厚度/偏压/温度/入射能量）
    FermiModel.bins(...)            能量箱阈值（等宽 / K 边优化）
    FermiModel.response_matrix(...) R[E_src, bin] 能量箱响应矩阵
    FermiModel.source_spectrum(...) 入射 X 射线谱（kVp 经验模型）
    FermiModel.bin_weights(...)     Σ_E φ(E)·R[E,k] —— 直接供 FBP 分箱使用

物理要点
--------
* 记录能量 = 沉积能量 + 电子学/暗电流噪声展宽（Fermi 的 ENC/FWHM 已含）
* bin 响应 = 把 `simulate_spectrum` 给出的记录谱按阈值区间积分
* 光子计数 CT 的第 k 箱正弦图：  g_k = Σ_E φ(E)·exp(-∫μ dl)·R[E,k]
  本模块产出 R 与箱权重，LEAP 侧只需对各箱做常规 FBP 再分解。
"""

import os
import sys
import numpy as np

def _find_fermi_root():
    """定位 Fermi 子项目根目录（需含有 src/）。

    源码运行：src/../Fermi。
    冻结运行：Fermi 被 --add-data "Fermi;Fermi" 打进 _internal/Fermi，
    此时 __file__ 在 _MEIPASS 下，不能再靠 dirname(dirname(...)) 推断。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.join(os.path.dirname(here), 'Fermi'),   # 源码布局
             os.path.join(here, 'Fermi')]
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        cands.append(os.path.join(meipass, 'Fermi'))
    if getattr(sys, 'frozen', False):
        exedir = os.path.dirname(os.path.abspath(sys.executable))
        cands += [os.path.join(exedir, 'Fermi'),
                  os.path.join(exedir, '_internal', 'Fermi')]
    for c in cands:
        if os.path.isdir(os.path.join(c, 'src')):
            return c
    return cands[0]


FERMI_ROOT = _find_fermi_root()
_SIM = {}


def _load_fermi():
    """载入 Fermi 的 simulator 模块（其内部为顶层导入，需把 src 加入 sys.path）。"""
    if 'sim' in _SIM:
        return _SIM['sim'], _SIM['mm']
    root, src = FERMI_ROOT, os.path.join(FERMI_ROOT, 'src')
    for p in (src, root):
        if p not in sys.path:
            sys.path.insert(0, p)
    cwd = os.getcwd()
    try:
        os.chdir(root)                     # MaterialManager 默认相对路径 data/materials.json
        from simulator import DetectorSimulator
        from material import MaterialManager
    finally:
        os.chdir(cwd)
    _SIM['sim'], _SIM['mm'] = DetectorSimulator, MaterialManager
    return DetectorSimulator, MaterialManager


def _as_pair(obj, keys, n_fallback=64, span=1.0):
    """Fermi 各方法返回值不统一（dict 或 (x, y) 元组）→ 统一取出 (x, y) 两个数组。"""
    if isinstance(obj, dict):
        for kx in keys[0]:
            if kx in obj:
                x = np.asarray(obj[kx], float)
                for ky in keys[1]:
                    if ky in obj:
                        return x, np.asarray(obj[ky], float)
                break
        vals = [v for v in obj.values() if isinstance(v, (list, tuple, np.ndarray))
                and np.asarray(v).ndim >= 1]
        if len(vals) >= 2:                     # 键名未知时取前两个数组量
            return np.asarray(vals[0], float), np.asarray(vals[1], float)
    if isinstance(obj, (tuple, list)) and len(obj) >= 2:
        try:
            return np.asarray(obj[0], float), np.asarray(obj[1], float)
        except Exception:
            pass
    arr = np.asarray(obj, float)
    if arr.ndim == 2 and arr.shape[0] == 2:
        return arr[0], arr[1]
    return np.linspace(0.0, float(span), int(n_fallback)), np.ones(int(n_fallback), float)


class FermiModel:
    """光子计数半导体探测器响应模型（带实例缓存，避免重复构建 FieldSolver/MC）。"""

    XRAYLIB = None

    def __init__(self):
        self._cache = {}

    # ---------------- 材料 ----------------
    @staticmethod
    def materials():
        try:
            _, MM = _load_fermi()
            cwd = os.getcwd()
            try:
                os.chdir(FERMI_ROOT)
                mm = MM(data_file=os.path.join(FERMI_ROOT, 'data', 'materials.json'))
                names = list(mm.get_all_names())
            finally:
                os.chdir(cwd)
            return names
        except Exception:
            return ['CZT', 'CdTe', 'TlBr', 'Si', 'Ge', 'GaAs', 'HgI2', 'Perovskite', '4H-SiC']

    def _sim(self, material, thickness_mm, voltage, temp_k):
        key = (str(material), round(float(thickness_mm), 4), round(float(voltage), 2),
               round(float(temp_k), 2))
        if key in self._cache:
            return self._cache[key]
        DS, _ = _load_fermi()
        cwd = os.getcwd()
        try:
            os.chdir(FERMI_ROOT)
            sim = DS(material, thickness_mm=float(thickness_mm),
                     voltage=float(voltage), temperature_k=float(temp_k))
        finally:
            os.chdir(cwd)
        self._cache[key] = sim
        return sim

    # ---------------- 单点响应 ----------------
    def noise(self, material, thickness_mm=2.0, voltage=1000.0, temp_k=300.0):
        s = self._sim(material, thickness_mm, voltage, temp_k)
        return s.calculate_noise_properties()

    def band(self, material, thickness_mm=2.0, voltage=1000.0, temp_k=300.0):
        return self._sim(material, thickness_mm, voltage, temp_k).get_band_structure()

    def spectrum(self, material, energy_keV, thickness_mm=2.0, voltage=1000.0,
                 temp_k=300.0, n_bins=200):
        """返回 dict(energy_axis[n_bins], spectrum[n_bins], interaction_efficiency, noise_stats)."""
        s = self._sim(material, thickness_mm, voltage, temp_k)
        out = s.simulate_spectrum(float(energy_keV), n_bins=int(n_bins))
        if isinstance(out, dict):
            return {'energy_axis': np.asarray(out.get('energy_axis'), float),
                    'spectrum': np.asarray(out.get('spectrum'), float),
                    'interaction_efficiency': float(out.get('interaction_efficiency', np.nan)),
                    'noise_stats': out.get('noise_stats')}
        arr = np.asarray(out, float)
        return {'energy_axis': np.linspace(0.0, float(energy_keV), arr.size),
                'spectrum': arr, 'interaction_efficiency': float('nan'), 'noise_stats': None}

    def cce_profile(self, material, thickness_mm=2.0, voltage=1000.0, temp_k=300.0):
        """电荷收集效率沿厚度 z 的分布（Hecht）。返回 (z_mm, cce)。"""
        s = self._sim(material, thickness_mm, voltage, temp_k)
        return _as_pair(s.get_transport_profile(),
                        (('z_mm', 'z_grid', 'z', 'depth_mm', 'x'),
                         ('cce_total', 'cce', 'CCE', 'collection', 'cce_e')),
                        span=float(thickness_mm))

    def field_profile(self, material, thickness_mm=2.0, voltage=1000.0, temp_k=300.0):
        """内部电场沿厚度 z 的分布。返回 (z_mm, E_V_per_mm)。"""
        s = self._sim(material, thickness_mm, voltage, temp_k)
        return _as_pair(s.get_electric_field_profile(),
                        (('z', 'z_grid', 'depth_mm', 'x'), ('E_field', 'E', 'field', 'Efield')),
                        span=float(thickness_mm))

    def attenuation(self, material, e_min=40.0, e_max=190.0, n=120):
        s = self._sim(material, 2.0, 1000.0, 300.0)
        try:
            z, mu = _as_pair(s.get_attenuation_profile((float(e_min), float(e_max))),
                             (('energy', 'energy_axis', 'e'), ('mu', 'attenuation', 'mu_total')),
                             n_fallback=int(n), span=float(e_max))
            if z.size and np.isfinite(mu).any():
                return z, mu
        except Exception:
            pass
        e = np.linspace(float(e_min), float(e_max), int(n))
        return e, np.full_like(e, np.nan)

    # ---------------- 能量箱 ----------------
    K_EDGES = {'CZT': [26.7, 31.8], 'CdTe': [26.7, 31.8], 'TlBr': [84.7],
               'GaAs': [10.4, 11.9], 'Ge': [11.1], 'Si': [1.84], 'HgI2': [83.1]}

    def bins(self, n_bins=8, mode='equal', e_min=20.0, e_max=190.0, material='CZT'):
        """能量箱阈值数组（长度 n_bins+1）。mode: equal(等宽) / kedge(K 边优化)。"""
        n = int(max(2, n_bins))
        if mode == 'kedge':
            edges = [float(e_min)]
            for ke in self.K_EDGES.get(material, []):
                if e_min < ke < e_max:
                    edges += [ke - 2.0, ke + 2.0]
            edges.append(float(e_max))
            edges = sorted(set(edges))
            if len(edges) < n + 1:                        # 不足则均分补齐
                extra = np.linspace(float(e_min), float(e_max), n + 1)
                edges = sorted(set([round(x, 2) for x in list(edges) + list(extra)]))
            edges = np.asarray(edges, float)
            idx = np.linspace(0, edges.size - 1, n + 1).round().astype(int)
            return edges[idx]
        return np.linspace(float(e_min), float(e_max), n + 1)

    def response_matrix(self, material, thresholds, e_src, thickness_mm=2.0,
                        voltage=1000.0, temp_k=300.0, n_bins=200):
        """R[E_src, bin]：入射能量 E_src 的光子被记录进各能量箱的概率（按行归一）。"""
        T = np.asarray(thresholds, float)
        e_src = np.asarray(e_src, float)
        R = np.zeros((e_src.size, T.size - 1), float)
        for i, es in enumerate(e_src):
            sp = self.spectrum(material, es, thickness_mm, voltage, temp_k, n_bins=n_bins)
            ea, y = sp['energy_axis'], np.maximum(sp['spectrum'], 0.0)
            if not np.any(y > 0):
                k = int(np.clip(np.searchsorted(T, es) - 1, 0, T.size - 2))
                R[i, k] = 1.0
                continue
            for k in range(T.size - 1):
                m = (ea >= T[k]) & (ea < T[k + 1])
                R[i, k] = float(y[m].sum())
            tot = R[i].sum()
            if tot > 0:
                R[i] /= tot
        return R

    # ---------------- 入射谱 & 箱权重 ----------------
    @staticmethod
    def source_spectrum(kvp=120.0, e_min=20.0, e_max=190.0, n=180, filtration_mm_al=2.5):
        """kVp 经验谱：Kramers 连续谱 × 铝滤过 × 钨 K 特征线（40–190 keV 段）。"""
        e = np.linspace(e_min, min(e_max, float(kvp)), int(n))
        kv = float(kvp)
        cont = np.maximum(kv - e, 0.0) * e / max(kv, 1e-6)          # Kramers 近似
        mu_al = 0.5 * (e / 20.0) ** (-2.8) + 0.02                    # 铝衰减近似 /mm
        cont *= np.exp(-mu_al * float(filtration_mm_al))
        for ek, wk in ((58.0, 0.55), (67.2, 1.0), (69.1, 1.6)):      # W Kα1/Kα2/Kβ
            cont += wk * np.exp(-0.5 * ((e - ek) / 1.2) ** 2)
        cont[e > kv] = 0.0
        return e, cont / max(cont.sum(), 1e-12)

    def bin_weights(self, material='CZT', kvp=120.0, n_bins=8, mode='equal',
                    thickness_mm=2.0, voltage=1000.0, temp_k=300.0, e_max=190.0):
        """Σ_E φ(E)·R[E,k]：各能量箱的相对计数权重（未归一化，供 FBP 分箱用）。"""
        e_s, phi = self.source_spectrum(kvp=kvp, e_max=e_max)
        T = self.bins(n_bins=n_bins, mode=mode, material=material, e_max=e_max)
        R = self.response_matrix(material, T, e_s, thickness_mm, voltage, temp_k)
        w = (R * phi[:, None]).sum(axis=0)
        return {'thresholds': T, 'e_src': e_s, 'phi': phi, 'R': R, 'weights': w,
                'weights_norm': w / max(w.sum(), 1e-12)}

    def layer_weights(self, material, kvp=120.0, e_max=190.0, d_top=0.5, d_bot=1.5,
                      thickness_mm=2.0, voltage=1000.0, temp_k=300.0):
        """**双层探测器**的层吸收权重（沿射线顺序吸收，非阈值分箱）。

            上层（近病人） w1(E) = 1 − exp(−μ(E)·d1)
            下层（在后方） w2(E) = exp(−μ(E)·d1)·(1 − exp(−μ(E)·d2))

        上层先吸收掉低能部分 → 下层看到的是**硬化后**的剩余束，谱更硬。
        返回 (e_src, phi, w1, w2)。
        """
        e_s, phi = self.source_spectrum(kvp=kvp, e_max=e_max)
        cr = ConeResponse(self)
        mu = np.array([cr.mu_mm(material, e) for e in e_s])
        w1 = 1.0 - np.exp(-mu * float(d_top))
        w2 = np.exp(-mu * float(d_top)) * (1.0 - np.exp(-mu * float(d_bot)))
        return e_s, phi, w1, w2

    # ---------------- 3D 锥束扩展支撑 ----------------
    @staticmethod
    def oblique_factor(gamma_rad, kappa_rad):
        """斜入射路径长度修正：1/(cosγ·cosκ)。γ 为通道扇角，κ 为行锥角。"""
        g = np.asarray(gamma_rad, float)
        k = np.asarray(kappa_rad, float)
        return 1.0 / np.maximum(np.cos(g) * np.cos(k), 1e-6)


class ConeResponse:
    """把 Fermi 的单点响应扩展到**锥束 3D**。

    三个扩展维度
    ------------
    1. **深度维**：XY 二维只给一个沉积深度；3D 需沿射线路径积分。
       吸收位置分布 p(l) ∝ exp(-μ(E)·l)，按 CCE(z) 加权 → 深度加权有效收集效率
           CCE_eff(E) = ∫ p(l)·CCE(z(l)) dl
       物理后果：低能光子在高 μ 下浅层就被吸收（z→0，CCE 低）；高能光子穿透到深层
       （z→d，CCE 高）→ **CCE_eff 随能量升高而升高**（与直觉相反，但正确）。
    2. **斜入射维**：路径长 L = d/(cosγ·cosκ)，z = l·cosγ·cosκ 折算回深度。
    3. **空间维**：在 (γ, κ) 粗网格上算响应（默认 9×5），再双线性插值到 (n_rows, n_ch) 全阵列。

    假设与边界
    ----------
    * 假定 X 射线从 **z=0（阴极侧）入射**——这是 Fermi 的 CCE(z) 坐标约定。
    * 假定能谱响应**形状**与位置无关，只有效率 η(E,γ,κ)=吸收×CCE 随几何变化。
      真实探测器的电荷分享/串扰会让边缘面元的谱形也变化，本模型未含。
    """

    MU_HINT = None

    def __init__(self, model=None):
        self.m = model or FermiModel()
        self._cce = {}

    # ---------- 基础量 ----------
    def cce_z(self, material, thickness_mm, voltage, temp_k):
        key = (str(material), round(float(thickness_mm), 3), round(float(voltage), 1),
               round(float(temp_k), 1))
        if key not in self._cce:
            z, c = self.m.cce_profile(material, thickness_mm, voltage, temp_k)
            o = np.argsort(z)
            self._cce[key] = (np.asarray(z, float)[o], np.asarray(c, float)[o])
        return self._cce[key]

    def _mat_obj(self, name):
        """取 Fermi 的材料对象（MockXrayLib 需要 .atomic_composition，不能传名字）。"""
        cache = getattr(self, '_mo', None)
        if cache is None:
            cache = self._mo = {}
        if name in cache:
            return cache[name]
        _load_fermi()
        _, MM = _load_fermi()
        cwd = os.getcwd()
        try:
            os.chdir(FERMI_ROOT)
            mm = MM(data_file=os.path.join(FERMI_ROOT, 'data', 'materials.json'))
            o = mm.get_material(name)
        finally:
            os.chdir(cwd)
        cache[name] = o
        return o

    def _mu_raw(self, material, E_keV):
        """注意：Fermi 的 `MockXrayLib.get_attenuation` 内部有 `mu_photo[jump_mask] /= 8.0`，
        对**标量**能量会抛 TypeError（只支持数组输入）→ 这里包成单元素数组调用。"""
        _load_fermi()
        from physics import MockXrayLib
        out = MockXrayLib.get_attenuation(self._mat_obj(material),
                                          np.array([float(E_keV)], dtype=float))
        return float(np.asarray(out, float).ravel()[0])

    def mu_mm(self, material, E_keV, thickness_mm=2.0, voltage=1000.0, temp_k=300.0):
        """线性衰减系数 μ(E)，统一到 1/mm。

        Fermi 的 `get_attenuation_profile` 只给区间两端 2 个点，无法用于能谱，
        因此逐点调用 `MockXrayLib.get_attenuation(材料对象, E)`；其输出单位不确定
        （1/m、1/cm、1/mm 皆有可能），用 CZT@100keV ≈ 0.155 /mm 作参考、
        把定标因子**吸附到最近的 10 的整数次幂**，避免过度拟合。
        """
        sc = getattr(self, '_scale', None)
        if sc is None:
            sc = self._scale = {}
        if material not in sc:
            try:
                v = self._mu_raw(material, 100.0)
                s = (0.155 / v) if v > 0 else 1.0
                sc[material] = 10.0 ** round(float(np.log10(max(s, 1e-12))))
            except Exception:
                sc[material] = 1.0
        try:
            return max(1e-6, self._mu_raw(material, float(E_keV)) * sc[material])
        except Exception:
            return 0.1

    def depth_response(self, material, E_keV, thickness_mm=2.0, voltage=1000.0,
                       temp_k=300.0, obl=1.0, n_l=240):
        """深度加权：返回 (CCE_eff, 吸收份额, 平均作用深度 z̄/d)。"""
        d = float(thickness_mm)
        mu = self.mu_mm(material, E_keV, thickness_mm, voltage, temp_k) * float(obl)
        l = np.linspace(0.0, d * float(obl), int(n_l))          # 沿射线路径坐标
        p = mu * np.exp(-mu * l)
        p /= max(p.sum(), 1e-12)
        abs_frac = float(1.0 - np.exp(-mu * d * float(obl)))
        # 入射面约定：Fermi 的 z=0 是**阳极**（由 cce_e/cce_h 端点判定：
        #   z=0 → cce_e=0.996 / cce_h=0.000；z=d → cce_e=0.000 / cce_h=0.632）。
        # 真实探测器 X 射线从**阴极**入射（阳极侧有读出 ASIC 会挡射线）→ 入射面为 z=d。
        z = np.clip(d - l / max(float(obl), 1e-6), 0.0, d)      # l=0 对应 z=d(阴极)
        zz, cc = self.cce_z(material, thickness_mm, voltage, temp_k)
        cce = np.interp(z, zz, cc)
        cce_eff = float((p * cce).sum())
        zbar = float((p * z).sum() / max(d, 1e-9))
        return cce_eff, abs_frac, zbar

    # ---------- 粗网格 → 全阵列 ----------
    @staticmethod
    def expand(grid, geom, n_rows):
        """把 (n_g, n_r) 粗网格（γ 维 × κ 维）双线性插值到 (n_rows, n_ch) 全阵列。"""
        G = np.asarray(grid, float)
        nch = int(geom.n_ch)
        gx = np.linspace(-1.0, 1.0, G.shape[0])
        cxp = np.linspace(-1.0, 1.0, nch)
        tmp = np.array([np.interp(cxp, gx, G[:, j]) for j in range(G.shape[1])])   # (n_r,nch)
        ky = np.linspace(-1.0, 1.0, G.shape[1])
        rxp = np.linspace(-1.0, 1.0, int(n_rows))
        out = np.array([np.interp(rxp, ky, tmp[:, i]) for i in range(nch)]).T      # (n_rows,nch)
        return out

    # ---------- 响应张量 ----------
    def sector_response(self, material, geom, n_rows=128, row_pitch_mm=None, kvp=120.0,
                        n_bins=8, mode='equal', thickness_mm=2.0, voltage=1000.0,
                        temp_k=300.0, n_g=9, n_r=5, e_max=190.0):
        """(γ,κ) 粗网格上的深度加权响应。

        返回 dict：
          gamma / kappa   采样点 (rad)
          cce_eff / abs_eff / zbar    (n_g, n_r) 的几何相关效率
          corr            相对中心面元的效率修正 (n_e, n_g, n_r)
          R_ref           中心面元的能量箱响应矩阵 (n_e, n_bin)
          W_bin           W[bin, n_g, n_r] = Σ_E φ(E)·R_ref[E,bin]·corr
          e_src / phi / thresholds
        """
        g = geom.gammas
        half_g = float(np.max(np.abs(g)))
        beta = float(getattr(geom, 'beta', half_g))
        if row_pitch_mm is None:
            row_pitch_mm = float(getattr(geom, 'row_pitch_det',
                                         geom.p_iso_z * geom.FDD / geom.R))
        cr = (int(n_rows) - 1) / 2.0
        z_iso = (np.arange(int(n_rows)) - cr) * row_pitch_mm * geom.R / geom.FDD
        half_k = float(np.arctan(np.max(np.abs(z_iso)) / geom.FDD)) if z_iso.size else 0.0

        gam_s = np.linspace(-half_g, half_g, int(n_g))
        kap_s = np.linspace(-half_k, half_k, int(n_r)) if half_k > 0 else np.zeros(int(n_r))
        e_s, phi = self.m.source_spectrum(kvp=kvp, e_max=e_max)
        T = self.m.bins(n_bins=n_bins, mode=mode, material=material, e_max=e_max)
        R_ref = self.m.response_matrix(material, T, e_s, thickness_mm, voltage, temp_k)

        cce0, abs0, _ = self.depth_response(material, float(np.average(e_s, weights=phi)),
                                            thickness_mm, voltage, temp_k, 1.0)
        cce_g = np.zeros((n_g, n_r))
        abs_g = np.zeros((n_g, n_r))
        zbar_g = np.zeros((n_g, n_r))
        corr = np.ones((e_s.size, n_g, n_r))
        for i, gv in enumerate(gam_s):
            for j, kv in enumerate(kap_s):
                obl = 1.0 / max(np.cos(gv) * np.cos(kv), 1e-6)
                for k, ek in enumerate(e_s):
                    ce, ab, zb = self.depth_response(material, ek, thickness_mm,
                                                     voltage, temp_k, obl)
                    corr[k, i, j] = (ce * ab) / max(cce0 * abs0, 1e-12)
                ce0, ab0, zb0 = self.depth_response(material, float(np.average(e_s, weights=phi)),
                                                    thickness_mm, voltage, temp_k, obl)
                cce_g[i, j], abs_g[i, j], zbar_g[i, j] = ce0, ab0, zb0
        W = np.einsum('eb,egr->bgr', R_ref * phi[:, None], corr)
        return {'gamma': gam_s, 'kappa': kap_s, 'cce_eff': cce_g, 'abs_eff': abs_g,
                'zbar': zbar_g, 'corr': corr, 'R_ref': R_ref, 'W_bin': W,
                'e_src': e_s, 'phi': phi, 'thresholds': T,
                'half_gamma': half_g, 'half_kappa': half_k}

    def bin_weight_map(self, material, geom, n_rows=128, **kw):
        """W[bin, row, col] 全阵列分箱权重：直接用粗网格结果插值展开。"""
        res = self.sector_response(material, geom, n_rows=n_rows, **kw)
        W = res['W_bin']
        maps = np.stack([self.expand(W[k], geom, n_rows) for k in range(W.shape[0])])
        return maps, res

