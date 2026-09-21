# -*- coding: utf-8 -*-
"""严格多材料谱分解（光子计数 / 双层 CT）。

与早先"单材料谱定标 p·μ(E)/μ_ref"的区别
----------------------------------------
近似做法把 LEAP 的**单一参考能量线积分**按 μ(E)/μ_ref 缩放 → 单材料严格、混合物有
残余束硬化偏差。严格做法是**逐基材、逐能量正投影**：

  ① 把体模拆成基材（水 / 碘）的等效路径图 L_b(x,y)   [mm of pure material]
  ② 对每个能量 E：μ_map_E = Σ_b μ_b(E)·L_b  → 用 LEAP 正投影得 p_E(β,γ)
  ③ 分箱计数：N_k = Σ_E φ(E)·R[E,k]·exp(−p_E)  →  p_k = −ln(N_k / N_air_k)
  ④ 逐射线解线性方程组  p_k = Σ_b μ_{k,b}·L_b   （最小二乘 + 非负）
       其中 μ_{k,b} = Σ_E φ R[E,k] μ_b(E) / Σ_E φ R[E,k]   （该基材在第 k 箱的有效衰减）
  ⑤ 对 L_b 各做一次 FBP → 水图 / 碘图（基材分解）

μ(E) 来源优先 xraylib（若已安装），否则用内置 NIST 量级表（本环境无 xraylib）。
"""

import numpy as np

# ---------------- μ/ρ 数据 [cm²/g] ----------------
MU_RHO = {
    'water': {40: 0.2682, 60: 0.2059, 80: 0.1837, 100: 0.1707, 120: 0.1610,
              150: 0.1505, 190: 0.1396},
    'iodine': {40: 22.05, 60: 8.03, 80: 4.03, 100: 2.41, 120: 1.61,
               150: 1.05, 190: 0.63},
}
DENSITY = {'water': 1.0, 'iodine': 4.93}          # g/cm³
COMPOSITION = {'water': 'H2O', 'iodine': 'I'}


# 元素质量分数（水面用元素加权构造，避免依赖 xraylib 的化合物表）
ELEMENT_MIX = {
    'water': {'H': (1, 1.008, 2), 'O': (8, 16.00, 1)},       # 符号: (Z, 原子量, 个数)
    'iodine': {'I': (53, 126.90, 1)},
}


def _xraylib_mu_rho(name, E):
    """优先用 xraylib；水用**元素加权**构造（pip wheel 可能缺化合物表）。

    μ/ρ(compound) = Σ_i w_i · (μ/ρ)_i ，w_i 为元素质量分数。
    """
    try:
        import xraylib as xr
    except Exception:
        return None
    mix = ELEMENT_MIX.get(name)
    if not mix:
        return None
    try:
        num = 0.0
        den = 0.0
        for sym, (Z, A, n) in mix.items():
            num += n * A * float(xr.CS_Total(int(Z), float(E)))
            den += n * A
        return num / max(den, 1e-12)
    except Exception:
        return None


def mu_per_mm(material, E_keV):
    """线性衰减系数 μ [1/mm]。"""
    v = _xraylib_mu_rho(material, E_keV)
    tab = MU_RHO[material]
    if v is None:
        es = np.array(sorted(tab.keys()), float)
        vs = np.array([tab[e] for e in es], float)
        v = float(np.interp(float(E_keV), es, vs))          # 表内线性插值 [cm²/g]
    return float(v) * DENSITY[material] / 10.0               # → 1/mm


def mu_per_mm_xraylib_used():
    try:
        import xraylib  # noqa: F401
        return True
    except Exception:
        return False


# ---------------- 体模：水 + 碘 ----------------
def two_material_phantom(N, sfov, iodine_mg_ml=10.0, insert_r_mm=30.0,
                         insert_offset_mm=70.0, body_r_mm=150.0):
    """返回 (L_water, L_iodine)，单位 mm 纯物质等效路径。

    水盘（半径 body_r）+ 碘插入（半径 insert_r，偏置 insert_offset）。
    碘浓度 c [mg/mL] → 碘密度 = c/1000 g/cm³ → 等效纯碘路径 = (c/1000)·厚度。
    """
    yy, xx = np.mgrid[0:N, 0:N]
    c = (N - 1) / 2.0
    v = sfov / N
    x = (xx - c) * v
    y = (yy - c) * v
    r = np.hypot(x, y)
    Lw = np.zeros((N, N), np.float32)
    Li = np.zeros((N, N), np.float32)
    Lw[r <= body_r_mm] = 1.0                                  # 每像素 1 mm 等效水
    m = np.hypot(x - insert_offset_mm, y) <= insert_r_mm
    Lw[m] = 1.0
    Li[m] = float(iodine_mg_ml) / 1000.0                      # mg/mL → g/cm³
    return Lw, Li, float(iodine_mg_ml)


# ---------------- 管线 ----------------
def bin_effective_mu(thresholds, e_src, phi, R):
    """各基材在各能量箱的有效衰减 μ_{k,b} [1/mm] 与箱空气权重。"""
    out = {}
    for b in ('water', 'iodine'):
        mu = np.array([mu_per_mm(b, e) for e in e_src])
        num = (R * (phi[:, None] * mu[:, None])).sum(axis=0)
        den = (R * phi[:, None]).sum(axis=0)
        out[b] = num / np.maximum(den, 1e-12)
    out['air'] = (R * phi[:, None]).sum(axis=0)
    return out


def per_energy_sinograms(engine, geom, Lw, Li, angles, energies):
    """逐能量正投影：返回 p_E[k_E] (n_ch, n_views) 与 μ_map 列表。

    μ_map_E = μ_water(E)·Lw + μ_iodine(E)·Li   （L 为 mm 等效路径）
    """
    Ps = []
    for e in energies:
        mp = (mu_per_mm('water', e) * Lw + mu_per_mm('iodine', e) * Li)
        mp = np.ascontiguousarray(mp, np.float32)
        Ps.append(np.asarray(engine.project_fan(mp, geom, angles), np.float32))
    return Ps


def bin_sinograms(Ps, energies, e_src, phi, R):
    """分箱：N_k = Σ_E φ R exp(−p_E) → p_k = −ln(N_k / N_air)。返回 (nb, n_ch, n_views)。"""
    nb = R.shape[1]
    es = np.asarray(energies, float)
    idx = [int(np.argmin(np.abs(e_src - e))) for e in es]     # 每个投影能量对应的谱采样
    n_ch, n_v = Ps[0].shape
    Pk = np.zeros((nb, n_ch, n_v), np.float32)
    for k in range(nb):
        acc = np.zeros((n_ch, n_v), np.float64)
        air = 0.0
        for e, p in zip(es, Ps):
            i = int(np.argmin(np.abs(e_src - e)))
            b = float(phi[i] * R[i, k])
            if b <= 1e-12:
                continue
            acc += b * np.exp(-p)
            air += b
        Pk[k] = -np.log(np.maximum(acc, 1e-9) / max(air, 1e-12))
    return Pk


def decompose(Pk, mu_kb, nonneg=True):
    """逐射线解 p_k = Σ_b μ_{k,b}·L_b（最小二乘；可选非负投影）。返回 (Lw_sino, Li_sino)。"""
    A = np.stack([mu_kb['water'], mu_kb['iodine']], axis=1)   # (nb, 2)
    nb, n_ch, n_v = Pk.shape
    Y = Pk.reshape(nb, -1)                                    # (nb, n_ch*n_v)
    # 正规方程 (AᵀA) L = Aᵀ y
    G = A.T @ A
    Gi = np.linalg.pinv(G)
    L = Gi @ (A.T @ Y)                                        # (2, n_ch*n_v)
    if nonneg:
        L = np.maximum(L, 0.0)
    return (L[0].reshape(n_ch, n_v).astype(np.float32),
            L[1].reshape(n_ch, n_v).astype(np.float32))



# ===================== 非线性（严格自洽）反问题 =====================
def energy_grid(n=16, e_min=40.0, e_max=190.0):
    return np.linspace(float(e_min), float(e_max), int(n))


def _cell_edges(e_grid):
    d0 = e_grid[1] - e_grid[0]
    d1 = e_grid[-1] - e_grid[-2]
    return np.concatenate([[e_grid[0] - d0 / 2.0],
                           (e_grid[:-1] + e_grid[1:]) / 2.0,
                           [e_grid[-1] + d1 / 2.0]])


def spectrum_on_grid(e_src, phi, e_grid):
    """把**细谱按网格单元积分**（非点采样）。

    W 特征线 σ≈1.2 keV 远窄于网格步长，点采样会丢线、其权重被摊到邻近格点 →
    能量求积误差的主因。单元积分保证 Σ_k φ_k = Σ_E φ(E) 守恒。
    """
    e = np.asarray(e_src, float)
    ph = np.asarray(phi, float)
    o = np.argsort(e)
    e, ph = e[o], ph[o]
    ed = _cell_edges(np.asarray(e_grid, float))
    out = np.zeros(e_grid.size)
    for k in range(e_grid.size):
        m = (e >= ed[k]) & (e < ed[k + 1])
        out[k] = ph[m].sum()
    s = out.sum()
    return out / s if s > 0 else ph / max(ph.sum(), 1e-12)


def response_on_grid(R, e_src, e_grid):
    """把箱响应按网格单元**取平均**（保留 K 边阶跃的权重分布）。"""
    e = np.asarray(e_src, float)
    Rm = np.asarray(R, float)
    ed = _cell_edges(np.asarray(e_grid, float))
    out = np.empty((Rm.shape[1], e_grid.size))
    for k in range(Rm.shape[1]):
        for j in range(e_grid.size):
            m = (e >= ed[j]) & (e < ed[j + 1])
            out[k, j] = Rm[m, k].mean() if m.sum() else np.interp(e_grid[j], e, Rm[:, k])
    return out


def bin_from_energy_maps(PE, phi16, R16):
    """(ne,n_ch,n_v) 各能量线积分 → (nb,n_ch,n_v) 分箱线积分。

    与反问题使用**完全相同**的公式 → 无模型失配：
        p_k = −ln( Σ_E φ R exp(−p_E) / Σ_E φ R )
    """
    PE = np.asarray(PE, np.float64)
    w = R16 * phi16[None, :]                       # (nb, ne)
    air = np.maximum(w.sum(axis=1), 1e-30)         # (nb,)
    att = np.exp(-np.clip(PE, -50.0, 50.0))        # (ne,n_ch,n_v)
    num = np.einsum('ke,ecv->kcv', w, att)
    return -np.log(np.maximum(num, 1e-30) / air[:, None, None])


def decompose_newton(Pk, phi16, R16, e16, n_iter=6, nonneg=True, mu_cache=None):
    """逐射线**非线性牛顿迭代**解双基材路径 (L_water, L_iodine)。

    正问题 p_k(L) 与 bin_from_energy_maps 完全一致；雅可比解析给出。
    初值用线性最小二乘，随后 Newton 修正（对高浓度/低能箱的非线性区尤其必要）。
    """
    Pk = np.asarray(Pk, np.float64)
    nb, n_ch, n_v = Pk.shape
    nr = n_ch * n_v
    Y = Pk.reshape(nb, nr)
    e16 = np.asarray(e16, float)
    mu_w = np.array([mu_per_mm('water', e) for e in e16])
    mu_i = np.array([mu_per_mm('iodine', e) for e in e16])
    w = R16 * phi16[None, :]                       # (nb, ne)
    air = np.maximum(w.sum(axis=1), 1e-30)
    # 初值：线性最小二乘（用一阶有效 μ）
    A0 = np.stack([(w * mu_w[None, :]).sum(1) / air,
                   (w * mu_i[None, :]).sum(1) / air], axis=1)
    L = np.maximum(np.linalg.pinv(A0.T @ A0) @ (A0.T @ Y), 1e-6)   # (2,nr)
    resid = float('nan')
    for _ in range(int(n_iter)):
        att = np.exp(-(mu_w[:, None] * L[0][None, :] + mu_i[:, None] * L[1][None, :]))
        num = w @ att                              # (nb,nr)
        num_s = np.maximum(num, 1e-30)
        p_mod = -np.log(num_s / air[:, None])
        # 解析雅可比（全部 (nb, nr)，沿轴 0 求和是各箱贡献）
        J0 = (w * mu_w[None, :]) @ att / num_s      # ∂p_k/∂L_w
        J1 = (w * mu_i[None, :]) @ att / num_s      # ∂p_k/∂L_i
        rr = Y - p_mod                              # (nb, nr)
        resid = float(np.abs(rr).mean())
        a11 = (J0 * J0).sum(0)
        a12 = (J0 * J1).sum(0)
        a22 = (J1 * J1).sum(0)
        b1 = (J0 * rr).sum(0)
        b2 = (J1 * rr).sum(0)
        det = np.where(np.abs(a11 * a22 - a12 * a12) < 1e-20, 1e-20,
                       a11 * a22 - a12 * a12)
        d0 = (a22 * b1 - a12 * b2) / det
        d1 = (a11 * b2 - a12 * b1) / det
        L[0] += d0
        L[1] += d1
        if nonneg:
            L = np.maximum(L, 0.0)
    return (L[0].reshape(n_ch, n_v).astype(np.float32),
            L[1].reshape(n_ch, n_v).astype(np.float32),
            resid)
