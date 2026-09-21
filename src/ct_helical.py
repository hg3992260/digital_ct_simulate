# -*- coding: utf-8 -*-
"""螺旋扫描仿真 + z 方向插值重建（单排探测器 / 扇束，LEAP 弯曲探测器正投影）。

物理模型
--------
床以 v = pitch·Z_cov / T_rot 匀速进床。以圈为单位，全局视角 i 的表位置
    z_t(i) = (i / n_views) · pitch · Z_cov
单排探测器只有一行，因此同一 (β, γ) 射线在第 k 圈测得的是**不同 z** 的样本：
    z_a(k) = (j/n + k)·Δz,   Δz = pitch·Z_cov

目标层 z0 的重建数据由 z 方向线性插值得到：
  * **360°LI**：只用同一条射线的相邻两圈样本插值（z 采样间隔 = Δz）
  * **180°LI**：把同射线样本与其**共轭射线** (β+π+2γ, −γ) 的样本合并后插值。
    两族样本在 z 方向交错（相位差 1/2 + γ/π 圈），等效 z 采样间隔减半
    → z 分辨率更好、螺旋伪影更少（这正是 180LI 相对 360LI 的意义）

超过几何螺距上限时角度覆盖 < 180°+扇角 → 插值权重无法覆盖整圈 → 可见螺旋伪影。
"""

import numpy as np


# ----------------------------------------------------------------------
# 体模（沿 z 有结构变化，否则看不出螺旋伪影）
# ----------------------------------------------------------------------

def build_helical_phantom(nz, n, sfov, dz_slice, z_center=0.0):
    """返回 (vol[nz, n, n], z_slices[nz])：柱体 + 球 + 圆锥 + 台阶。"""
    vol = np.zeros((nz, n, n), dtype=np.float32)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    c = (n - 1) / 2.0
    v = sfov / n
    x = (xx - c) * v
    y = (yy - c) * v
    r = np.hypot(x, y)
    z_slices = z_center + (np.arange(nz) - (nz - 1) / 2.0) * dz_slice
    for k, z in enumerate(z_slices):
        sl = np.zeros((n, n), dtype=np.float32)
        body = r <= 180.0
        sl[body] = 0.02                                  # 均匀柱体
        sph = (x ** 2 + y ** 2 + (z - 0.0) ** 2) <= 45.0 ** 2
        sl[sph] = 0.05                                   # 球：z 方向有界定 → 观察 z 模糊
        cone_h = 90.0 * max(0.0, 1.0 - abs(z) / 70.0)
        cone = r <= cone_h
        sl[cone] = np.maximum(sl[cone], 0.035)           # 圆锥：沿 z 渐缩
        step = (x > 100.0) & (x < 150.0) & (np.abs(y) < 60.0) & (z > -25.0)
        sl[step] = 0.06                                  # 台阶：z 方向突变
        small = (x + 80.0) ** 2 + y ** 2 + (z - 25.0) ** 2 <= 18.0 ** 2
        sl[small] = 0.08                                 # 小高对比球 → 螺旋伪影敏感
        vol[k] = sl * body
    return vol, z_slices


# ----------------------------------------------------------------------
# 采集：逐切片正投影 + 床运动合成
# ----------------------------------------------------------------------

def slice_sinograms(engine, vol, geom, angles, progress=None):
    """对每个 z 切片做扇束正投影，返回 S[nz, n_ch, n_views]。"""
    out = np.empty((vol.shape[0], geom.n_ch, len(angles)), dtype=np.float32)
    for k in range(vol.shape[0]):
        out[k] = engine.project_fan(vol[k], geom, angles)
        if progress:
            progress(k + 1, vol.shape[0])
    return out


def acquire_helical(S, z_slices, geom, pitch, n_rot):
    """按床运动合成螺旋采集数据，返回 (G[n_ch, n_rot*n_views], z_table[n_rot*n_views])。"""
    nz, n_ch, n = S.shape
    dz_rot = float(pitch) * geom.z_cov                     # 每圈进床
    n_all = n_rot * n
    i = np.arange(n_all)
    # 目标层 z0 置于采集范围中央
    z0 = (n_rot - 1) / 2.0 * dz_rot
    z_table = (i / float(n)) * dz_rot - z0                 # 相对目标层的表位置
    z_slices = np.asarray(z_slices, dtype=float)
    G = np.empty((n_ch, n_all), dtype=np.float32)
    for t in range(n_all):
        f = np.interp(z_table[t], z_slices, np.arange(nz))
        a = int(np.clip(np.floor(f), 0, nz - 2))
        w = float(np.clip(f - a, 0.0, 1.0))
        vv = t % n
        G[:, t] = (1.0 - w) * S[a][:, vv] + w * S[a + 1][:, vv]
    return G, z_table, z0, dz_rot


# ----------------------------------------------------------------------
# 多层锥束螺旋：多排探测器 + 按 z 重排
# ----------------------------------------------------------------------

def _row_and_table_z(geom, n_rows, row_pitch_mm, pitch, n_rot):
    """行 z 偏移(ISO) 与视角表位置 —— 与 ct_index 的解析关系一致。"""
    if row_pitch_mm is None:
        row_pitch_mm = float(getattr(geom, 'row_pitch_det',
                                     geom.p_iso_z * geom.FDD / geom.R))
    cr = (n_rows - 1) / 2.0
    z_iso = (np.arange(n_rows) - cr) * row_pitch_mm * geom.R / geom.FDD
    n = geom.n_views
    n_all = n_rot * n
    dz_rot = float(pitch) * geom.z_cov
    z0 = (n_rot - 1) / 2.0 * dz_rot
    z_tab = (np.arange(n_all) / float(n)) * dz_rot - z0
    return z_iso, z_tab, dz_rot, z0, row_pitch_mm


def acquire_helical_multislice(S, z_slices, geom, pitch, n_rot, n_rows, row_pitch_mm=None):
    """多层锥束螺旋采集：S (nz,n_ch,n_views) 为逐片扇束投影。

    行 r 的样本取自 z = z_table(i) + z_iso(r) 处的切片（同一排的各通道共享该 z，
    即 CT 的"某一排对应某一层"关系；锥角 κ 只在寸法上体现，本模型按行独立近似）。
    返回 G3 (n_rows, n_ch, n_rot*n_views)。
    """
    nz, n_ch, n = S.shape
    z_iso, z_tab, dz_rot, z0, rp = _row_and_table_z(geom, n_rows, row_pitch_mm, pitch, n_rot)
    n_all = n_rot * n
    zs = np.asarray(z_slices, float)
    G3 = np.empty((n_rows, n_ch, n_all), dtype=np.float32)
    kk = np.arange(n_rot)[None, :]
    for v in range(n):
        vv = np.broadcast_to(v + kk * n, (n_rows, n_rot))
        vv = vv % n                                                 # S 只有一圈的角度列
        Z = z_tab[v::n][None, :] + z_iso[:, None]                   # (n_rows, n_rot)
        f = np.interp(Z.ravel(), zs, np.arange(nz)).reshape(Z.shape)
        a = np.clip(np.floor(f).astype(int), 0, nz - 2)
        w = np.clip(f - a, 0.0, 1.0)
        lo = S[a, :, vv]                                            # (n_rows,n_rot,n_ch)
        hi = S[a + 1, :, vv]
        G3[:, :, v::n] = np.transpose((1.0 - w)[..., None] * lo + w[..., None] * hi, (0, 2, 1))
    return G3, {'z_iso': z_iso, 'z_table': z_tab, 'dz_rot': dz_rot, 'z0': z0,
                'row_pitch_mm': rp}


def multislice_slice_sinogram(G3, geom, pitch, n_rot, z0, row_pitch_mm, zf=1.0):
    """按 z 距离对多排样本做三角核加权 → 目标层 (n_ch, n_views) 扇形束正弦图。

    这就是 reorder_to_slice 的连续形式：把属于同一 z 层、来自不同排/不同圈的
    全部探测器单元样本，按视角分组加权平均，得到该层的完整角度采样。
    """
    nr, n_ch, n_all = G3.shape
    n = geom.n_views
    z_iso, z_tab, dz_rot, z_off, _ = _row_and_table_z(geom, nr, row_pitch_mm, pitch, n_rot)
    W = max(float(zf) * max(float(row_pitch_mm) * geom.R / geom.FDD, dz_rot / n), 1e-6)
    P = np.zeros((n_ch, n), dtype=np.float32)
    cnt = 0
    for v in range(n):
        Z = z_tab[v::n][None, :] + z_iso[:, None]
        w = np.clip(1.0 - np.abs(Z - z0) / W, 0.0, 1.0)             # (nr, n_rot)
        s = w.sum()
        if s <= 0:
            continue
        blk = G3[:, :, v::n]                                        # (nr,n_ch,n_rot)
        P[:, v] = (w[:, None, :] * blk).sum(axis=(0, 2)) / s
        cnt += 1
    return P, {'z_bracket': float(2.0 * W), 'views_used': cnt,
               'samples_per_view': float(nr * max(1, n_rot) * W / max(dz_rot, 1e-9))}




# ----------------------------------------------------------------------
# z 方向插值（单排重建目标层）
# ----------------------------------------------------------------------

def _catmull3(t):
    """Catmull-Rom 三次样条四点权重 (w_{-1}, w_0, w_1, w_2)，t 为 p0→p1 之间的分数位置。
    相比线性插值把截断误差从 O(h²) 降到 O(h⁴)，用于共轭射线的视角方向插值。"""
    t2 = t * t
    t3 = t2 * t
    return (-0.5 * t3 + t2 - 0.5 * t,
            1.5 * t3 - 2.5 * t2 + 1.0,
            -1.5 * t3 + 2.0 * t2 + 0.5 * t,
            0.5 * t3 - 0.5 * t2)


def interpolate_slice(G, geom, pitch, n_rot, dz_rot, z0, mode='180LI', zfilter_w=1.0):
    """目标层 z0 的扇形束正弦图（n_ch, n_views）及诊断信息。

    mode: '360LI' 同射线两圈线性；'180LI' 共轭射线合并 + **三次样条**共轭插值；
          '180LI-lin' 共轭插值退化为线性（用于对比）。
    """
    n_ch, n_all = G.shape
    n = geom.n_views
    j = np.arange(n)
    dz = float(dz_rot)
    info = {'mode': mode, 'dz_rot': dz, 'z0': z0}

    # ---- 同射线族：全局视角 j + k·n ----
    t_a = z0 / dz - j / n                                   # (n,)
    k_a = np.clip(np.floor(t_a).astype(int), 0, max(0, n_rot - 2))
    w_a = np.clip(t_a - k_a, 0.0, 1.0)
    ia0 = j + k_a * n
    ia1 = np.minimum(ia0 + n, n_all - 1)
    A0 = G[:, ia0]                                          # (n_ch, n)
    A1 = G[:, ia1]
    z_a_lo = (j / n + k_a) * dz
    z_a_hi = z_a_lo + dz
    if mode == '360LI':
        P = A0 * (1.0 - w_a)[None, :] + A1 * w_a[None, :]
        info['z_bracket'] = float(dz)
        return P.astype(np.float32), info

    # ---- 180LI：共轭射线 (β+π+2γ, −γ)，视角索引需同时插值 ----
    gamma = geom.gammas                                     # (n_ch,)
    frac = 0.5 + gamma / np.pi                              # 共轭视角偏移（以圈为单位）= (π+2γ)/2π
    t_b = z0 / dz - (j[None, :] / n + frac[:, None])        # (n_ch, n)
    k_b = np.clip(np.floor(t_b).astype(int), 0, max(0, n_rot - 2))
    w_b = np.clip(t_b - k_b, 0.0, 1.0)
    idx_c = j[None, :] + frac[:, None] * n + k_b * n        # 连续全局索引 (n_ch, n)
    i0 = np.floor(idx_c).astype(int)
    wi = np.clip(idx_c - i0, 0.0, 1.0)
    cm = (n_ch - 1) - np.arange(n_ch)                       # 共轭通道镜像

    if mode == '180LI-lin':
        B_lo = (1.0 - wi) * G[cm[:, None], np.clip(i0, 0, n_all - 1)] \
            + wi * G[cm[:, None], np.clip(i0 + 1, 0, n_all - 1)]
        B_hi = (1.0 - wi) * G[cm[:, None], np.clip(i0 + n, 0, n_all - 1)] \
            + wi * G[cm[:, None], np.clip(i0 + n + 1, 0, n_all - 1)]
        info['conj_interp'] = '线性'
    else:
        # 三次样条（Catmull-Rom）：四点加权，显著降低共轭射线视角插值偏差
        wm, w0, w1, w2 = _catmull3(wi)

        def gath(off):
            return G[cm[:, None], np.clip(i0 + off, 0, n_all - 1)]

        B_lo = wm * gath(-1) + w0 * gath(0) + w1 * gath(1) + w2 * gath(2)
        B_hi = wm * gath(n - 1) + w0 * gath(n) + w1 * gath(n + 1) + w2 * gath(n + 2)
        info['conj_interp'] = '三次样条 (Catmull-Rom)'
    info['_cubic'] = (mode != '180LI-lin')

    # ---- 180MF：两族在 ±W 内的全部样本按三角核加权融合 ----
    # 目的：180LI 的高频不一致项会被 ramp 滤波放大（z 均匀体模上即 ~3%），
    # z 方向滤波用更宽的平滑核把它压下去，代价是 z 分辨率略降。
    if mode == '180MF':
        tgt = z0 / dz
        Wn = max(float(zfilter_w), 1e-6)                 # 半宽（单位：Δz）
        num = np.zeros((n_ch, n), dtype=np.float32)
        den = np.zeros((n_ch, n), dtype=np.float32)
        K = int(np.ceil(Wn)) + 1
        for k in range(-K, K + 1):
            # 族 A：同射线，圈号 k_a + k
            zz = (j / n + k_a + k)
            okA = ((k_a + k) >= 0) & ((k_a + k) <= n_rot - 2)
            wt = np.clip(1.0 - np.abs(zz - tgt) / Wn, 0.0, 1.0) * okA
            num += wt[None, :] * G[:, np.clip(j + (k_a + k) * n, 0, n_all - 1)]
            den += wt[None, :]
            # 族 B：共轭射线，圈号 k_b + k（视角方向已插值）
            zz_b = j[None, :] / n + frac[:, None] + k_b + k
            okB = ((k_b + k) >= 0) & ((k_b + k) <= n_rot - 2)
            wtb = np.clip(1.0 - np.abs(zz_b - tgt) / Wn, 0.0, 1.0) * okB
            if mode == '180MF':
                i0k = i0 + k * n
                if mode != '180LI-lin':
                    def gk(off):
                        return G[cm[:, None], np.clip(i0k + off, 0, n_all - 1)]
                    Bk = wm * gk(-1) + w0 * gk(0) + w1 * gk(1) + w2 * gk(2)
                else:
                    Bk = (1.0 - wi) * G[cm[:, None], np.clip(i0k, 0, n_all - 1)] \
                        + wi * G[cm[:, None], np.clip(i0k + 1, 0, n_all - 1)]
            num += wtb * Bk
            den += wtb
        P = num / np.maximum(den, 1e-9)
        info['conj_interp'] = f'180MF z 滤波 (三角核半宽 {Wn:.1f}×Δz)'
        info['z_bracket'] = float(2.0 * Wn * dz)
        return P.astype(np.float32), info
    z_b_lo = (j[None, :] / n + frac[:, None] + k_b) * dz
    z_b_hi = z_b_lo + dz

    # 两族合并：取最紧的夹逼区间
    z_lo = np.maximum(z_a_lo[None, :], z_b_lo)
    z_hi = np.minimum(z_a_hi[None, :], z_b_hi)
    p_lo = np.where(z_a_lo[None, :] >= z_b_lo, A0, B_lo)
    p_hi = np.where(z_a_hi[None, :] <= z_b_hi, A1, B_hi)
    w = np.clip((z0 - z_lo) / np.maximum(z_hi - z_lo, 1e-9), 0.0, 1.0)
    P = (1.0 - w) * p_lo + w * p_hi
    info['z_bracket'] = float(np.mean(z_hi - z_lo))
    return P.astype(np.float32), info
