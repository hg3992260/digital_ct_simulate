# -*- coding: utf-8 -*-
"""sinogram ↔ 探测器单元 ↔ 空间几何 的解析索引与按 z 轴重排。

要"从 sinogram 还原 FBP 图像"，必须先知道 sinogram 里每一条线到底对应哪个探测器单元、
以及该单元在空间中的位置；螺旋/多层采集下还要按各样本的**绝对 z** 重新组织数据。

坐标与约定（系统 A，ISO 为原点）
--------------------------------
* 环架角 β：源位于  S = sod·(cosβ, sinβ, 0)
* 弯曲探测器（等角）：通道 c 的射线与中心线夹角
      γ_c = (c - center_col)·dγ ,   dγ = p_det / sdd
  其在 ISO 平面内的横向坐标（平行束等价坐标）
      t_c = sod·sin(γ_c)
* 探测器行 r 相对中心行的 z 偏移（探测器平面上）
      z_row(r) = (r - center_row)·row_pitch
  对应的 ISO 平面 z 偏移（除以放大比 m = sdd/sod）
      z_iso(r) = z_row(r)·sod/sdd
* 螺旋：视角 j 的表位置  z_table(j) = (j / n_views)·pitch·Z_cov
* ⇒ 样本 (j, r, c) 的**绝对 z**：
      z = z0 + z_table(j) + z_iso(r)
  射线方向（含锥角 κ_r）：
      d ∝ -( cos(β+γ), sin(β+γ), z_row(r)/sdd )
  探测器单元中心（圆柱面）：
      P = S + sdd·(cos(β+γ), sin(β+γ), 0) + z_row(r)·ẑ

`z_reorder()` 按绝对 z 把全部样本分箱到目标层，并给出每层的角度覆盖
——后者正是螺距上限判据（覆盖 ≥ 180°+扇角）的数据层实现。
"""

import numpy as np


# ----------------------------------------------------------------------
# 1. 探测器单元 ↔ 几何
# ----------------------------------------------------------------------

def channel_map(geom):
    """通道 → 扇角 γ(rad/deg)、ISO 横向坐标 t(mm)、探测器弧坐标 s(mm)。"""
    g = geom.gammas                                  # (n_ch,) rad
    d = {                          # 与 ct_geometry 的 beta_A=asin(SFOV/2R) 严格一致
        'channel': np.arange(geom.n_ch),
        'gamma_rad': g,
        'gamma_deg': np.rad2deg(g),
        't_iso_mm': geom.R * np.sin(g),              # 平行束等价坐标
        'arc_mm': geom.FDD * g,                      # 沿探测器弧的位置（相对中心）
        'u_tan': np.tan(g),                          # 平板探测器等价坐标（=t/z 方向斜率）
    }
    return d


def row_map(geom, n_rows, row_pitch_mm=None, center_row=None):
    """探测器行 → z 偏移（探测器平面 / ISO 平面）、锥角 κ。"""
    if row_pitch_mm is None:
        row_pitch_mm = geom.p_iso_z * geom.FDD / geom.R      # ISO 行距 → 探测器行距
    if center_row is None:
        center_row = (n_rows - 1) / 2.0
    r = np.arange(n_rows, dtype=float)
    z_row = (r - center_row) * row_pitch_mm                  # 探测器平面 z
    return {
        'row': r,
        'z_row_mm': z_row,
        'z_iso_mm': z_row * geom.R / geom.FDD,               # 折算到 ISO 平面
        'kappa_deg': np.rad2deg(np.arctan(z_row / geom.FDD)),
        'row_pitch_mm': row_pitch_mm,
    }


def view_map(geom, n_rot=1, pitch=1.0):
    """视角 → 环架角 β 与表位置 z_table。"""
    n = geom.n_views
    idx = np.arange(n_rot * n)
    beta = (idx % n) * (360.0 / n)                            # 环架角（度）
    dz_rot = float(pitch) * geom.z_cov                        # 每圈进床
    z0 = (n_rot - 1) / 2.0 * dz_rot                           # 让目标层落在采集范围中央
    z_tab = (idx / float(n)) * dz_rot - z0
    return {'idx': idx, 'rot': idx // n, 'view_in_rot': idx % n,
            'beta_deg': beta, 'z_table_mm': z_tab, 'dz_per_rot_mm': dz_rot, 'z0_mm': z0}


def ray_table(geom, n_rows=1, row_pitch_mm=None, n_rot=1, pitch=1.0, channels=None):
    """解析索引表：每个 (row, channel, view) 样本 → 探测器单元与空间射线几何。

    返回 dict of ndarray；为控制体积，view 维默认只取一圈（n_rot=1 时即整圈）。
    形状约定：(n_rows, n_ch, n_views) 的物理量见各键名后缀。
    """
    cm = channel_map(geom)
    rm = row_map(geom, n_rows, row_pitch_mm)
    vm = view_map(geom, n_rot, pitch)
    if channels is not None:
        sel = np.asarray(channels)
        gam = cm['gamma_rad'][sel]
    else:
        sel = np.arange(geom.n_ch)
        gam = cm['gamma_rad']

    beta = np.deg2rad(vm['beta_deg'])                         # (n_all,)
    g = gam[None, :, None]                                    # (1, n_ch, 1)
    b = beta[None, None, :]                                   # (1, 1, n_all)
    z_row = rm['z_row_mm'][:, None, None]                     # (n_rows,1,1)

    psi = b + g                                               # 探测器单元方位角
    cell_x = geom.R * np.cos(b) + geom.FDD * np.cos(psi)
    cell_y = geom.R * np.sin(b) + geom.FDD * np.sin(psi)
    cell_z = z_row + vm['z_table_mm'][None, None, :] * 0.0    # 单元在机架平面内的 z（随行）
    # 绝对 z：表位置 + 行偏移折算
    z_abs = (vm['z_table_mm'][None, None, :]
             + rm['z_iso_mm'][:, None, None] * np.ones((1, len(sel), 1)))
    return {
        'channel': np.broadcast_to(sel[None, :, None], (n_rows, len(sel), len(beta))),
        'row': np.broadcast_to(rm['row'][:, None, None], (n_rows, len(sel), len(beta))),
        'view': np.broadcast_to(vm['idx'][None, None, :], (n_rows, len(sel), len(beta))),
        'gamma_rad': np.broadcast_to(gam[None, :, None], (n_rows, len(sel), len(beta))),
        'beta_deg': np.broadcast_to(vm['beta_deg'][None, None, :], (n_rows, len(sel), len(beta))),
        't_iso_mm': np.broadcast_to(cm['t_iso_mm'][None, :, None], (n_rows, len(sel), len(beta))),
        'z_row_mm': np.broadcast_to(z_row, (n_rows, len(sel), len(beta))),
        'z_abs_mm': z_abs,
        'cell_x': cell_x, 'cell_y': cell_y, 'cell_z': cell_z,
        'src_x': (geom.R * np.cos(b)).repeat(n_rows, 0).repeat(len(sel), 1),
        'src_y': (geom.R * np.sin(b)).repeat(n_rows, 0).repeat(len(sel), 1),
        'row_pitch_mm': rm['row_pitch_mm'], 'n_rows': n_rows,
        'channel_map': cm, 'row_map': rm, 'view_map': vm,
    }


# ----------------------------------------------------------------------
# 2. 按 z 轴重排（切片分箱 + 角度覆盖）
# ----------------------------------------------------------------------

def z_reorder(geom, target_z, tol_mm=None, n_rows=1, row_pitch_mm=None,
              n_rot=1, pitch=1.0, dual_alpha_deg=0.0):
    """把全部样本按绝对 z 分箱到目标层 target_z（mm 数组）。

    返回列表：每层 {z, n_samples, views, rows_used, angle_span_deg, samples_per_view, ok}
    ok 判据：该层可用的环架角覆盖 ≥ 180°+扇角（与几何螺距上限同一判据）。

    dual_alpha_deg > 0 时按双源处理：第二套系统的样本位于 β+α（同一 z、同一通道阵列），
    其角度样本并入同一层 —— 这正是双源能把可用螺距上限从 360/(180+fan) 提到
    360/(180+fan−α) 的数据层原因。
    """
    vm = view_map(geom, n_rot, pitch)
    rm = row_map(geom, n_rows, row_pitch_mm)
    need = 180.0 + geom.fan_deg
    z_all = (vm['z_table_mm'][None, :] + rm['z_iso_mm'][:, None])      # (n_rows, n_all)
    beta = vm['beta_deg'][None, :] * np.ones((n_rows, 1))
    out = []
    if tol_mm is None:
        tol_mm = max(geom.p_iso_z, 1e-3) * 0.5
    for z0 in np.atleast_1d(target_z):
        m = np.abs(z_all - z0) <= tol_mm
        n_s = int(m.sum())
        if n_s == 0:
            out.append({'z': float(z0), 'n_samples': 0, 'angle_span_deg': 0.0,
                        'samples_per_view': 0, 'ok': False, 'views': 0, 'rows_used': 0,
                        'need_deg': need})
            continue
        base = beta[m]
        ang = np.unique(np.round(base, 6))
        if dual_alpha_deg:
            ang = np.unique(np.round(np.concatenate([base, np.mod(base + dual_alpha_deg, 360.0)]), 6))
        if ang.size > 1:
            gaps = np.diff(ang)
            wrap_gap = 360.0 - (ang[-1] - ang[0])
            biggest = max(float(gaps.max()) if gaps.size else 0.0, float(wrap_gap))
            span = float(360.0 - biggest)          # 圆上真实连续覆盖（最大间隙法）
        else:
            span = 0.0
        per_view = n_s / max(1, ang.size)
        out.append({'z': float(z0), 'n_samples': n_s, 'angle_span_deg': span,
                    'samples_per_view': per_view, 'need_deg': need,
                    'ok': bool(span >= need),
                    'views': int(ang.size), 'rows_used': int(np.unique(np.where(m)[0]).size)})
    return out


def reorder_to_slice(sino3d, geom, z0, tol_mm=None, n_rows=1, row_pitch_mm=None,
                     n_rot=1, pitch=1.0):
    """按 z 组装目标层数据：返回 (values, meta) —— 每个可用样本的值与其视角/通道/行索引。

    sino3d 形状 (n_rows, n_ch, n_rot*n_views)；单排时为 (1, n_ch, n_views)。
    """
    s = np.asarray(sino3d)
    if s.ndim == 2:
        s = s[None, :, :]
    nr, nch, nall = s.shape
    vm = view_map(geom, n_rot, pitch)
    rm = row_map(geom, nr, row_pitch_mm)
    z_all = vm['z_table_mm'][None, :] + rm['z_iso_mm'][:, None]
    if tol_mm is None:
        tol_mm = max(geom.p_iso_z, 1e-3) * 0.5
    rr, vv = np.where(np.abs(z_all - z0) <= tol_mm)
    vals = s[rr, :, vv].T                                  # (n_ch, n_sel)
    return vals, {'row': rr, 'view': vv, 'n_sel': int(rr.size),
                  'beta_deg': vm['beta_deg'][vv], 'z': float(z0), 'tol_mm': tol_mm}


# ----------------------------------------------------------------------
# 3. 文本摘要（供界面读数）
# ----------------------------------------------------------------------

def describe(geom, n_rows=None, row_pitch_mm=None, n_rot=1, pitch=1.0, max_lines=14):
    if n_rows is None:
        n_rows = geom.n_rows
    cm = channel_map(geom)
    rm = row_map(geom, n_rows, row_pitch_mm)
    vm = view_map(geom, n_rot, pitch)
    d = {}
    d['探测器单元阵列'] = f'{geom.n_ch} 通道 × {n_rows} 排  (共 {geom.n_ch * n_rows} 个单元)'
    d['通道扇角 γ 范围'] = f'±{np.rad2deg(abs(cm["gamma_rad"]).max()):.4f}°  (中心 {cm["gamma_rad"][geom.n_ch // 2]:.4f}°)'
    d['通道角间距 dγ'] = f'{np.rad2deg(geom.d_gamma):.5f}° = {geom.p_det:.3f} mm / FDD'
    d['ISO 横向坐标 t'] = f'±{abs(cm["t_iso_mm"]).max():.1f} mm  (= R·sinγ, 步距 {geom.p_iso:.3f} mm)'
    d['行 z 偏移(ISO)'] = (f'±{abs(rm["z_iso_mm"]).max():.1f} mm  行距 {rm["row_pitch_mm"]:.3f} mm'
                       f'  (Z 覆盖 {geom.z_cov:.0f} mm)')
    d['锥角 κ 范围'] = f'±{np.abs(rm["kappa_deg"]).max():.3f}°'
    d['视角/表位置'] = (f'{len(vm["idx"])} views, 每圈进床 {vm["dz_per_rot_mm"]:.1f} mm, '
                    f'目标层 z0={vm["z0_mm"]:.1f} mm')
    return d
