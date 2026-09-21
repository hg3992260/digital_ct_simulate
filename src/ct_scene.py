# -*- coding: utf-8 -*-
"""双源 CT 三维场景网格生成（纯 numpy，无 Qt 依赖）。

本模块为 simulate_ct6.py 的 3D 视口提供"可建模"级别的几何体，
取代早期版本中用散点/四棱锥近似球管与锥束的做法：

  curved_panel        曲面探测器面板（带厚度与四周边壁）
  cone_beam_surface   锥束体积表面（焦点为顶点、探测器网格为底的连续曲面）
  beam_rays           代表性 X 射线（焦点 → 探测器模块中心）
  revolve_x           绕局部 X 轴旋转成形（球管外壳 / 准直器 / 罩壳）
  tube_housing        球管总成（真空壳 + 准直器 + 焦点 + 卡箍）
  sphere_wire         球面经纬线（把"半透明球"变成"看得出是球"）
  spherical_shell     球壳带（内外两层 + 端面，用于"受影响 FOV 壳"）
  ring_points         水平圆环折线
  polar_grid          极坐标网格（同心圆 + 放射线）
  arc_with_arrow      带箭头的圆弧（旋转方向指示）
  segment_list        折线集合 → pyqtgraph 线段的 (N,2,3) 数组
"""

import numpy as np


# ----------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------

def _as_faces(tri_list):
    return np.asarray(tri_list, dtype=np.uint32).reshape(-1, 3)


def bowtie_colors(t_profile, mu, alpha=255):
    """按透射率 T(γ)=exp(-μ·t(γ)) 生成逐顶点颜色：中心亮（衰减弱）、边缘暗（衰减强）。"""
    T = np.exp(-mu * np.asarray(t_profile, dtype=float))
    lo, hi = float(T.min()), float(T.max())
    u = (T - lo) / (hi - lo + 1e-12)                 # 边缘 0 → 中心 1
    c_edge = np.array([56.0, 38.0, 12.0])            # 深褐（强衰减）
    c_ctr = np.array([255.0, 238.0, 176.0])          # 亮黄（弱衰减）
    rgb = c_edge[None, :] + u[:, None] * (c_ctr - c_edge)[None, :]
    out = np.concatenate([rgb, np.full((len(u), 1), float(alpha))], axis=1)
    return np.clip(out, 0, 255).astype(np.uint8)


def bowtie_mesh(focus, aim_angle, r_in, t_edge, gamma_bt, z_half, n_g=33,
                mu=0.0196, colorize=False):
    """Bowtie 滤过体实体（真实形状 + 可选按透射率上色）。

    以焦点为中心、半径 r_in 起，沿扇角 γ∈[-γ_bt, γ_bt] 的**弧形领结**：
      厚度剖面   t(γ) = t_edge·(|γ|/γ_bt)²      （中心最薄、两侧最厚）
      外半径     r_out(γ) = r_in + t(γ)
      透射率     T(γ) = exp(-μ·t(γ))            （colorize=True 时映射到顶点色）
    沿 Z 方向按该半径处的射束高度拉伸（剖面与 z 无关）。
    """
    g = np.linspace(-gamma_bt, gamma_bt, n_g)
    t = t_edge * (np.abs(g) / max(gamma_bt, 1e-9)) ** 2
    r_out = r_in + t
    psi = aim_angle + g
    fx, fy, fz = focus

    def ring(r, z):
        return np.column_stack([fx + r * np.cos(psi), fy + r * np.sin(psi),
                                np.full_like(psi, fz + z)])

    blocks = [ring(r_in, -z_half), ring(r_in, +z_half),
              ring(r_out, -z_half), ring(r_out, +z_half)]
    verts = np.vstack(blocks)
    idx = lambda b, i: b * n_g + i
    faces = []
    for i in range(n_g - 1):                       # 内表面 / 外表面 / 上下端面
        faces += [[idx(0, i), idx(1, i), idx(1, i + 1)], [idx(0, i), idx(1, i + 1), idx(0, i + 1)]]
        faces += [[idx(2, i), idx(3, i + 1), idx(3, i)], [idx(2, i), idx(2, i + 1), idx(3, i + 1)]]
        faces += [[idx(0, i), idx(2, i + 1), idx(2, i)], [idx(0, i), idx(0, i + 1), idx(2, i + 1)]]
        faces += [[idx(1, i), idx(3, i), idx(3, i + 1)], [idx(1, i), idx(3, i + 1), idx(1, i + 1)]]
    for i in (0, n_g - 1):                         # 两个端面（扇角边界）
        faces += [[idx(0, i), idx(1, i), idx(3, i)], [idx(0, i), idx(3, i), idx(2, i)]]
    if not colorize:
        return verts, _as_faces(faces)
    cols = np.vstack([bowtie_colors(t, mu)] * 4)   # 四个 blocks 的 γ 顺序相同
    return verts, _as_faces(faces), cols


def helix_path(r, z_per_turn, n_turn=1.6, z_center=0.0, n=360):
    """螺旋进床轨迹（用于把 pitch 可视化）：半径 r、每圈上升 z_per_turn。"""
    tt = np.linspace(0.0, 2.0 * np.pi * n_turn, n)
    zz = (tt / (2.0 * np.pi)) * z_per_turn
    zz = zz - zz.mean() + z_center
    return np.column_stack([r * np.cos(tt), r * np.sin(tt), zz])


def segment_list(polylines):
    """把若干折线（每条 (k,3)）拼成 GLLinePlotItem 需要的 (N,2,3) 线段数组。"""
    segs = []
    for pl in polylines:
        pl = np.asarray(pl, dtype=float)
        if len(pl) < 2:
            continue
        segs.append(np.stack([pl[:-1], pl[1:]], axis=1))
    if not segs:
        return np.zeros((0, 2, 3))
    return np.concatenate(segs, axis=0)


def make_ring(r, z=0.0, a0=0.0, a1=2 * np.pi, n=160, closed=True):
    a = np.linspace(a0, a1, n)
    pts = np.column_stack([r * np.cos(a), r * np.sin(a), np.full_like(a, z)])
    if closed and abs((a1 - a0) - 2 * np.pi) < 1e-6:
        pts = np.vstack([pts, pts[:1]])
    return pts


# ----------------------------------------------------------------------
# 曲面探测器面板（带厚度）
# ----------------------------------------------------------------------

def curved_panel(r_in, r_out, a0, a1, z0, z1, n_ang=64, n_z=10):
    """绕 Z 轴的圆柱壳片段：内弧面 + 外弧面 + 四周边壁（封闭实体）。"""
    th = np.linspace(a0, a1, n_ang)
    zs = np.linspace(z0, z1, n_z)
    TH, ZS = np.meshgrid(th, zs, indexing='ij')
    idx = lambda i, j: i * n_z + j
    n = n_ang * n_z

    inner = np.stack([r_in * np.cos(TH), r_in * np.sin(TH), ZS], axis=-1).reshape(-1, 3)
    outer = np.stack([r_out * np.cos(TH), r_out * np.sin(TH), ZS], axis=-1).reshape(-1, 3)
    verts = np.vstack([inner, outer])

    faces = []
    for i in range(n_ang - 1):
        for j in range(n_z - 1):
            a, b = idx(i, j), idx(i + 1, j)
            c, d = idx(i + 1, j + 1), idx(i, j + 1)
            faces += [[a, b, c], [a, c, d]]                    # 内弧面
            A, B, C, D = n + a, n + b, n + c, n + d
            faces += [[A, C, B], [A, D, C]]                    # 外弧面（反向绕序）
    # 四周边壁
    for i in range(n_ang - 1):
        for j, flip in ((0, False), (n_z - 1, True)):
            a, b = idx(i, j), idx(i + 1, j)
            A, B = n + a, n + b
            faces += [[a, b, B], [a, B, A]] if not flip else [[a, B, b], [a, A, B]]
    for j in range(n_z - 1):
        for i, flip in ((0, False), (n_ang - 1, True)):
            a, b = idx(i, j), idx(i, j + 1)
            A, B = n + a, n + b
            faces += [[a, B, b], [a, A, B]] if not flip else [[a, b, B], [a, B, A]]
    return verts, _as_faces(faces)


def ring_segments(r_in, r_out, z_half, n_seg=24, gap_frac=0.10, n_ang=9, colors=None):
    """整圈**分段探测器环**（绕 Z 轴的实体环，按 n_seg 分段，段间留缝）。

    用于静态多源 CT：一整圈探测器按源数量均分成 n_seg 段，每段对侧于一个射线源。
    colors 给出每段的 RGB(0-255) 时可返回逐顶点色（段与段颜色区分）。
    """
    verts, faces, vcols, off = [], [], [], 0
    seg_span = 2.0 * np.pi / n_seg
    for k in range(n_seg):
        a0 = k * seg_span + seg_span * gap_frac * 0.5
        a1 = (k + 1) * seg_span - seg_span * gap_frac * 0.5
        v, f = curved_panel(r_in, r_out, a0, a1, -z_half, z_half, n_ang=n_ang, n_z=3)
        v = np.asarray(v, float)
        verts.append(v)
        faces.append(np.asarray(f) + off)
        off += len(v)
        if colors is not None:
            c = colors[k % len(colors)]
            vcols.append(np.tile(np.array([c[0], c[1], c[2], 255.0]), (len(v), 1)))
    V, F = np.vstack(verts), np.vstack(faces)
    if colors is None:
        return V, _as_faces(F)
    return V, _as_faces(F), np.clip(np.vstack(vcols), 0, 255).astype(np.uint8)


def source_cluster(centers, radius, colors, nu=10, nv=7):
    """多个射线源：在每个中心放一个实心球（合并为一个网格，逐球着色）。

    用于静态多源 CT 的 n 个源可视化 —— 每个源用不同颜色，
    颜色与对侧探测器分段一致，便于看清"源 ↔ 探测段"配对关系。
    """
    V, F, vcols, off = [], [], [], 0
    u = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
    vv = np.linspace(0.0, np.pi, nv)
    U, VV = np.meshgrid(u, vv, indexing='ij')
    Uf, Vf = U.ravel(), VV.ravel()
    for k, c in enumerate(centers):
        pts = np.column_stack([c[0] + radius * np.sin(Vf) * np.cos(Uf),
                               c[1] + radius * np.sin(Vf) * np.sin(Uf),
                               c[2] + radius * np.cos(Vf)])
        faces = []
        for i in range(nu):
            for j in range(nv - 1):
                a = (i % nu) * nv + j
                b = ((i + 1) % nu) * nv + j
                faces += [[a, b, b + 1], [a, b + 1, a + 1]]
        V.append(pts)
        F.append(np.asarray(faces) + off)
        col = colors[k % len(colors)]
        vcols.append(np.tile(np.array([col[0], col[1], col[2], 255.0]), (len(pts), 1)))
        off += len(pts)
    return (np.vstack(V), _as_faces(np.vstack(F)),
            np.clip(np.vstack(vcols), 0, 255).astype(np.uint8))


def detector_layers(focus, aim, r_centers, half_t, gamma_bt, z_half, n_ang=33, n_z=5,
                    colors=None):
    """双层/多能层探测器：各层沿**射线方向（径向）叠放**，Z 向等高。

    真实双层探测器（如 Philips 双层能谱）是把两层闪烁体沿射线穿行方向前后叠放：
      * 上层（半径小、靠近病人）：优先吸收**低能**光子 → 低能层
      * 下层（半径大、在后方）：接收硬化后的剩余束 → **高能**层
    Z 方向是探测器排/层方向，各层 Z 向尺寸完全相同（= 探测器高度），不做错位。

    r_centers 为各层中心半径（沿焦点→探测器方向），half_t 为单层径向半厚。
    """
    g = np.linspace(-gamma_bt, gamma_bt, n_ang)
    zs = np.linspace(-z_half, z_half, n_z)
    psi = aim + g
    G, Z = np.meshgrid(g, zs, indexing='ij')          # (n_ang, n_z)
    PSI = aim + G
    V, F, vcols, off = [], [], [], 0
    for li, rc in enumerate(r_centers):
        for r_ in (rc - half_t, rc + half_t):         # 内表面 / 外表面
            V.append(np.column_stack([focus[0] + r_ * np.cos(PSI).ravel(),
                                      focus[1] + r_ * np.sin(PSI).ravel(),
                                      (focus[2] + Z).ravel()]))
        n = n_ang * n_z
        idx = lambda b, i, j: off + b * n + i * n_z + j
        for i in range(n_ang - 1):                    # 两个柱面
            for j in range(n_z - 1):
                for b in (0, 1):
                    F += [[idx(b, i, j), idx(b, i + 1, j), idx(b, i + 1, j + 1)],
                          [idx(b, i, j), idx(b, i + 1, j + 1), idx(b, i, j + 1)]]
        for i in range(n_ang - 1):                    # 上下端面
            for b_z in (0, n_z - 1):
                F += [[idx(0, i, b_z), idx(1, i, b_z), idx(1, i + 1, b_z)],
                      [idx(0, i, b_z), idx(1, i + 1, b_z), idx(0, i + 1, b_z)]]
        for j in range(n_z - 1):                      # 扇角两侧端面
            for b_i in (0, n_ang - 1):
                F += [[idx(0, b_i, j), idx(1, b_i, j), idx(1, b_i, j + 1)],
                      [idx(0, b_i, j), idx(1, b_i, j + 1), idx(0, b_i, j + 1)]]
        off += 2 * n
        if colors is not None:
            c = colors[li % len(colors)]
            vcols.append(np.tile(np.array([c[0], c[1], c[2], 255.0]), (2 * n, 1)))
    VV, FF = np.vstack(V), np.vstack(F)
    if colors is None:
        return VV, _as_faces(FF)
    return VV, _as_faces(FF), np.clip(np.vstack(vcols), 0, 255).astype(np.uint8)


def layer_ribbons(focus, aim, r, gamma_bt, z_centers, half_h, n_ang=41, d_r=4.0,
                  colors=None):
    """多层/多能量箱探测器：每层在焦点前方生成一段弧形薄板（内外面 + 上下边缘）。

    用于双层探测器（2 层）与光子计数（N 能量箱）的可视化：
    层与层之间在 Z 方向留缝，因此"第二层"清晰可见；colors 给出每层 RGB 时逐层着色。
    """
    g = np.linspace(-gamma_bt, gamma_bt, n_ang)
    psi = aim + g
    V, F, vcols, off = [], [], [], 0
    for li, zc in enumerate(z_centers):
        for r_ in (r, r + d_r):
            p_lo = np.column_stack([focus[0] + r_ * np.cos(psi), focus[1] + r_ * np.sin(psi),
                                    np.full_like(psi, zc - half_h)])
            p_hi = np.column_stack([focus[0] + r_ * np.cos(psi), focus[1] + r_ * np.sin(psi),
                                    np.full_like(psi, zc + half_h)])
            blk = np.vstack([p_lo, p_hi])
            V.append(blk)
            for i in range(n_ang - 1):
                F += [[off + i, off + i + 1, off + n_ang + i + 1],
                      [off + i, off + n_ang + i + 1, off + n_ang + i]]
            off += 2 * n_ang
            if colors is not None:
                c = colors[li % len(colors)]
                vcols.append(np.tile(np.array([c[0], c[1], c[2], 255.0]), (2 * n_ang, 1)))
    VV, FF = np.vstack(V), np.vstack(F)
    if colors is None:
        return VV, _as_faces(FF)
    return VV, _as_faces(FF), np.clip(np.vstack(vcols), 0, 255).astype(np.uint8)


def detector_grid_lines(focus, det_angle, det_r, beta, kappa, r_off=0.0,
                        n_ang=48, n_z=16, n_ang_lines=16, n_z_lines=5):
    """探测器上的模块分割线与防散射栅（贴在面板内表面）。"""
    r = det_r + r_off
    z_half = det_r * np.tan(kappa)
    lines = []

    # 沿 Z 方向的弧线（列模块边界）
    a = np.linspace(det_angle - beta, det_angle + beta, n_ang)
    for z in np.linspace(-z_half * 0.98, z_half * 0.98, n_z_lines):
        lines.append(np.column_stack([focus[0] + r * np.cos(a),
                                      focus[1] + r * np.sin(a),
                                      np.full_like(a, focus[2] + z)]))
    # 沿弧方向的横向线（行模块边界）
    z = np.linspace(-z_half, z_half, n_z)
    for th in np.linspace(det_angle - beta * 0.99, det_angle + beta * 0.99, n_ang_lines):
        lines.append(np.column_stack([np.full_like(z, focus[0] + r * np.cos(th)),
                                      np.full_like(z, focus[1] + r * np.sin(th)),
                                      focus[2] + z]))
    return segment_list(lines)


def pixel_columns(focus, det_angle, det_r, beta, kappa, n=64, r_off=-4.0, frac=0.85):
    """防散射栅：内表面之前的密集细线（示意 ASG 栅格方向）。"""
    r = det_r + r_off
    z_half = det_r * np.tan(kappa) * frac
    lines = []
    for th in np.linspace(det_angle - beta * 0.98, det_angle + beta * 0.98, n):
        lines.append(np.array([[focus[0] + r * np.cos(th), focus[1] + r * np.sin(th), focus[2] - z_half],
                               [focus[0] + r * np.cos(th), focus[1] + r * np.sin(th), focus[2] + z_half]]))
    return segment_list(lines)


# ----------------------------------------------------------------------
# 锥束
# ----------------------------------------------------------------------

def _det_grid(focus, det_angle, det_r, beta, kappa, n_ang, n_z):
    th = np.linspace(det_angle - beta, det_angle + beta, n_ang)
    zh = det_r * np.tan(kappa)
    zs = np.linspace(-zh, zh, n_z)
    TH, ZS = np.meshgrid(th, zs, indexing='ij')
    pts = np.stack([focus[0] + det_r * np.cos(TH),
                    focus[1] + det_r * np.sin(TH),
                    focus[2] + ZS], axis=-1).reshape(-1, 3)
    return pts, n_ang, n_z


def cone_beam_surface(focus, det_angle, det_r, beta, kappa, n_ang=28, n_z=10):
    """锥束包络：焦点为顶点、探测器网格为底的**闭合实体**。

    侧面只沿底面边界环生成（若对每个网格单元都从顶点发三角形，
    相邻单元的共享边会被正反两次覆盖，法线互相抵消 → 顶点法线出现 NaN）。
    底面按网格三角化以封闭实体，使束流看起来是"体积"而不是空壳。
    """
    grid, na, nz = _det_grid(focus, det_angle, det_r, beta, kappa, n_ang, n_z)
    apex = np.asarray(focus, dtype=float).reshape(1, 3)
    verts = np.vstack([apex, grid])
    idx = lambda i, j: 1 + i * nz + j
    faces = []

    # 侧面：沿底面边界环一圈（顶点 → 相邻两个边界点）
    ring = ([(i, 0) for i in range(na)] +
            [(na - 1, j) for j in range(1, nz)] +
            [(i, nz - 1) for i in range(na - 2, -1, -1)] +
            [(0, j) for j in range(nz - 2, 0, -1)])
    for k in range(len(ring)):
        i0, j0 = ring[k]
        i1, j1 = ring[(k + 1) % len(ring)]
        faces.append([0, idx(i0, j0), idx(i1, j1)])

    # 底面：网格三角化（绕序与侧面相反，封闭成体）
    for i in range(na - 1):
        for j in range(nz - 1):
            a, b = idx(i, j), idx(i + 1, j)
            c, d = idx(i + 1, j + 1), idx(i, j + 1)
            faces += [[a, c, b], [a, d, c]]
    return verts, _as_faces(faces)


def beam_rays(focus, det_angle, det_r, beta, kappa, n_fan=11, n_cone=5, inset=0.92):
    """代表性射线：焦点 → 探测器网格点（含中央轴与外沿）。"""
    grid, na, nz = _det_grid(focus, det_angle, det_r, beta * inset, kappa * inset,
                             max(2, n_fan), max(2, n_cone))
    f = np.asarray(focus, dtype=float)
    lines = []
    for j in range(nz):
        pl = np.vstack([f, np.vstack([grid[i * nz + j] for i in range(na)])])
        lines.append(pl)
    for i in range(na):
        pl = np.vstack([f, np.vstack([grid[i * nz + j] for j in range(nz)])])
        lines.append(pl)
    # 中央轴延长到探测器
    c = det_angle
    lines.append(np.array([f, [f[0] + det_r * np.cos(c), f[1] + det_r * np.sin(c), f[2]]]))
    return segment_list(lines)


def iso_fan_footprint(focus, det_angle, det_r, beta, r_fov, z=0.0, n=44):
    """束流在 ISO 平面内、半径 r_fov 的 FOV 边界上的出口弧。
    对每条射线解 |focus + s·d̂| = r_fov，取穿过 ISO 之后的那一支。"""
    f = np.asarray(focus, dtype=float)
    apex = np.array([f[0], f[1], z])
    out = []
    for t in np.linspace(det_angle - beta, det_angle + beta, n):
        d = np.array([np.cos(t), np.sin(t), 0.0])
        b = float(np.dot(apex, d))
        c = float(np.dot(apex, apex)) - r_fov ** 2
        disc = b * b - c
        if disc <= 0:
            continue
        s = -b + np.sqrt(disc)
        if s <= 0:
            continue
        out.append(apex + s * d)
    if len(out) < 2:
        return np.zeros((0, 3))
    out = np.array(out)
    return np.vstack([out, out[:1]])


# ----------------------------------------------------------------------
# 球管总成
# ----------------------------------------------------------------------

def revolve_x(profile, n=28):
    """把 (x, r) 轮廓绕局部 X 轴旋转成实体。profile 中 r=0 视为极点。"""
    profile = [(float(x), float(r)) for x, r in profile]
    rings = []
    poles = []
    for x, r in profile:
        if r <= 1e-9:
            poles.append(len(rings))
            rings.append(None)
        else:
            a = np.linspace(0, 2 * np.pi, n, endpoint=False)
            rings.append(np.column_stack([np.full(n, x), r * np.cos(a), r * np.sin(a)]))
    verts = []
    ring_index = []
    for rr in rings:
        if rr is None:
            ring_index.append(None)
        else:
            ring_index.append(len(verts))
            verts.extend(rr)
    verts = np.array(verts)
    faces = []
    for k in range(len(rings) - 1):
        A, B = rings[k], rings[k + 1]
        if A is None or B is None:
            continue
        ia, ib = ring_index[k], ring_index[k + 1]
        for i in range(n):
            j = (i + 1) % n
            faces += [[ia + i, ia + j, ib + j], [ia + i, ib + j, ib + i]]
    # 封口（极点 → 相邻环的三角扇）
    for k, rr in enumerate(rings):
        if rr is not None:
            continue
        x = profile[k][0]
        pidx = len(verts)
        verts = np.vstack([verts, [x, 0.0, 0.0]])
        for nb in (k - 1, k + 1):
            if 0 <= nb < len(rings) and rings[nb] is not None:
                ib = ring_index[nb]
                for i in range(n):
                    j = (i + 1) % n
                    if nb < k:
                        faces.append([pidx, ib + j, ib + i])
                    else:
                        faces.append([pidx, ib + i, ib + j])
    return verts, _as_faces(faces)


def tube_housing(focus, aim_angle, body_r=70.0, body_len=210.0, coll_out=48.0,
                 coll_len=56.0, n=28):
    """球管总成：真空壳（后段）+ 准直器（前段）+ 卡箍。
    本地 +X 指向 aim_angle（射线出束方向），焦点在本地原点。"""
    profile = [
        (-body_len, 0.0),
        (-body_len, body_r * 0.82),
        (-body_len * 0.86, body_r),
        (-body_r * 0.9, body_r),
        (-coll_out * 0.75, body_r * 0.97),
        (-coll_out * 0.78, coll_out),
        (coll_len * 0.5, coll_out * 0.55),
        (coll_len, coll_out * 0.42),
        (coll_len, 0.0),
    ]
    verts, faces = revolve_x(profile, n=n)
    c, s = np.cos(aim_angle), np.sin(aim_angle)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    world = np.column_stack([focus[0] + x * c - y * s,
                             focus[1] + x * s + y * c,
                             focus[2] + z])
    return world, faces


def collar_ring(focus, aim_angle, x_local, r=80.0, thick=14.0, n=28):
    """卡箍/法兰环（独立网格，便于单独上色）。"""
    verts, faces = revolve_x([(x_local, r), (x_local + thick, r),
                              (x_local + thick, r * 0.72), (x_local, r * 0.72)], n=n)
    c, s = np.cos(aim_angle), np.sin(aim_angle)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    world = np.column_stack([focus[0] + x * c - y * s,
                             focus[1] + x * s + y * c,
                             focus[2] + z])
    return world, faces


# ----------------------------------------------------------------------
# 球面 / 球壳
# ----------------------------------------------------------------------

def uv_sphere(radius, n_lat=26, n_lon=48):
    """UV 球（纬度略内缩，避免极点退化三角形）。"""
    eps = np.pi / (2.0 * n_lat)
    phi = np.linspace(-np.pi / 2 + eps, np.pi / 2 - eps, n_lat)
    lam = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    PH, LA = np.meshgrid(phi, lam, indexing='ij')
    verts = np.stack([radius * np.cos(PH) * np.cos(LA),
                      radius * np.cos(PH) * np.sin(LA),
                      radius * np.sin(PH)], axis=-1).reshape(-1, 3)
    idx = lambda i, j: i * n_lon + (j % n_lon)
    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            a, b = idx(i, j), idx(i, j + 1)
            c, d = idx(i + 1, j + 1), idx(i + 1, j)
            faces += [[a, b, c], [a, c, d]]
    return verts, _as_faces(faces)


def sphere_wire(radius, center=(0, 0, 0), n_lat=10, n_lon=24):
    """经纬线，让半透明球"看得出是球"。center 是球心（本例恒为 ISO）。"""
    cx, cy, cz = center
    lines = []
    for phi in np.linspace(-np.pi / 2, np.pi / 2, n_lat + 2)[1:-1]:
        a = np.linspace(0, 2 * np.pi, 96)
        lines.append(np.column_stack([cx + radius * np.cos(phi) * np.cos(a),
                                      cy + radius * np.cos(phi) * np.sin(a),
                                      np.full_like(a, cz + radius * np.sin(phi))]))
    for lam in np.linspace(0, np.pi, n_lon, endpoint=False):
        phi = np.linspace(-np.pi / 2, np.pi / 2, 64)
        lines.append(np.column_stack([cx + radius * np.cos(phi) * np.cos(lam),
                                      cy + radius * np.cos(phi) * np.sin(lam),
                                      cz + radius * np.sin(phi)]))
    return segment_list(lines)


def spherical_shell(r_in, r_out, n_lat=26, n_lon=52, keep=None):
    """球壳带：内外两层 + 顶底端面。
    keep(u, v) -> bool 可选，用于只保留某一侧（非对称壳）。
    纬度略向内收缩，避免极点处退化三角形导致法线归一化除零。"""
    eps = np.pi / (2.0 * n_lat)
    phi = np.linspace(-np.pi / 2 + eps, np.pi / 2 - eps, n_lat)
    lam = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)   # 末列不得与首列重合
    PH, LA = np.meshgrid(phi, lam, indexing='ij')

    def surf(r):
        return np.stack([r * np.cos(PH) * np.cos(LA),
                         r * np.cos(PH) * np.sin(LA),
                         r * np.sin(PH)], axis=-1).reshape(-1, 3)

    inner, outer = surf(r_in), surf(r_out)
    ncell = n_lat * n_lon
    verts = np.vstack([inner, outer])
    idx = lambda i, j: i * n_lon + (j % n_lon)
    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            a, b = idx(i, j), idx(i, j + 1)
            c, d = idx(i + 1, j + 1), idx(i + 1, j)
            if keep is not None and not keep(i, j):
                continue
            faces += [[a, b, c], [a, c, d]]
            A, B, C, D = ncell + a, ncell + b, ncell + c, ncell + d
            faces += [[A, C, B], [A, D, C]]
    if keep is None:
        for j in range(n_lon):
            for i, flip in ((0, False), (n_lat - 1, True)):
                a, b = idx(i, j), idx(i, j + 1)
                A, B = ncell + a, ncell + b
                faces += [[a, b, B], [a, B, A]] if not flip else [[a, B, b], [a, A, B]]
    return verts, _as_faces(faces)


# ----------------------------------------------------------------------
# 平面网格 / 旋转指示
# ----------------------------------------------------------------------

def polar_grid(radii, n_spokes=24, z=0.0, r_max=None):
    """极坐标网格：同心圆 + 放射线。"""
    lines = []
    for r in radii:
        lines.append(make_ring(r, z=z, n=140))
    if r_max is None:
        r_max = max(radii)
    for a in np.linspace(0, 2 * np.pi, n_spokes, endpoint=False):
        lines.append(np.array([[0, 0, z], [r_max * np.cos(a), r_max * np.sin(a), z]]))
    return segment_list(lines)


def arc_with_arrow(r, a0, a1, z=0.0, n=64, head=26.0, head_deg=9.0):
    """带箭头的圆弧（旋转方向）。"""
    a = np.linspace(a0, a1, n)
    arc = np.column_stack([r * np.cos(a), r * np.sin(a), np.full_like(a, z)])
    tip = arc[-1]
    tang = arc[-1] - arc[-3]
    tang = tang / (np.linalg.norm(tang) + 1e-9)
    for sgn in (+1, -1):
        rot = np.array([[np.cos(np.radians(sgn * head_deg)), -np.sin(np.radians(sgn * head_deg)), 0],
                        [np.sin(np.radians(sgn * head_deg)), np.cos(np.radians(sgn * head_deg)), 0],
                        [0, 0, 1]])
        back = -tang * head
        arc = np.vstack([arc, tip, tip + rot.dot(back)])
    return segment_list([arc])


def crosshair(size=140.0, z=0.0):
    """ISO 处的十字准线。"""
    segs = [np.array([[-size, 0, z], [size, 0, z]]),
            np.array([[0, -size, z], [0, size, z]])]
    return segment_list(segs)


def gantry_cage(r_in, r_out, z_half, n=48, n_vert=8):
    """机架罩壳骨架：内外两圈 + 若干立柱。"""
    lines = [make_ring(r_in, z=-z_half, n=n), make_ring(r_in, z=z_half, n=n),
             make_ring(r_out, z=-z_half, n=n), make_ring(r_out, z=z_half, n=n)]
    for a in np.linspace(0, 2 * np.pi, n_vert, endpoint=False):
        ca, sa = np.cos(a), np.sin(a)
        lines.append(np.array([[r_in * ca, r_in * sa, -z_half], [r_in * ca, r_in * sa, z_half]]))
        lines.append(np.array([[r_out * ca, r_out * sa, -z_half], [r_out * ca, r_out * sa, z_half]]))
    return segment_list(lines)
