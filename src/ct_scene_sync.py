# -*- coding: utf-8 -*-
"""同步辐射 CT 三维场景基本体（纯 numpy，无 Qt 依赖）。

与 ct_scene.py 并列的**独立模块**，不修改既有文件。

物理布局（真实同步辐射束线）：
    储存环/波荡器 ──30~50 m 束线──> 单色器 → 狭缝 → 样品(转台) → 平板探测器
                                                    ↑ ODD 0.05~1 m

**断轴（broken axis）**：源距 30 m、样品 0.3 m，两者在同一个视图里差 100 倍。
本模块把束线沿 X 轴做**分段映射**：
    样品侧 0~NEAR      1:1 显示（看细节）
    中间   NEAR~cut    压缩成固定显示长度 + 画 // 断裂符号（表示"这里省略了很多"）
    源侧   cut~SOD     1:1 显示（看源结构）
真实距离由尺寸线标注，不靠目测。

坐标约定：束流沿 +X，等中心（样品）在原点，探测器在 +X。
"""

import numpy as np

# ---------------------------------------------------------------- 断轴映射
NEAR_MM = 420.0        # 样品侧 1:1 段（mm，真实）
FAR_MM = 420.0         # 源侧 1:1 段（mm，真实）
BREAK_DISP = 300.0     # 断轴段在显示中的固定长度（mm）


def axis_break(d_real, sod_mm, near=NEAR_MM, far=FAR_MM, brk=BREAK_DISP):
    """真实距离 d(0=样品, sod=源) → 显示距离。分段线性 + 中间压缩。

    d<=near             -> d                （样品侧 1:1）
    near<d<sod-far      -> 压缩到 brk 长度  （断轴段）
    d>=sod-far          -> 源侧 1:1
    """
    d = float(d_real)
    cut = max(float(sod_mm) - far, near + 1e-6)
    if d <= near:
        return d
    if d >= cut:
        return near + brk + (d - cut)
    return near + brk * (d - near) / (cut - near)


def axis_break_inv(s_disp, sod_mm, near=NEAR_MM, far=FAR_MM, brk=BREAK_DISP):
    """显示距离 → 真实距离（标注用）。"""
    s = float(s_disp)
    cut = max(float(sod_mm) - far, near + 1e-6)
    if s <= near:
        return s
    if s >= near + brk:
        return cut + (s - near - brk)
    return near + (s - near) / brk * (cut - near)


def break_center(sod_mm, near=NEAR_MM, brk=BREAK_DISP):
    """断轴符号应放置的显示位置。"""
    return -(near + brk * 0.5)


# ---------------------------------------------------------------- 基础几何
def _basis(axis):
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    a = a / (n if n > 1e-12 else 1.0)
    tmp = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, tmp)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(a, u)
    return a, u, v


def _faces_from_grid(n_ring, n_seg, base0, base1, closed=True):
    """两圈顶点之间的侧面三角形。"""
    f = []
    m = n_ring if closed else n_ring - 1
    for i in range(m):
        j = (i + 1) % n_ring
        a, b = base0 + i, base0 + j
        c, d = base1 + j, base1 + i
        f += [[a, b, c], [a, c, d]]
    return f


def cylinder(p0, p1, r, n=24, caps=True):
    """圆柱（可含端盖）。返回 (verts(N,3) float32, faces(M,3) uint32)。"""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    ax, u, v = _basis(p1 - p0)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.cos(th)[:, None] * u[None, :] + np.sin(th)[:, None] * v[None, :]
    A = p0[None, :] + r * ring
    B = p1[None, :] + r * ring
    V = np.vstack([A, B])
    F = _faces_from_grid(n, 1, 0, n)
    if caps:
        c0 = len(V); V = np.vstack([V, p0[None, :]])
        c1 = len(V); V = np.vstack([V, p1[None, :]])
        for i in range(n):
            j = (i + 1) % n
            F += [[c0, j, i]]
            F += [[c1, n + i, n + j]]
    return V.astype(np.float32), np.asarray(F, dtype=np.uint32).reshape(-1, 3)


def box(center, half, axis=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    """长方体：half = (沿轴, 横向, 竖向) 半长。返回 (v,f)。"""
    c = np.asarray(center, float)
    a, u, v = _basis(axis)
    if up is not None:
        w = np.asarray(up, float)
        w = w - a * np.dot(w, a)
        nw = np.linalg.norm(w)
        if nw > 1e-9:
            v = w / nw
            u = np.cross(v, a)
    h = np.asarray(half, float)
    pts = []
    for sa in (-1, 1):
        for su in (-1, 1):
            for sv in (-1, 1):
                pts.append(c + sa * h[0] * a + su * h[1] * u + sv * h[2] * v)
    V = np.asarray(pts, dtype=np.float32)
    # 顶点顺序: (sa,su,sv) -> index = 4*ia + 2*iu + iv
    idx = lambda ia, iu, iv: 4 * ia + 2 * iu + iv
    F = []
    for ia in (0, 1):
        for iu in (0, 1):
            F.append([idx(ia, iu, 0), idx(ia, iu, 1), idx(1 - ia, iu, 1)])
            F.append([idx(ia, iu, 0), idx(1 - ia, iu, 1), idx(1 - ia, iu, 0)])
        for iv in (0, 1):
            F.append([idx(ia, 0, iv), idx(ia, 1, iv), idx(1 - ia, 1, iv)])
            F.append([idx(ia, 0, iv), idx(1 - ia, 1, iv), idx(1 - ia, 0, iv)])
    for iu in (0, 1):
        for iv in (0, 1):
            F.append([idx(0, iu, iv), idx(1, iu, iv), idx(1, 1 - iu, iv)])
            F.append([idx(0, iu, iv), idx(1, 1 - iu, iv), idx(0, 1 - iu, iv)])
    return V, np.asarray(F, dtype=np.uint32).reshape(-1, 3)


# ---------------------------------------------------------------- 同步辐射构件
def beamline_tube(x0, x1, r, n=20, rib_every=140.0):
    """束线真空管：圆柱 + 周期性加强环（让"长"看得出来）。返回 (v,f,vr,fr)。"""
    V, F = cylinder((x0, 0, 0), (x1, 0, 0), r, n=n, caps=False)
    Vs, Fs = [V], [F]
    off = len(V)
    k = 0
    x = x0 + rib_every * 0.5
    while x < x1:
        rv, rf = cylinder((x - r * 0.10, 0, 0), (x + r * 0.10, 0, 0), r * 1.16, n=n, caps=False)
        Vs.append(rv); Fs.append(rf + off)
        off += len(rv); k += 1
        x += rib_every
    return (np.vstack(Vs).astype(np.float32),
            np.vstack(Fs).astype(np.uint32),
            np.vstack(Vs).astype(np.float32), np.vstack(Fs).astype(np.uint32))


def crystal_plate(center, w=90.0, h=60.0, t=8.0, tilt_deg=12.0):
    """双晶单色器晶板（可倾斜）。返回 (v,f)。"""
    c = np.asarray(center, float)
    th = np.deg2rad(tilt_deg)
    a = np.array([np.cos(th), 0.0, np.sin(th)])
    return box(c, (t * 0.5, w * 0.5, h * 0.5), axis=a, up=(0.0, 0.0, 1.0))


def slit_jaws(center, gap=26.0, w=70.0, h=70.0, t=14.0):
    """四刀口狭缝：上下左右四块，中间留 gap。返回 (v,f)。"""
    c = np.asarray(center, float)
    parts = []
    for sign in (1, -1):
        parts.append(box(c + np.array([0.0, 0.0, sign * (gap * 0.5 + h * 0.5)]),
                         (t * 0.5, w * 0.5, h * 0.5)))
        parts.append(box(c + np.array([0.0, sign * (gap * 0.5 + w * 0.5), 0.0]),
                         (t * 0.5, w * 0.5, h * 0.5)))
    V = np.vstack([p[0] for p in parts])
    F = []
    off = 0
    for p in parts:
        F.append(p[1] + off); off += len(p[0])
    return V.astype(np.float32), np.vstack(F).astype(np.uint32)


def flat_panel(center, w=200.0, h=160.0, t=10.0, n_grid=9):
    """平板探测器（非弧形）：面板 + 网格线。返回 (v,f,grid_polylines)。"""
    c = np.asarray(center, float)
    V, F = box(c, (t * 0.5, w * 0.5, h * 0.5), axis=(0, 1, 0), up=(0, 0, 1))
    lines = []
    x = c[0] + t * 0.5 + 0.5
    for i in range(n_grid):
        u = (i / (n_grid - 1) - 0.5) * w
        lines.append(np.array([[x, c[1] + u, c[2] - h / 2], [x, c[1] + u, c[2] + h / 2]]))
        vv = (i / (n_grid - 1) - 0.5) * h
        lines.append(np.array([[x, c[1] - w / 2, c[2] + vv], [x, c[1] + w / 2, c[2] + vv]]))
    return V.astype(np.float32), F, lines


def stage_disk(center, r=70.0, h=16.0, r_post=26.0, post_h=110.0):
    """样品转台：转盘 + 立柱 + 旋转箭头。返回 (v,f,arrow_polylines,disk_ring)。"""
    c = np.asarray(center, float)
    Vd, Fd = cylinder(c + np.array([0, 0, -h * 0.5]), c + np.array([0, 0, h * 0.5]), r, n=32)
    Vp, Fp = cylinder(c + np.array([0, 0, -h * 0.5 - post_h]),
                      c + np.array([0, 0, -h * 0.5]), r_post, n=20)
    V = np.vstack([Vd, Vp]); F = np.vstack([Fd, Fp + len(Vd)])
    # 旋转箭头（水平圆弧 + 箭头）
    th = np.linspace(np.deg2rad(-40), np.deg2rad(220), 60)
    rr = r * 1.35
    arc = np.stack([c[0] + rr * np.cos(th), c[1] + rr * np.sin(th),
                    np.full_like(th, c[2] + h * 0.5 + 2.0)], axis=1)
    tip = arc[-1]
    t0 = arc[-2]
    d = tip - t0
    dn = d / (np.linalg.norm(d) + 1e-12)
    perp = np.array([-dn[1], dn[0], 0.0]) * 16.0
    head = [np.vstack([tip, tip - dn * 26.0 + perp]),
            np.vstack([tip, tip - dn * 26.0 - perp])]
    ring = np.stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th),
                     np.full_like(th, c[2] + h * 0.5 + 0.4)], axis=1)
    return (V.astype(np.float32), F.astype(np.uint32), [arc] + head, ring)


def parallel_rays(x0, x1, half_span=110.0, n=9, z=0.0):
    """近平行束：沿 +X 的平行线（**不是锥形扇**）。返回折线列表。"""
    lines = []
    for yy in np.linspace(-half_span, half_span, n):
        lines.append(np.array([[x0, yy, z], [x1, yy, z]]))
    return lines


def break_marks(x, size=70.0, gap=26.0, z=0.0):
    """断轴符号 // —— 两道平行斜线，表示"此处省略了很长距离"。"""
    out = []
    for sx in (x - gap * 0.5, x + gap * 0.5):
        out.append(np.array([[sx - size * 0.22, -size, z], [sx + size * 0.22, size, z]]))
        out.append(np.array([[sx - size * 0.22, -size, z + 2.0], [sx + size * 0.22, size, z + 2.0]]))
    return out


def storage_ring_symbol(center, r=150.0, z=0.0, n=72):
    """储存环示意：两个同心圆 + 电子束箭头。返回折线列表。"""
    c = np.asarray(center, float)
    th = np.linspace(0, 2 * np.pi, n)
    out = []
    for rr in (r, r * 0.82):
        out.append(np.stack([c[0] + rr * np.cos(th), c[1] + rr * np.sin(th),
                             np.full_like(th, z)], axis=1))
    th2 = np.linspace(np.deg2rad(20), np.deg2rad(120), 40)
    out.append(np.stack([c[0] + r * 0.91 * np.cos(th2), c[1] + r * 0.91 * np.sin(th2),
                         np.full_like(th2, z + 1.0)], axis=1))
    return out


def undulator_symbol(x0, x1, half=42.0, n_period=9, z=0.0):
    """波荡器示意：周期性磁极（上下交替的短块）+ 中心电子束线。返回折线列表。"""
    out = []
    xs = np.linspace(x0, x1, n_period * 2 + 1)
    for i in range(len(xs) - 1):
        sgn = 1 if i % 2 == 0 else -1
        y = sgn * half
        out.append(np.array([[xs[i], y, z], [xs[i + 1], y, z]]))
        out.append(np.array([[xs[i], y, z], [xs[i], 0.0, z]]))
    out.append(np.array([[x0, 0.0, z], [x1, 0.0, z]]))
    return out


def dimension_line(p0, p1, text_z=0.0, ticks=26.0):
    """尺寸线（两端带刻度），用于标注真实距离。返回折线列表。"""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    out = [np.vstack([p0, p1])]
    for p in (p0, p1):
        out.append(np.array([[p[0], p[1] - ticks, p[2]], [p[0], p[1] + ticks, p[2]]]))
    return out
