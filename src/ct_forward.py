# -*- coding: utf-8 -*-
"""M1 · 体模与投影几何 + 正向投影器（纯 numpy + scipy.ndimage，无 Qt）。

依据《PCCT 模拟同步辐射 CT · 算法交接手册》§8：

    M1 体模与几何（无依赖）
      * 输入：SOD / ODD / 探测器像素 p / 阵列尺寸 / 样品 μ(x,y,E)
      * 输出：投影几何、M、样品处采样网格
      * 验收：对已知解析体模（圆柱 / Shepp-Logan）的正向投影与解析解误差 < 1%

为什么先做 M1
-------------
手册 §1.2 列出的第一个缺口就是"**无正向投影器** → 无法生成过采样数据"。
没有它，M4 的 K 帧过采样、M5 的重建、M7 的尺度验证全都悬空。

本模块给出两条**互相独立**的路径，互为判据：

  1. `analytic_sinogram()` —— **解析**线积分（椭圆的弦长有闭式解），当"真值"。
  2. `numeric_sinogram()`  —— **数值**射线积分（对离散体模做双线性采样求和）。

两者之差即 M1 的验收误差。

单位与坐标
----------
* 体模以"体模单位"定义（Shepp-Logan 外椭圆 a=0.69 ≈ 头半轴），经 `scale_mm` 换算。
* 探测器像素 p 用 µm；投影几何在**样品面**展开，p_eff = p / M。
* 平行束约定：探测角 θ 的射线方向 d=(cosθ,sinθ)，探测坐标轴 n=(−sinθ,cosθ)，
  点 (x,y) 投到 s = −x·sinθ + y·cosθ。
"""

import numpy as np

try:
    from scipy.ndimage import map_coordinates
except Exception:                                    # pragma: no cover
    map_coordinates = None


# ---------------------------------------------------------------------------
# 体模：椭圆叠加 (A, x0, y0, a, b, phi_deg) —— 体模单位
# ---------------------------------------------------------------------------
SHEPP_LOGAN = [
    (1.0, 0.0000, 0.0000, 0.6900, 0.9200, 0.0),
    (-0.8, 0.0000, -0.0184, 0.6624, 0.8740, 0.0),
    (-0.2, 0.2200, 0.0000, 0.1100, 0.3100, -18.0),
    (-0.2, -0.2200, 0.0000, 0.1600, 0.4100, 18.0),
    (0.1, 0.0000, 0.3500, 0.2100, 0.2500, 0.0),
    (0.1, 0.0000, 0.1000, 0.0460, 0.0460, 0.0),
    (0.1, 0.0000, -0.1000, 0.0460, 0.0460, 0.0),
    (0.1, -0.0800, -0.6050, 0.0460, 0.0230, 0.0),
    (0.1, 0.0000, -0.6060, 0.0230, 0.0230, 0.0),
    (0.1, 0.0600, -0.6050, 0.0230, 0.0460, 0.0),
]

CYLINDER = [(1.0, 0.0, 0.0, 0.6900, 0.6900, 0.0)]


def bars_phantom(n_bars=5, period=0.10, duty=0.5):
    """分辨率卡：等宽条纹（M7 的"最小可分辨特征"用）。"""
    out, w = [], period * duty / 2.0
    for i in range(n_bars):
        x0 = -(n_bars - 1) * period / 2.0 + i * period
        out.append((1.0, x0, 0.0, w, 0.34, 0.0))
    return out


PHANTOMS = {'shepp_logan': SHEPP_LOGAN, 'cylinder': CYLINDER, 'bars': bars_phantom()}


def phantom(name='shepp_logan', extra=None):
    el = [tuple(e) for e in PHANTOMS.get(str(name), SHEPP_LOGAN)]
    return el + [tuple(e) for e in (extra or [])]


def mu_extent(ellipses):
    """体模单位下、以原点为中心的最大外接半径（用于铺够栅格）。

    注意要取 max(a,b)：Shepp-Logan 的 b=0.92 > a=0.69，
    只按 a 计算会把体模裁在网格外。
    """
    r = 0.0
    for A, x0, y0, a, b, phi in ellipses:
        r = max(r, np.hypot(x0, y0) + max(abs(a), abs(b)))
    return float(r)


# ---------------------------------------------------------------------------
# 解析投影
# ---------------------------------------------------------------------------
def ellipse_chord(s, a, b, theta_deg):
    """椭圆被直线截得的弦长（闭式解）。

    直线在**椭圆主轴坐标系**里的方向角为 θ，距椭圆中心的有符号距离为 s。
    把 (x,y)=s·n + t·d 代入 (x/a)²+(y/b)²=1 得关于 t 的二次方程：

        t²·P + B·t + C = 0
        P = cos²θ/a² + sin²θ/b²
        B = 2·s·sinθcosθ·(1/b² − 1/a²)      ← 交叉项只在 a=b 或 θ=0 时才消失
        C = s²·Q − 1,  Q = sin²θ/a² + cos²θ/b²

    弦长 = √(B² − 4PC) / P（判别式 > 0 时）。

    注意：常见的"2√((1−s²Q)/P)"写法**只对未旋转或圆截面成立**，
    旋转椭圆下交叉项不抵消，会算错。
    """
    s = np.asarray(s, dtype=float)
    th = np.deg2rad(float(theta_deg))
    ct, st = np.cos(th), np.sin(th)
    P = ct ** 2 / a ** 2 + st ** 2 / b ** 2
    Q = st ** 2 / a ** 2 + ct ** 2 / b ** 2
    B = 2.0 * s * st * ct * (1.0 / b ** 2 - 1.0 / a ** 2)
    C = s ** 2 * Q - 1.0
    disc = B ** 2 - 4.0 * P * C
    out = np.zeros_like(s)
    m = disc > 0
    out[m] = np.sqrt(disc[m]) / P
    return out


def analytic_sinogram(ellipses, n_cols, p_eff_mm, angles_deg, scale_mm, mu_scale=1.0):
    """解析正弦图，形状 (n_angles, n_cols)。线积分可加 → 逐椭圆叠加。"""
    s = (np.arange(n_cols) - (n_cols - 1) / 2.0) * p_eff_mm
    out = np.zeros((len(angles_deg), n_cols), dtype=np.float64)
    for i, th_deg in enumerate(angles_deg):
        th = np.deg2rad(float(th_deg))
        nx, ny = -np.sin(th), np.cos(th)
        acc = np.zeros(n_cols, dtype=np.float64)
        for A, x0, y0, a, b, phi in ellipses:
            s0 = (x0 * scale_mm) * nx + (y0 * scale_mm) * ny
            # 射线方向在椭圆主轴系里的角度 = 探测角 − 椭圆倾角
            acc += A * ellipse_chord(s - s0, a * scale_mm, b * scale_mm, th_deg - phi)
        out[i] = acc * mu_scale
    return out


# ---------------------------------------------------------------------------
# 离散体模与数值投影
# ---------------------------------------------------------------------------
def rasterize_phantom(ellipses, n, fov_mm, scale_mm, mu_scale=1.0, aa=1):
    """把椭圆体模铺到 n×n 网格上（返回 μ 图，单位 1/mm）。

    aa > 1 时对每个像素做 aa×aa 子采样取覆盖率（抗锯齿）：
    硬阈值栅格化在**锐利边缘**（如分辨率卡的条纹边）上会带来与网格无关的
    系统偏差，表现为误差收敛很慢；子采样把边缘覆盖率算准，误差随 n 正常下降。
    """
    v = fov_mm / n
    c = (n - 1) / 2.0
    k = max(1, int(aa))
    off = (np.arange(k) - (k - 1) / 2.0) / k          # 子采样偏移（单位：像素）
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    img = np.zeros((n, n), dtype=np.float64)
    for A, x0, y0, a, b, phi in ellipses:
        th = np.deg2rad(float(phi))
        ct, st = np.cos(th), np.sin(th)
        cov = np.zeros((n, n), dtype=np.float64)
        for oy in off:
            for ox in off:
                X = (xx + ox - c) * v
                Y = (yy + oy - c) * v
                dx, dy = X - x0 * scale_mm, Y - y0 * scale_mm
                u = dx * ct + dy * st
                w = -dx * st + dy * ct
                cov += ((u / (a * scale_mm)) ** 2 + (w / (b * scale_mm)) ** 2 <= 1.0)
        img += A * (cov / float(k * k))
    return img * mu_scale


def numeric_sinogram(img, fov_mm, n_cols, p_eff_mm, angles_deg, n_t=None):
    """数值正向投影（射线驱动 + 双线性采样），形状 (n_angles, n_cols)。

    与解析解的区别正是 M1 要量化的对象：
      * 体模离散化（网格 n）
      * 射线积分步长（n_t）
      * 探测器采样（p_eff）
    """
    if map_coordinates is None:                      # pragma: no cover
        raise RuntimeError('scipy.ndimage 不可用，无法做数值正向投影')
    n = img.shape[0]
    v = fov_mm / n
    c = (n - 1) / 2.0
    if n_t is None:
        n_t = max(512, 2 * n)
    # t 覆盖整个视场对角线长度
    half = 0.5 * fov_mm * 1.5
    t = np.linspace(-half, half, int(n_t))
    dt = t[1] - t[0]

    s = (np.arange(n_cols) - (n_cols - 1) / 2.0) * p_eff_mm
    out = np.zeros((len(angles_deg), n_cols), dtype=np.float64)
    for i, th_deg in enumerate(angles_deg):
        th = np.deg2rad(float(th_deg))
        dx, dy = np.cos(th), np.sin(th)
        nx, ny = -np.sin(th), np.cos(th)
        # 所有 (s, t) 组合一次性采样
        X = s[:, None] * nx + t[None, :] * dx
        Y = s[:, None] * ny + t[None, :] * dy
        rows = Y / v + c
        cols = X / v + c
        samp = map_coordinates(img, [rows.ravel(), cols.ravel()], order=1,
                               mode='constant', cval=0.0).reshape(X.shape)
        out[i] = samp.sum(axis=1) * dt
    return out


# ---------------------------------------------------------------------------
# 投影几何（M1 的第二个输出）
# ---------------------------------------------------------------------------
def m1_geometry(sod_mm, odd_mm, pixel_um, n_cols, n_rows=1, n_angles=180,
                span_deg=180.0, voxel_um=None):
    """投影几何与样品面采样网格。

    * 放大率 M = (SOD+ODD)/SOD
    * 样品面有效像素 p_eff = p/M
    * 样品面视场 FOV = n_cols·p_eff（探测器宽 / M）
    * 探测器坐标 u ↔ 样品面坐标 s：s = u/M
    """
    sod, odd = float(sod_mm), float(odd_mm)
    M = (sod + odd) / sod if sod > 0 else 1.0
    p_mm = float(pixel_um) / 1000.0
    p_eff_mm = p_mm / M
    fov_mm = n_cols * p_eff_mm
    fov_z_mm = n_rows * p_eff_mm
    vox = float(voxel_um) / 1000.0 if voxel_um else p_eff_mm
    return {
        'sod_mm': sod, 'odd_mm': odd, 'M': float(M),
        'pixel_mm': p_mm, 'pixel_um': float(pixel_um),
        'p_eff_mm': float(p_eff_mm), 'p_eff_um': float(p_eff_mm * 1000.0),
        'n_cols': int(n_cols), 'n_rows': int(n_rows),
        'fov_mm': float(fov_mm), 'fov_z_mm': float(fov_z_mm),
        'n_angles': int(n_angles), 'span_deg': float(span_deg),
        'ang_step_deg': float(span_deg) / max(int(n_angles) - 1, 1),
        'voxel_mm': float(vox), 'voxel_um': float(vox * 1000.0),
        # 采样网格信息（M1 要求的"样品处采样网格"）
        'grid_pitch_um': float(p_eff_mm * 1000.0),
        'nyquist_cyc_mm': float(1.0 / (2.0 * p_eff_mm)) if p_eff_mm > 0 else None,
        'fov_over_voxel': int(round(fov_mm / vox)) if vox > 0 else None,
    }


# ---------------------------------------------------------------------------
# 验收
# ---------------------------------------------------------------------------
def validate_m1(name='cylinder', n_grid=1024, n_cols=256, pixel_um=55.0,
                sod_mm=100.0, odd_mm=300.0, n_angles=90, sample_mm=10.0,
                mu_scale=0.02, aa=2):
    """按手册 M1 的验收标准，比较解析与数值正向投影。

    返回 dict：几何、两种正弦图的最大/平均相对误差、是否通过（< 1%）。
    相对误差以解析正弦图的峰值归一（线积分幅值），避免在接近零处放大噪声。
    """
    ell = phantom(name)
    g = m1_geometry(sod_mm, odd_mm, pixel_um, n_cols, 1, n_angles, 180.0)
    # Shepp-Logan 外椭圆 a=0.69 → 半轴对应 sample_mm/2
    scale_mm = (sample_mm / 2.0) / 0.69
    # 视场必须由**体模自身范围**决定，不能只看 a=0.69：
    # Shepp-Logan 的 b=0.92 > a=0.69，只看 a 会让栅格把体模截断，
    # 数值弦长随之偏短 —— 这会伪装成"公式错"，实际是网格没铺够。
    fov_mm = max(g['fov_mm'], 2.4 * mu_extent(ell) * scale_mm)
    ang = np.linspace(0.0, 180.0, int(n_angles), endpoint=False)

    a_sino = analytic_sinogram(ell, n_cols, g['p_eff_mm'], ang, scale_mm, mu_scale)

    img = rasterize_phantom(ell, int(n_grid), fov_mm, scale_mm, mu_scale, aa=aa)
    n_sino = numeric_sinogram(img, fov_mm, n_cols, g['p_eff_mm'], ang)

    peak = float(np.max(np.abs(a_sino))) or 1.0
    diff = np.abs(n_sino - a_sino)
    # 只在解析解非零的区域统计（线积分恰好穿过体模外的位置本就是 0）
    mask = np.abs(a_sino) > 0.02 * peak
    rel_max = float(diff[mask].max() / peak) if mask.any() else 0.0
    rel_mean = float(diff[mask].mean() / peak) if mask.any() else 0.0
    rel_p99 = float(np.percentile(diff[mask], 99) / peak) if mask.any() else 0.0

    return {
        'phantom': str(name),
        'geometry': g,
        'n_grid': int(n_grid), 'n_cols': int(n_cols), 'n_angles': int(n_angles),
        'aa': int(aa),
        'fov_mm': float(fov_mm), 'scale_mm': float(scale_mm),
        'voxel_um': float(fov_mm / int(n_grid) * 1000.0),
        'analytic_peak': peak,
        # max 会被"椭圆切线处弦长 √ 奇异"的单格跳变主导，故同时给出 P99 与均值；
        # 三者都随网格加密一阶收敛（体素减半、误差减半），这是离散化的正常行为。
        'rel_err_max': rel_max,
        'rel_err_p99': rel_p99,
        'rel_err_mean': rel_mean,
        'tolerance': 0.01,
        'passed': bool(rel_max < 0.01),
    }
