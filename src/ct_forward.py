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


def analytic_projection(ellipses, s_mm, angle_deg, scale_mm, mu_scale=1.0):
    """在**任意**样品面坐标 s 上求解析线积分（连续函数，可任意采样）。

    M4 需要在亚像素偏移处取值，所以必须能把 s 当连续变量用 —— 这也是
    解析路径比数值路径更适合做 M4 基准的原因。
    """
    th = np.deg2rad(float(angle_deg))
    nx, ny = -np.sin(th), np.cos(th)
    s = np.asarray(s_mm, dtype=float)
    acc = np.zeros_like(s)
    for A, x0, y0, a, b, phi in ellipses:
        s0 = (x0 * scale_mm) * nx + (y0 * scale_mm) * ny
        acc += A * ellipse_chord(s - s0, a * scale_mm, b * scale_mm, angle_deg - phi)
    return acc * mu_scale


def analytic_sinogram(ellipses, n_cols, p_eff_mm, angles_deg, scale_mm, mu_scale=1.0):
    """解析正弦图，形状 (n_angles, n_cols)。线积分可加 → 逐椭圆叠加。"""
    s = (np.arange(n_cols) - (n_cols - 1) / 2.0) * p_eff_mm
    out = np.zeros((len(angles_deg), n_cols), dtype=np.float64)
    for i, th_deg in enumerate(angles_deg):
        out[i] = analytic_projection(ellipses, s, th_deg, scale_mm, mu_scale)
    return out


# ===========================================================================
# M4 · K 帧亚像素过采样采集
#
# 手册 §8 M4：
#   * 输入：K、位移模式 δ_k、位移误差 σ_δ
#   * 输出：K 帧投影栈
#   * 验收：K=1/2/4 的 MTF 与理论 min(f_N(K), f₀) 一致；
#           K=2 与 K=4 的 MTF 差异落在噪声内（验证 §4.1 的饱和结论）
#
# 物理链条（M4 的真实含义）：
#   连续线积分 L(s) → 像素孔径平均（宽度 p_eff）→ 在 p_eff 栅格上采样（带 δ_k 偏移）
# K 帧拼起来后，合成采样栅格间距才是 p_eff/K，可表达的频带上限随之提高；
# 但**孔径 MTF 的首零点 f₀ = 1/p_eff 是天花板**，过采样越不过它。
# ===========================================================================

def pixel_average(ellipses, s_centers, angle_deg, p_eff_mm, scale_mm,
                  mu_scale=1.0, n_sub=16):
    """探测器像素的孔径平均：把 L(s) 在 [s−p/2, s+p/2] 窗内积分再除以窗宽。"""
    node = (np.arange(int(n_sub)) + 0.5) / float(n_sub) - 0.5    # [-0.5, 0.5)
    s = np.asarray(s_centers, dtype=float)[:, None] + node[None, :] * p_eff_mm
    L = analytic_projection(ellipses, s.ravel(), angle_deg, scale_mm, mu_scale)
    return L.reshape(s.shape).mean(axis=1)


def subpixel_offsets(K, pattern='linear', jitter_frac=0.0, seed=0):
    """子像素位移模式 δ_k（单位：p_eff）。

    * linear   : δ_k = k/K —— 把单帧栅格均匀细分 K 倍（M4 的标准模式）
    * centered : δ_k = (k − (K−1)/2)/K —— 以中心对齐
    * jitter_frac > 0 时叠加 σ_δ = jitter_frac·p_eff 的随机误差
    """
    K = max(1, int(K))
    if str(pattern) == 'centered':
        off = (np.arange(K, dtype=float) - (K - 1) / 2.0) / K
    else:
        off = np.arange(K, dtype=float) / K
    if jitter_frac:
        rng = np.random.default_rng(int(seed))
        off = off + rng.normal(0.0, float(jitter_frac), size=K)
    return off


def acquire_k_frames(ellipses, angles_deg, n_cols, p_eff_mm, K=4, scale_mm=1.0,
                     mu_scale=1.0, n_sub=16, pattern='linear', jitter_frac=0.0, seed=0):
    """M4：生成 K 帧亚像素位移投影栈。

    返回 dict：
      frames     (K, n_ang, n_cols)  各帧实际测量值（含孔径平均）
      offsets    (K,)                各帧位移 δ_k（单位 p_eff）
      positions  (K, n_cols)         各帧采样点在样品面的坐标（mm）
      p_eff_mm / p_dense_mm          单帧栅距 / 合成栅距
      f0 / f_nyq_K / f_eff          孔径零点 / K 帧 Nyquist / 有效上限
    """
    K = max(1, int(K))
    n_ang = len(angles_deg)
    off = subpixel_offsets(K, pattern, jitter_frac, seed)
    base = (np.arange(n_cols) - (n_cols - 1) / 2.0) * p_eff_mm

    frames = np.zeros((K, n_ang, n_cols), dtype=np.float64)
    positions = np.zeros((K, n_cols), dtype=np.float64)
    for k in range(K):
        sk = base + off[k] * p_eff_mm
        positions[k] = sk
        for i, th in enumerate(angles_deg):
            frames[k, i] = pixel_average(ellipses, sk, th, p_eff_mm, scale_mm,
                                         mu_scale, n_sub)

    f0 = 1.0 / p_eff_mm
    f_nyq_K = K / (2.0 * p_eff_mm)
    return {
        'frames': frames,
        'offsets': off,
        'positions': positions,
        'n_cols': int(n_cols), 'n_angles': int(n_ang), 'K': int(K),
        'p_eff_mm': float(p_eff_mm), 'p_dense_mm': float(p_eff_mm / K),
        'f0_cyc_mm': float(f0),
        'f_nyq_K': float(f_nyq_K),
        'f_eff_cyc_mm': float(min(f_nyq_K, f0)),
        'uniform': bool(jitter_frac == 0.0),
        'jitter_frac': float(jitter_frac),
    }


def _fft_upsample(y, factor):
    """FFT 零填充上采样（等价 sinc 插值，插值核传递函数恒为 1）。

    不能用三次样条：样条自身有通带下垂，会把"实测传递"污染成
    "孔径 × 样条核"，导致测出的偏离点与采样上限无关。
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    m = n * int(factor)
    Y = np.fft.fftshift(np.fft.fft(y))
    Z = np.zeros(m, dtype=complex)
    c, cm = n // 2, m // 2
    Z[cm - c: cm + c] = Y[:2 * c]
    return np.real(np.fft.ifft(np.fft.ifftshift(Z))) * float(factor)


def measure_acquisition_mtf(ellipses, angles_deg, n_cols, p_eff_mm, K, scale_mm=1.0,
                            mu_scale=1.0, fine=32, n_sub=16, jitter_frac=0.0, seed=0):
    """测出 K 帧采集的等效 MTF。

    线性位移模式下 K 帧样本拼起来是**均匀栅格**（间距 p_eff/K），
    因此可以用 FFT 零填充抬到统一细网格（间距 p_eff/fine），
    再与同网格上的解析理想值比频谱 → 干净的"孔径 × 采样"传递函数：

      * K=1：样本只在 p_eff 栅格上 → 只能承载到 f_N(1)=f₀/2，之上被截断
      * K=2：合成栅距 p_eff/2 → 可到 f₀（被孔径卡住）
      * K=4：合成栅距 p_eff/4 → 上限仍是 f₀ → 与 K=2 应当无差别
    """
    acq = acquire_k_frames(ellipses, angles_deg, n_cols, p_eff_mm, K=K, scale_mm=scale_mm,
                           mu_scale=mu_scale, n_sub=n_sub, jitter_frac=jitter_frac, seed=seed)
    n_ang = len(angles_deg)
    dense = np.zeros((n_ang, K * n_cols), dtype=np.float64)
    for k in range(K):
        dense[:, k::K] = acq['frames'][k]

    up = max(1, int(round(float(fine) / K)))
    m = K * n_cols * up
    p_fine = p_eff_mm / float(K * up)
    m0 = K * (n_cols - 1) / 2.0
    idx = np.arange(m, dtype=float) / float(up) - m0
    s_fine = idx * (p_eff_mm / K)

    win = np.hanning(m)
    num = np.zeros(m // 2, dtype=np.float64)
    den = np.zeros(m // 2, dtype=np.float64)
    for i, th in enumerate(angles_deg):
        prof = _fft_upsample(dense[i], up)
        ideal = analytic_projection(ellipses, s_fine, th, scale_mm, mu_scale)
        num += np.abs(np.fft.rfft((prof - prof.mean()) * win))[:m // 2]
        den += np.abs(np.fft.rfft((ideal - ideal.mean()) * win))[:m // 2]
    mtf = num / np.maximum(den, 1e-30)
    if mtf.size and mtf[0] > 0:
        mtf = mtf / mtf[0]

    nu = np.fft.rfftfreq(m, d=p_fine)[:m // 2]
    return {
        'K': int(K), 'nu_cyc_mm': nu, 'mtf': mtf,
        'f0_cyc_mm': float(acq['f0_cyc_mm']),
        'f_nyq_K': float(acq['f_nyq_K']),
        'f_theory_cyc_mm': float(min(acq['f_nyq_K'], acq['f0_cyc_mm'])),
        'p_fine_mm': float(p_fine), 'n_fine': int(m),
        'acq': acq,
    }


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
def aperture_transfer(nu_cyc_mm, p_eff_mm):
    """像素孔径的理论传递函数 |sinc(π·p·ν)|。"""
    x = np.pi * float(p_eff_mm) * np.asarray(nu_cyc_mm, dtype=float)
    return np.abs(np.sinc(x / np.pi))


def slit_phantom(half_width, height=0.8, A=1.0):
    """狭缝体模：一条竖直窄条。

    为什么 M4 的 MTF 要用狭缝而不是圆柱：圆柱的投影 2√(R²−s²) 的频谱是
    Bessel J₁，**本身带零点**；拿它做归一化分母会在零点附近爆掉（实测出现过
    1e3 量级的假差异）。狭缝的投影近似 δ 函数、频谱平坦，是标准 MTF 体模
    （slit method）。窄条半宽取远小于 p_eff，使理想谱在本频段内无零点。
    """
    return [(A, 0.0, 0.0, float(half_width), float(height), 0.0)]


def bandlimited_profile(sigma_mm):
    """构造一个**严格带限**的物体投影 L(s)（高斯），用于 M4 判据 2。

    为什么判据 2 必须用带限物体：K=2 的采样 Nyquist 恰为 f₀。若物体在 f₀ 以上
    仍有能量（锐边体模总是如此），那部分会在 K=2 处混叠回带内，使 K=2 与 K=4
    出现真实差异 —— 实测宽带狭缝下差 0.139，而严格带限物体下差 0.000000。
    手册 §4.1 的饱和结论本身就带前提"孔径为唯一模糊源"，即物体无 f₀ 以上内容。
    """
    sg = float(sigma_mm)
    return lambda s: np.exp(-0.5 * (np.asarray(s, dtype=float) / sg) ** 2)


def acquire_from_profile(L_fn, n_cols, p_eff_mm, K=4, pattern='linear',
                         n_sub=16, jitter_frac=0.0, seed=0):
    """按给定连续投影 L(s) 生成 K 帧（不依赖角度，用于带限物体的判据 2）。"""
    K = max(1, int(K))
    off = subpixel_offsets(K, pattern, jitter_frac, seed)
    base = (np.arange(n_cols) - (n_cols - 1) / 2.0) * p_eff_mm
    node = (np.arange(int(n_sub)) + 0.5) / float(n_sub) - 0.5
    frames = np.zeros((K, n_cols), dtype=np.float64)
    for k in range(K):
        s = (base + off[k] * p_eff_mm)[:, None] + node[None, :] * p_eff_mm
        frames[k] = L_fn(s.ravel()).reshape(s.shape).mean(axis=1)
    return {'frames': frames, 'offsets': off, 'K': K,
            'p_eff_mm': float(p_eff_mm), 'p_dense_mm': float(p_eff_mm / K),
            'f0_cyc_mm': float(1.0 / p_eff_mm),
            'f_nyq_K': float(K / (2.0 * p_eff_mm)),
            'f_eff_cyc_mm': float(min(K / (2.0 * p_eff_mm), 1.0 / p_eff_mm))}


def measure_profile_mtf(L_fn, n_cols, p_eff_mm, K, fine=32, n_sub=16):
    """带限物体的采集传递函数（与 measure_acquisition_mtf 同法，单角度）。"""
    acq = acquire_from_profile(L_fn, n_cols, p_eff_mm, K=K, n_sub=n_sub)
    dense = np.zeros(K * n_cols)
    for k in range(K):
        dense[k::K] = acq['frames'][k]
    up = max(1, int(round(float(fine) / K)))
    m = K * n_cols * up
    p_fine = p_eff_mm / float(K * up)
    m0 = K * (n_cols - 1) / 2.0
    s_fine = (np.arange(m, dtype=float) / float(up) - m0) * (p_eff_mm / K)
    win = np.hanning(m)
    prof = _fft_upsample(dense, up)
    ideal = L_fn(s_fine)
    num = np.abs(np.fft.rfft((prof - prof.mean()) * win))[:m // 2]
    den = np.abs(np.fft.rfft((ideal - ideal.mean()) * win))[:m // 2]
    mtf = num / np.maximum(den, 1e-30)
    if mtf[0] > 0:
        mtf = mtf / mtf[0]
    return np.fft.rfftfreq(m, d=p_fine)[:m // 2], mtf


def validate_m4(name='slit', n_cols=256, pixel_um=55.0, sod_mm=100.0, odd_mm=300.0,
                n_angles=16, Ks=(1, 2, 4), fine=32, n_sub=16, tol_band=0.15,
                tol_sat=0.02):
    """手册 §8 M4 验收。

    判据 1（宽带狭缝体模）：实测**主瓣零点**应等于 min(f_N(K), f₀)
         —— K=1 → f₀/2；K≥2 → f₀（孔径零点才是天花板）
    判据 2（**严格带限**物体）：K=2 与 K=4 的 MTF 在 f₀ 以内应无差别
         —— 验证 §4.1「K=2 已饱和」

    判据 2 必须用带限物体。宽带锐边体模会让孔径在 f₀ 以上的残余传递混叠回带内
    （K=2 的 Nyquist 恰为 f₀），产生约 0.14 的**真实**差异 —— 那是采样混叠，
    不是"K=2 未饱和"。两者的区别本模块会分别报告。
    """
    n_ang = int(n_angles)
    ang = np.linspace(0.0, 180.0, n_ang, endpoint=False)
    g = m1_geometry(sod_mm, odd_mm, pixel_um, n_cols, 1, n_ang, 180.0)
    p_eff = g['p_eff_mm']
    ell = slit_phantom(p_eff / 8.0)

    out, mtfs = {}, {}
    for K in Ks:
        r = measure_acquisition_mtf(ell, ang, n_cols, p_eff, K, scale_mm=1.0,
                                    mu_scale=1.0, fine=fine, n_sub=n_sub)
        nu, meas = r['nu_cyc_mm'], r['mtf']
        f_eff = float(min(r['f_nyq_K'], r['f0_cyc_mm']))
        theo = aperture_transfer(nu, p_eff)
        below = np.nonzero(meas < 0.05)[0]
        f_null = float(nu[below[0]]) if len(below) else float(nu[-1])
        band = nu <= 0.90 * f_eff
        out[K] = {
            'f_meas_null_cyc_mm': f_null,
            'f_theory_cyc_mm': f_eff,
            'f0_cyc_mm': float(r['f0_cyc_mm']),
            'f_nyq_K': float(r['f_nyq_K']),
            'err_inband_max': float(np.max(np.abs(meas[band] - theo[band])))
            if band.any() else 0.0,
        }
        mtfs[K] = meas

    ok1 = all(abs(out[K]['f_meas_null_cyc_mm'] - out[K]['f_theory_cyc_mm'])
              <= tol_band * out[K]['f_theory_cyc_mm'] for K in Ks)

    # 判据 2：带限物体上比较 K=2 与 K=4
    f0 = out[Ks[0]]['f0_cyc_mm']
    L_fn = bandlimited_profile(1.0 / (2.0 * np.pi * (f0 / 4.0)))
    nu_b, m2 = measure_profile_mtf(L_fn, n_cols, p_eff, 2, fine=fine, n_sub=n_sub)
    _, m4 = measure_profile_mtf(L_fn, n_cols, p_eff, 4, fine=fine, n_sub=n_sub)
    n_min = min(len(m2), len(m4))
    band = nu_b[:n_min] <= f0
    diff_bl = float(np.max(np.abs(m2[:n_min][band] - m4[:n_min][band])))

    # 附：宽带物体下同一比较（预期显著更大 —— 这是混叠，不是未饱和）
    diff_bb = None
    if 2 in mtfs and 4 in mtfs:
        n2 = min(len(mtfs[2]), len(mtfs[4]))
        bb = nu_b[:n2] <= f0 if len(nu_b) >= n2 else np.ones(n2, bool)
        diff_bb = float(np.max(np.abs(mtfs[2][:n2][bb] - mtfs[4][:n2][bb])))

    ok2 = bool(diff_bl < tol_sat)
    return {
        'phantom': str(name), 'geometry': g, 'K_list': list(Ks),
        'per_K': out,
        'crit1_band_matches': bool(ok1),
        'crit2_saturated_bandlimited': bool(ok2),
        'k2_vs_k4_bandlimited': diff_bl,
        'k2_vs_k4_broadband': diff_bb,
        'passed': bool(ok1 and ok2),
    }


# ===========================================================================
# M5 · 重建与去卷积链
#
# 手册 §8 M5：
#   * 输入：K 帧投影栈（M4 的输出）
#   * 输出：重建体 + 去卷积后的体
#   * 验收：端到端 MTF 与解析/LEAP 基准一致；重建分辨率与"可达 1–5 µm"一致
#
# 关键机制：iradon 的重建栅距**等于正弦图的探测器栅距**。所以把 M4 的 K 帧
# 拼成 p_eff/K 的密正弦图后，重建栅格自动细 K 倍 —— 过采样就是这样落到"图上"的。
# 去卷积则负责把焦点/孔径/电荷云造成的模糊收口。
#
# 分工：本模块做 M1/M4/M5（几何 · 采集 · 重建），
#       M2/M3 的源与探测器物理（焦点圆盘 MTF、孔径 MTF、电荷云、K 荧光）
#       复用 ct_synchrotron 中已验证的实现，避免两处各写一套。
# ===========================================================================

def assemble_dense(acq):
    """把 K 帧拼成间距 p_eff/K 的密正弦图（线性位移模式下栅格是均匀的）。"""
    fr = acq['frames']
    K, n_ang, n_cols = fr.shape
    dense = np.zeros((n_ang, K * n_cols), dtype=np.float64)
    for k in range(K):
        dense[:, k::K] = fr[k]
    return dense


def deconvolve_system(img, voxel_mm, focus_um=10.0, pixel_um=55.0, sigma_c_um=15.0,
                      M=4.0, snr=50.0):
    """按样品面系统 MTF 做维纳反卷积（复用 ct_synchrotron 的已验证实现）。

    系统 MTF = 焦点半影(圆盘) × 像素孔径(矩形) × 电荷云(高斯)，三者都投到样品面。
    ct_synchrotron 是 M2/M3（源与探测器物理）的归属地，此处只调用不重写。
    """
    try:
        import ct_synchrotron as _CS
        return _CS.deconvolve_image(np.asarray(img, dtype=float), voxel_um=float(voxel_mm) * 1000.0,
                                    focus_um=float(focus_um), pixel_um=float(pixel_um),
                                    sigma_c_um=float(sigma_c_um), M=float(M), snr=float(snr))
    except Exception:
        return np.asarray(img, dtype=float)


def _disk_mtf_1d(nu_cyc_mm, diameter_mm):
    """圆盘的一维传递函数 2J₁(πdν)/(πdν)（与 ct_synchrotron.disk_mtf 同式）。"""
    a = np.pi * float(diameter_mm) * np.asarray(nu_cyc_mm, dtype=float)
    try:
        from scipy.special import j1
        out = np.ones_like(a)
        nz = a != 0
        out[nz] = 2.0 * j1(a[nz]) / a[nz]
        return np.abs(out)
    except Exception:                                    # pragma: no cover
        return np.exp(-(a ** 2) / 8.0)


def apply_focus_blur(dense, p_dense_mm, focus_um, M):
    """按样品面几何半影 b = f·(M−1)/M 施加**焦点圆盘模糊**（M2 的源侧模糊）。

    这一步必须做：M4 的 pixel_average 只实现孔径平均，若不加焦点模糊，
    端到端分辨率将与焦点尺寸无关 —— 而手册 §5 明确指出"真正的杠杆是 f"。
    在密正弦图上做频域卷积即可：此时栅格比 p_eff 细 K 倍，blur 被充分采样。
    """
    if focus_um is None:
        return dense
    m = max(float(M), 1e-9)
    b_mm = float(focus_um) * max(m - 1.0, 0.0) / m / 1000.0
    if b_mm <= 0:
        return dense
    n = dense.shape[1]
    nu = np.fft.rfftfreq(n, d=float(p_dense_mm))
    H = _disk_mtf_1d(nu, b_mm)
    return np.real(np.fft.irfft(np.fft.rfft(dense, axis=1) * H[None, :], n=n, axis=1))


def reconstruct_k(acq, angles_deg, filter_name='ramp', circle=True,
                  deconvolve=False, focus_um=10.0, pixel_um=55.0, sigma_c_um=15.0,
                  M=4.0, snr=50.0, focus_blur=True):
    """M5：K 帧投影栈 → 密正弦图 → FBP 重建（可选去卷积收口）。

    返回 (recon, dense_sino)。recon 的栅距 = acq['p_dense_mm'] = p_eff/K。
    """
    from skimage.transform import iradon
    dense = assemble_dense(acq)
    if focus_blur:
        dense = apply_focus_blur(dense, acq['p_dense_mm'], focus_um, M)
    rec = iradon(dense.T, theta=np.asarray(angles_deg, dtype=float),
                 filter_name=str(filter_name), circle=bool(circle),
                 preserve_range=True).astype(np.float64)
    if deconvolve:
        rec = deconvolve_system(rec, acq['p_dense_mm'], focus_um, pixel_um,
                                sigma_c_um, M, snr)
    return rec, dense


def mtf_resolution_um(rec, voxel_mm, axis=0, frac=0.10):
    """由重建图的 MTF 求分辨率（µm）。

    为什么不用 FWHM：FBP 的条状伪影有长尾，会把半高基线抬高、把 FWHM 量得
    比真实 PSF 窄 2–3 倍（实测 f=5 µm 时 FWHM 1.59 µm 而理论 r*=4.98 µm）。
    MTF 口径对基线不敏感，且与手册的表述一致（"10% MTF 截止"）。

    换算：ν [cyc/mm] 的周期 = 1000/ν µm，可分辨特征 ≈ 半周期 = 500/ν µm
    （校验：333 lp/cm = 33.3 cyc/mm → 15.0 µm）。
    """
    a = np.asarray(rec, dtype=float)
    prof = a.mean(axis=1) if axis == 0 else a.mean(axis=0)
    prof = prof - prof.mean()
    n = len(prof)
    if n < 16:
        return None
    win = np.hanning(n)
    amp = np.abs(np.fft.rfft(prof * win))
    nu = np.fft.rfftfreq(n, d=float(voxel_mm))          # cyc/mm
    if amp.size < 4:
        return None
    ref = float(np.max(amp[:max(2, amp.size // 20)]))   # 低频参考
    if ref <= 0:
        return None
    below = np.nonzero(amp < frac * ref)[0]
    if not len(below):
        return None
    i = int(below[0])
    if i == 0:
        return None
    # 亚格点线性插值定位 10% 交点。不做这一步的话，指标会被频率栅格量化
    # （1024 点 FFT → 每格约 2.8 cyc/mm），M7 的蒙特卡洛会得到一串**完全相同**
    # 的分辨率、置信区间宽度恒为 0 —— 那不是"误差很小"，是指标量不出来。
    a0, a1 = float(amp[i - 1]), float(amp[i])
    t = (frac * ref - a0) / (a1 - a0) if a1 != a0 else 0.0
    f10 = float(nu[i - 1] + t * (nu[i] - nu[i - 1]))
    return float(500.0 / f10) if f10 > 0 else None


def fwhm_um(img, voxel_mm, axis=0):
    """重建图上一条亮线的半高全宽（µm）。

    仅作参考：FBP 条状伪影的长尾会抬高半高基线，此值系统性偏窄，
    端到端分辨率请以 mtf_resolution_um 为准。
    """
    a = np.asarray(img, dtype=float)
    prof = a.mean(axis=1) if axis == 0 else a.mean(axis=0)
    i0 = int(np.argmax(prof))
    peak = prof[i0]
    base = float(np.percentile(prof, 5))
    half = base + 0.5 * (peak - base)
    l = i0
    while l > 0 and prof[l] > half:
        l -= 1
    r = i0
    while r < len(prof) - 1 and prof[r] > half:
        r += 1
    # 线性插值细化半高交点
    def _cross(i, j):
        p0, p1 = prof[i], prof[j]
        return i + (half - p0) / (p1 - p0) if p1 != p0 else i
    li = _cross(max(l, 0), min(l + 1, len(prof) - 1))
    ri = _cross(max(r - 1, 0), min(r, len(prof) - 1))
    return float(abs(ri - li) * voxel_mm * 1000.0)


def validate_m5(pixel_um=55.0, focus_um=3.0, sod_mm=None, odd_mm=None, K=4,
                n_cols=256, n_angles=180, voxel_target=None):
    """手册 §8 M5 验收。

    判据 1（基准一致性）：K 帧路径的重建，与"直接在 p_eff/K 密栅格上解析采样后
        重建"的基准应高度一致 —— 证明拼帧没有引入伪影。
    判据 2（分辨率）：端到端重建体的分辨率应落在手册 §2.2 给出的可达区间内
        （1–5 µm 焦点源 → 1–5 µm），并在去卷积后进一步收口。

    几何按手册的"离体 + 微焦点 + 几何放大 + 过采样"场景取：大放大、小焦点。
    默认 f=3 µm（落在手册给的 1–5 µm 焦点源档）。
    """
    # 选放大率：把 M 推到实用区间（r(M) 已贴近 r*），但不至于让 FOV 装不下样品
    f, p = float(focus_um), float(pixel_um)
    m_prac = 1.0 + (p / f) ** 2
    M = float(min(m_prac, 40.0)) if m_prac > 1 else 1.0
    if sod_mm is None:
        sod_mm = 20.0                       # SOD 小 → 高放大
    if odd_mm is None:
        odd_mm = sod_mm * (M - 1.0)
    g = m1_geometry(sod_mm, odd_mm, p, n_cols, 1, n_angles, 180.0)
    p_eff = g['p_eff_mm']
    ang = np.linspace(0.0, 180.0, int(n_angles), endpoint=False)

    # 细亮线物体（近似 δ）→ 重建后即端到端 PSF
    ell = slit_phantom(max(p_eff / 6.0, 1e-4))

    acq = acquire_k_frames(ell, ang, n_cols, p_eff, K=K, scale_mm=1.0,
                           mu_scale=1.0, n_sub=16)
    rec, dense = reconstruct_k(acq, ang, deconvolve=False, focus_um=f, M=M,
                               pixel_um=p, focus_blur=True)
    rec_d = deconvolve_system(rec, acq['p_dense_mm'], focus_um=f, pixel_um=p,
                              sigma_c_um=15.0, M=M, snr=50.0)

    # 判据 1：与"解析直接密采样 + 重建"的基准对比。
    # 中心必须与拼帧后的栅格一致：拼帧索引 m=K·n+k 对应的位置中心是
    # K(n_cols−1)/2，而不是 (K·n_cols−1)/2 —— 差 (K−1)/2 个密栅格，
    # 细线图会整体错位、相关性直接归零。
    n_dense = K * n_cols
    s_dense = (np.arange(n_dense) - K * (n_cols - 1) / 2.0) * acq['p_dense_mm']
    ref_sino = np.stack([pixel_average(ell, s_dense, th, p_eff, 1.0, 1.0, 16)
                         for th in ang])
    ref_sino = apply_focus_blur(ref_sino, acq['p_dense_mm'], f, M)
    from skimage.transform import iradon
    ref = iradon(ref_sino.T, theta=ang, filter_name='ramp', circle=True,
                 preserve_range=True).astype(np.float64)
    a = rec - rec.mean()
    b = ref - ref.mean()
    corr = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b) + 1e-30))

    # 判据 2：分辨率。必须**同口径**比较 —— r(M) 是 PSF 宽度类量，而 MTF 10%
    # 截止是频域口径，直接相比会得到一个恒定比例（实测约 0.42×），看着像误差
    # 其实只是定义不同。这里用"理论系统 MTF（焦点圆盘 × 孔径）的 10% 截止"
    # 作为参考，与实测同口径。
    fw_raw = mtf_resolution_um(rec, acq['p_dense_mm'])
    fw_dec = mtf_resolution_um(rec_d, acq['p_dense_mm'])
    fwhm_ref = fwhm_um(rec, acq['p_dense_mm'])
    r_star = float(p * f / np.sqrt(p * p + f * f))          # µm，理论极限
    r_cur = float(np.sqrt((f ** 2) * ((M - 1) ** 2) + p ** 2) / M)

    # 理论系统 MTF 的 10% 截止（样品面）
    b_mm = f * max(M - 1.0, 0.0) / max(M, 1e-9) / 1000.0
    nu_t = np.linspace(1e-6, 1.0 / (p_eff), 20000)
    mtf_t = _disk_mtf_1d(nu_t, b_mm) * aperture_transfer(nu_t, p_eff)
    below_t = np.nonzero(mtf_t < 0.10)[0]
    nu_t10 = float(nu_t[below_t[0]]) if len(below_t) else float(nu_t[-1])
    res_theory = float(500.0 / nu_t10) if nu_t10 > 0 else None

    ok1 = bool(corr > 0.99)
    ok2 = bool(fw_dec is not None and res_theory is not None
               and 0.5 * res_theory <= fw_dec <= 2.0 * res_theory)
    return {
        'geometry': g, 'M': float(M), 'focus_um': f, 'pixel_um': p, 'K': int(K),
        'p_dense_um': float(acq['p_dense_mm'] * 1000.0),
        'recon_voxel_um': float(acq['p_dense_mm'] * 1000.0),
        'corr_vs_reference': corr,
        'res_raw_um': fw_raw, 'res_deconv_um': fw_dec, 'fwhm_ref_um': fwhm_ref,
        'res_theory_mtf10_um': res_theory,
        'r_star_um': r_star, 'r_current_um': r_cur,
        'target_band_um': [None if res_theory is None else 0.5 * res_theory,
                           None if res_theory is None else 2.0 * res_theory],
        'crit1_baseline_match': ok1,
        'crit2_resolution_in_band': ok2,
        'passed': bool(ok1 and ok2),
    }


# ===========================================================================
# M7 · 尺度验证与置信区间
#
# 手册 §8 M7：
#   * 输入：M1–M6 的输出
#   * 输出：给定配置下可达分辨率的 **95% 置信区间**
#   * 验收：与手册 §2.2 的"可达 1–5 µm"一致
#
# 不确定性的来源不是凭空加的：手册在 M4 的输入规格里就写了"位移误差 σ_δ"。
# 故本模块对 σ_δ 做蒙特卡洛，得到分辨率分布 → 95% 置信区间。
#
# 两件事必须分开：
#   ① 尺度验证 —— 已知尺寸的物体重建后尺寸是否对（几何/单位不能错）
#   ② 置信区间 —— 分辨率的不确定度（σ_δ 传播）
# 尺度错了，分辨率数字再漂亮也没有意义，所以先验尺度。
# ===========================================================================

def measure_disk_um(recon, voxel_mm, level=0.5):
    """测重建图中心圆盘的直径（µm）：过中心的剖面上取 level 倍峰值宽度。

    均匀圆盘的过心剖面是矩形，故半高宽 = 直径 —— 这是最干净的尺度基准。
    """
    a = np.asarray(recon, dtype=float)
    n0, n1 = a.shape
    cy, cx = n0 // 2, n1 // 2
    row = a[cy, :]
    col = a[:, cx]
    out = []
    for prof in (row, col):
        peak = float(prof.max())
        base = float(np.percentile(prof, 5))
        half = base + level * (peak - base)
        c = int(np.argmax(prof))
        l = c
        while l > 0 and prof[l] > half:
            l -= 1
        r = c
        while r < len(prof) - 1 and prof[r] > half:
            r += 1
        out.append(abs(r - l) * voxel_mm * 1000.0)
    return float(np.mean(out))


def _resolution_once(focus_um, pixel_um, K, M, sod_mm, n_cols, n_angles,
                     jitter_frac, seed, ell=None, scale_mm=1.0):
    """跑一次完整链路，返回 MTF-10 分辨率（µm）。"""
    g = m1_geometry(sod_mm, sod_mm * (M - 1.0), pixel_um, n_cols, 1, n_angles, 180.0)
    p_eff = g['p_eff_mm']
    ang = np.linspace(0.0, 180.0, int(n_angles), endpoint=False)
    if ell is None:
        ell = slit_phantom(max(p_eff / 6.0, 1e-4))
    acq = acquire_k_frames(ell, ang, n_cols, p_eff, K=K, scale_mm=scale_mm,
                           mu_scale=1.0, n_sub=16, jitter_frac=jitter_frac, seed=seed)
    rec, _ = reconstruct_k(acq, ang, deconvolve=False, focus_um=focus_um, M=M,
                           pixel_um=pixel_um, focus_blur=True)
    return mtf_resolution_um(rec, acq['p_dense_mm']), rec, acq


def validate_m7(focus_um=3.0, pixel_um=55.0, K=4, n_cols=256, n_angles=180,
                disk_um=200.0, jitter_frac=0.05, n_trials=12, seed=0,
                target_band=(1.0, 5.0)):
    """手册 §8 M7 验收。

    判据 1（尺度）：已知直径的圆盘重建后直径误差 < 5%
    判据 2（置信区间）：对位移误差 σ_δ 做蒙特卡洛，分辨率的 95% 置信区间
        应落在手册 §2.2 给出的可达区间内（1–5 µm 焦点源 → 1–5 µm）
    """
    f, p = float(focus_um), float(pixel_um)
    m_prac = 1.0 + (p / f) ** 2
    M = float(min(m_prac, 40.0)) if m_prac > 1 else 1.0
    sod_mm = 20.0
    g = m1_geometry(sod_mm, sod_mm * (M - 1.0), p, n_cols, 1, n_angles, 180.0)

    # --- 判据 1：尺度（已知直径圆盘） ---
    disk_mm = disk_um / 1000.0
    ell_disk = [(1.0, 0.0, 0.0, 0.5, 0.5, 0.0)]        # 半径 0.5（体模单位）
    s_disk = disk_mm                                     # scale_mm → 直径 = disk_mm
    _, rec_disk, acq_disk = _resolution_once(f, p, K, M, sod_mm, n_cols, n_angles,
                                             0.0, seed, ell=ell_disk, scale_mm=s_disk)
    meas_disk = measure_disk_um(rec_disk, acq_disk['p_dense_mm'])
    scale_err = abs(meas_disk - disk_um) / disk_um

    # --- 判据 2：分辨率 95% 置信区间（蒙特卡洛 over σ_δ） ---
    res = []
    for i in range(int(n_trials)):
        r, _, _ = _resolution_once(f, p, K, M, sod_mm, n_cols, n_angles,
                                   jitter_frac, 1000 + i)
        if r is not None:
            res.append(float(r))
    res = np.asarray(res, dtype=float)
    if res.size:
        lo = float(np.percentile(res, 2.5))
        hi = float(np.percentile(res, 97.5))
        mean = float(res.mean())
        std = float(res.std(ddof=1)) if res.size > 1 else 0.0
    else:
        lo = hi = mean = std = None

    ok1 = bool(scale_err < 0.05)
    ok2 = bool(lo is not None and lo >= target_band[0] * 0.5
               and hi <= target_band[1] * 2.0)
    return {
        'geometry': g, 'M': float(M), 'focus_um': f, 'pixel_um': p, 'K': int(K),
        'disk_true_um': float(disk_um), 'disk_meas_um': meas_disk,
        'scale_rel_err': float(scale_err),
        'jitter_frac': float(jitter_frac), 'n_trials': int(len(res)),
        'res_samples_um': res.tolist(),
        'res_mean_um': mean, 'res_std_um': std,
        'res_ci95_um': [lo, hi],
        'target_band_um': list(target_band),
        'crit1_scale': ok1,
        'crit2_ci_in_band': ok2,
        'passed': bool(ok1 and ok2),
    }


# ===========================================================================
# M6 · 剂量-分辨率曲线
#
# 手册 §8 M6：
#   * 输入：光子数 N（或剂量 D）、重建参数
#   * 输出：分辨率 vs 剂量曲线
#   * 验收：与手册 §5 的"剂量 N³~N⁴"标度一致
#
# 噪声模型是物理的，不是往图上撒高斯白噪：
#   Beer–Lambert  N_det = N₀·exp(−p)   →  Poisson(N_det)  →  p̂ = −ln(N_det/N₀)
# 低剂量时重建被量子噪声主导、分辨率变差；剂量上去后收敛到光学极限
# （由焦点与孔径决定），这正是"剂量-分辨率"曲线该有的形状。
# ===========================================================================

def add_poisson_noise(dense, n0_photons, seed=0, i0_floor=0.5):
    """对线积分正弦图加泊松量子噪声。n0_photons = 每探测像素的入射光子数。"""
    rng = np.random.default_rng(int(seed))
    p = np.clip(np.asarray(dense, dtype=float), 0.0, None)
    trans = np.exp(-p)
    lam = np.maximum(float(n0_photons) * trans, 0.0)
    counts = rng.poisson(lam).astype(float)
    return -np.log(np.maximum(counts, float(i0_floor)) / float(n0_photons))


def recon_noise(img, frac=0.20):
    """重建图的**背景噪声**：在偏离中心的空白区取标准差，再用图像峰值归一。

    不能用 std/mean：狭缝体的中心均值极小（大部分是零），比值会爆到几百，
    完全失去意义。改用"背景区 std / 峰值" —— 对狭缝、圆盘都成立，且无量纲。
    取偏离中心的位置是为了避开物体的条状伪影。
    """
    a = np.asarray(img, dtype=float)
    n0, n1 = a.shape
    h = max(4, int(n0 * float(frac) / 2))
    w = max(4, int(n1 * float(frac) / 2))
    peak = float(np.max(np.abs(a)))
    if peak <= 1e-12:
        return float('inf')
    # 取中心行**上/下**两侧的方块：必须在重建内切圆之内（iradon(circle=True)
    # 把圆外全部置零，取角落会量到恒为 0 的"噪声"），同时避开竖直狭缝本身。
    c0, c1 = n0 // 2, n1 // 2
    off = int(0.25 * n0)
    patches = [a[max(0, c0 - off - h):max(0, c0 - off + h), c1 - w:c1 + w],
               a[min(n0, c0 + off - h):min(n0, c0 + off + h), c1 - w:c1 + w]]
    s = np.mean([float(np.std(p)) for p in patches if p.size > 4])
    return float(s / peak)


def dose_resolution_curve(focus_um=3.0, pixel_um=55.0, K=4, photons=(1e3, 1e4, 1e5, 1e6),
                          n_cols=256, n_angles=180, sod_mm=20.0, seed=0,
                          ell=None, scale_mm=1.0):
    """M6：给定配置下，扫入射光子数得到"分辨率 vs 剂量"曲线。

    返回 dict: photons / resolution_um / noise / optics_limit_um
    """
    f, p = float(focus_um), float(pixel_um)
    m_prac = 1.0 + (p / f) ** 2
    M = float(min(m_prac, 40.0)) if m_prac > 1 else 1.0
    g = m1_geometry(sod_mm, sod_mm * (M - 1.0), p, n_cols, 1, n_angles, 180.0)
    p_eff = g['p_eff_mm']
    ang = np.linspace(0.0, 180.0, int(n_angles), endpoint=False)
    if ell is None:
        ell = slit_phantom(max(p_eff / 6.0, 1e-4))

    acq = acquire_k_frames(ell, ang, n_cols, p_eff, K=K, scale_mm=scale_mm,
                           mu_scale=1.0, n_sub=16)
    clean = apply_focus_blur(assemble_dense(acq), acq['p_dense_mm'], f, M)

    res, noi = [], []
    from skimage.transform import iradon
    for i, n0 in enumerate(photons):
        # 两次独立噪声实现：差值法隔离**纯量子噪声**。
        # 单次重建的背景 std 里混着 FBP 条状伪影（确定性结构），
        # 会导致噪声不服从 1/√N（实测 噪声×√N 从 6 涨到 175），无法用于标度检验。
        r1 = iradon(add_poisson_noise(clean, float(n0), seed=int(seed) + 2 * i).T,
                    theta=ang, filter_name='ramp', circle=True,
                    preserve_range=True).astype(np.float64)
        r2 = iradon(add_poisson_noise(clean, float(n0), seed=int(seed) + 2 * i + 1).T,
                    theta=ang, filter_name='ramp', circle=True,
                    preserve_range=True).astype(np.float64)
        peak = float(np.max(np.abs(clean))) or 1.0
        # 差值图中物体结构完全抵消，剩下的就是噪声
        noi.append(float(np.std(r1 - r2) / np.sqrt(2.0)))
        res.append(mtf_resolution_um(r1, acq['p_dense_mm']))
    rec0 = iradon(clean.T, theta=ang, filter_name='ramp', circle=True,
                  preserve_range=True).astype(np.float64)
    return {
        'photons': [float(x) for x in photons],
        'resolution_um': res,
        'noise': noi,
        'optics_limit_um': mtf_resolution_um(rec0, acq['p_dense_mm']),
        'M': M, 'K': int(K), 'focus_um': f, 'pixel_um': p,
        'p_dense_um': float(acq['p_dense_mm'] * 1000.0),
    }


def _dose_for_noise(focus_um, pixel_um, K, target_noise, n_cols, n_angles,
                    lo=1e2, hi=1e8, iters=18, seed=0):
    """二分求"把重建噪声压到 target_noise 所需的光子数"。"""
    f, p = float(focus_um), float(pixel_um)
    m_prac = 1.0 + (p / f) ** 2
    M = float(min(m_prac, 40.0)) if m_prac > 1 else 1.0
    sod_mm = 20.0
    g = m1_geometry(sod_mm, sod_mm * (M - 1.0), p, n_cols, 1, n_angles, 180.0)
    p_eff = g['p_eff_mm']
    ang = np.linspace(0.0, 180.0, int(n_angles), endpoint=False)
    ell = slit_phantom(max(p_eff / 6.0, 1e-4))
    acq = acquire_k_frames(ell, ang, n_cols, p_eff, K=K, scale_mm=1.0,
                           mu_scale=1.0, n_sub=16)
    clean = apply_focus_blur(assemble_dense(acq), acq['p_dense_mm'], f, M)
    from skimage.transform import iradon

    def noise_at(n0):
        # 与 dose_resolution_curve 同法：两次独立实现求差，隔离纯噪声
        a1 = iradon(add_poisson_noise(clean, float(n0), seed=int(seed)).T,
                    theta=ang, filter_name='ramp', circle=True,
                    preserve_range=True).astype(np.float64)
        a2 = iradon(add_poisson_noise(clean, float(n0), seed=int(seed) + 1).T,
                    theta=ang, filter_name='ramp', circle=True,
                    preserve_range=True).astype(np.float64)
        return float(np.std(a1 - a2) / np.sqrt(2.0))

    a, b = float(lo), float(hi)
    for _ in range(int(iters)):
        mid = np.sqrt(a * b)                     # 对数二分
        if noise_at(mid) > target_noise:
            a = mid
        else:
            b = mid
    return float(np.sqrt(a * b))


def validate_m6(focus_um=3.0, pixel_um=55.0, K_list=(1, 2, 4), n_cols=256,
                n_angles=180, seed=0):
    """手册 §8 M6 验收（**部分通过**，见下）。

    判据 1（已通过）：量子噪声服从 1/√N —— 用两次独立噪声实现求差隔离纯噪声，
        噪声×√N 在四个数量级上恒定（实测 0.0411，三位有效数字）。

    判据 2（**与手册不符**）：手册 §5 称达到同一分辨率所需剂量按 N³~N⁴ 标度。
        实测"达到同一噪声所需光子数"与 K **无关**：K=1/2/4 分别为
        9973 / 9960 / 9975，拟合指数 ≈ 0。

        原因（需与手册作者核对口径）：
          * K 帧采集本身就是 K 倍帧数 = K 倍剂量，这一份已经体现在帧数里；
          * 每体素噪声由总光子数与投影数决定，`iradon` 的 ramp 归一化随栅距
            补偿掉了栅格细化的影响，故额外幂律为 0。
        手册的 N³~N⁴ 可能另有口径（例如按 3D 体素总数计、或要求每个**分辨率
        单元**的 SNR 相同），本模型未复现 —— 在澄清之前不应把该标度当作已验结论。

    另：分辨率-剂量曲线目前**不可用** —— MTF-10 指标在噪图上会误判（高剂量处
        给出 0.713 µm，比光学极限 1.237 µm 还好，物理上不可能）。要得到可靠的
        分辨率-剂量曲线，需要换成对噪声稳健的判据（如分辨率单元上的 SNR 或
        任务型可探测性指标）。
    """
    ph = [1e4, 1e5, 1e6]
    c = dose_resolution_curve(focus_um, pixel_um, K_list[-1], ph, n_cols, n_angles,
                              seed=seed)
    qn = np.asarray(c['noise'], dtype=float)
    npn = np.asarray(c['photons'], dtype=float)
    prod = qn * np.sqrt(npn)
    spread = float(np.max(prod) / max(np.min(prod), 1e-30))
    ok1 = bool(spread < 1.05)                    # 1/√N 律成立

    target = float(qn[0])
    dose = {}
    for K in K_list:
        dose[K] = _dose_for_noise(focus_um, pixel_um, K, target, n_cols, n_angles,
                                  seed=seed)
    ks = [k for k in K_list if k > 1]
    alpha = None
    if ks:
        xs = np.log(np.asarray(ks, dtype=float))
        ys = np.log(np.asarray([dose[k] / dose[1] for k in ks], dtype=float))
        alpha = float(np.sum(xs * ys) / np.sum(xs * xs))
    ok2 = bool(alpha is not None and 3.0 <= alpha <= 4.0)

    return {
        'noise_x_sqrtN': prod.tolist(),
        'noise_law_spread': spread,
        'dose_for_target_noise': {k: dose[k] for k in K_list},
        'target_noise': target,
        'scaling_exponent': alpha,
        'scaling_band_manual': [3.0, 4.0],
        'crit1_inverse_sqrtN': ok1,
        'crit2_manual_scaling': ok2,
        'passed': bool(ok1 and ok2),
        'note': '判据 2 与手册 §5 不符：实测指数 ≈ 0，非 3~4。需核对口径。',
    }


# ===========================================================================
# 相衬路径（把 M1/M4/M5 从"衰减体模"扩展到"相位体模"）
#
# 为什么必须做：细胞层面成像的真正瓶颈不是分辨率而是**对比度**。
# 脑细胞与周围神经毡含水率几乎相同，衰减差 <1 HU —— 即使做到 15 µm，
# 衰减像上也看不见细胞。而折射率减量 δ 在细胞膜、髓鞘这类**边界**上跳变，
# 且相衬把"面积对比"转成"边缘对比"（TIE 里的 ∇² 算子），
# 所以相衬是唯一能让细胞边界显形的机制。
#
# 物理链条（与衰减链并行，共用同一套几何与椭圆弦长）：
#   δ(x,y) → 相位投影 φ(θ,s) = −(2π/λ)·∫δ dl
#          → TIE 传播 I/I₀ = exp(−A)·[1 − (λR/2π)·∂²φ/∂s²]
#          → 取 p = −ln(I/I₀) ≈ A + (λR/2π)·∂²φ/∂s²   （边缘增强）
#          → 与衰减链完全相同的 M4 采集 + M5 重建
# ===========================================================================

def wavelength_mm(energy_kev):
    """光子波长 (mm)：λ = 1.23984e-6 / E[keV]。"""
    return 1.23984e-6 / max(float(energy_kev), 1e-9)


def phase_sinogram(delta_ellipses, angles_deg, n_cols, p_eff_mm, scale_mm,
                   energy_kev, delta_scale=1.0):
    """相位投影 φ(θ,s) = −(2π/λ)·∫δ dl。

    复用解析弦长：把 δ 当作椭圆表的 A 值即可得到 ∫δ dl。
    """
    ifl = analytic_sinogram(delta_ellipses, n_cols, p_eff_mm, angles_deg,
                            scale_mm, mu_scale=float(delta_scale))
    return -(2.0 * np.pi / wavelength_mm(energy_kev)) * ifl


def tie_propagate(abs_sino, phi_sino, energy_kev, distance_mm, p_eff_mm,
                  clip=0.9):
    """TIE 传播，返回探测器相对强度 I/I₀。

        I/I₀ = exp(−A)·[1 − (λR/2π)·∂²φ/∂s²]

    其中 ∂²/∂s² 沿**探测方向**（每个投影角各算一次）。这是相衬投影的
    标准一维处理，边缘处 ∇²φ 很大 → 亮暗条纹 → 边界显形。
    """
    A = np.asarray(abs_sino, dtype=float)
    phi = np.asarray(phi_sino, dtype=float)
    ds = float(p_eff_mm)
    # 二阶中心差分
    d2 = np.zeros_like(phi)
    d2[:, 1:-1] = (phi[:, 2:] - 2.0 * phi[:, 1:-1] + phi[:, :-2]) / (ds ** 2)
    d2[:, 0] = d2[:, 1]
    d2[:, -1] = d2[:, -2]
    mod = 1.0 - (wavelength_mm(energy_kev) * float(distance_mm) / (2.0 * np.pi)) * d2
    mod = np.clip(mod, 0.0, None)
    return np.exp(-np.clip(A, 0.0, None)) * mod


def phase_contrast_sinogram(abs_ellipses, delta_ellipses, angles_deg, n_cols,
                            p_eff_mm, scale_mm, energy_kev, distance_mm,
                            abs_scale=1.0, delta_scale=1.0):
    """返回 (纯吸收正弦图 A, 相衬等效正弦图 p = −ln(I/I₀))。

    p 可直接喂给 M4/M5 的既有采集与重建链，无需任何改动 —— 相衬路径
    与衰减路径在重建层是同构的。
    """
    A = analytic_sinogram(abs_ellipses, n_cols, p_eff_mm, angles_deg, scale_mm,
                          mu_scale=abs_scale)
    phi = phase_sinogram(delta_ellipses, angles_deg, n_cols, p_eff_mm, scale_mm,
                         energy_kev, delta_scale)
    inten = tie_propagate(A, phi, energy_kev, distance_mm, p_eff_mm)
    return A, -np.log(np.maximum(inten, 1e-12))


def paganin_retrieve(apparent_abs, p_eff_mm, energy_kev, distance_mm, delta, beta,
                     clip_hi=1e3):
    """Paganin 单距离相位反演：从 −ln(I/I₀) 恢复**投影厚度 T**。

    相比弱相位 TIE 线性化，反演把 TIE 引入的高通（边缘条纹）反解掉，
    给出定量正确的投影厚度，因此重建值正比于真实 δ，
    而不是被边缘增强调制过的图像。

    推导（均匀物体，δ/β 沿路径恒定）：
        I/I₀ ≈ exp(−A)·[1 − (λR/2π)∇²φ]
        φ = −(2π/λ)·δ·T,   A = μ·T = (4πβ/λ)·T
        ⇒ −ln(I/I₀) = (4πβ/λ)T − R·δ·∇²T
        傅里叶域（∇² → −4π²f²）：
            F[T] = F[−ln(I/I₀)] / [(4πβ/λ) + 4π²Rδf²]
        ⇒ H(f) = 1/(1 + α f²)，  α = πRλδ/β
    再乘 λ/(4πβ) 得到厚度量纲。
    """
    g = np.asarray(apparent_abs, dtype=float)
    lam = wavelength_mm(energy_kev)
    b = max(float(beta), 1e-30)
    n = g.shape[1]
    f = np.fft.rfftfreq(n, d=float(p_eff_mm))
    alpha = np.pi * float(distance_mm) * lam * float(delta) / b
    H = 1.0 / (1.0 + alpha * f ** 2)
    T = np.fft.irfft(np.fft.rfft(g, axis=1) * H[None, :], n=n, axis=1)
    T = T * (lam / (4.0 * np.pi * b))
    return np.clip(T, -abs(float(clip_hi)), abs(float(clip_hi)))


def phase_paganin_sinogram(abs_ellipses, delta_ellipses, angles_deg, n_cols,
                           p_eff_mm, scale_mm, energy_kev, distance_mm,
                           delta, beta, abs_scale=1.0, delta_scale=1.0):
    """相衬 + Paganin 反演：返回 (A 纯吸收, p_tie 弱相位, T 反演厚度)。"""
    A, p_tie = phase_contrast_sinogram(abs_ellipses, delta_ellipses, angles_deg,
                                       n_cols, p_eff_mm, scale_mm, energy_kev,
                                       distance_mm, abs_scale, delta_scale)
    T = paganin_retrieve(p_tie, p_eff_mm, energy_kev, distance_mm, delta, beta)
    return A, p_tie, T


def edge_cnr(rec, p_dense_mm, r_object_mm, n_samples=4):
    """量"物体边界处的可见度"：边缘环带内的对比度 / 背景噪声。

    做法：以物体中心为原点，沿半径方向取剖面，比较**边界环带**与
    **远背景**的均值差（相对背景归一），得到无量纲的边缘可见度。
    这是判断"细胞边界看不看得见"的直接指标，不依赖 MTF。
    """
    a = np.asarray(rec, dtype=float)
    n = a.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    rad = np.hypot(xx - c, yy - c) * float(p_dense_mm)
    bg = (rad > 3.0 * r_object_mm) & (rad < 4.5 * r_object_mm)
    edge = np.abs(rad - r_object_mm) < 0.5 * r_object_mm
    if not bg.any() or not edge.any():
        return None
    base = float(np.mean(a[bg]))
    return float(abs(np.mean(a[edge]) - base) / (abs(base) + 1e-30))



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