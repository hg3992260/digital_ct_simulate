# -*- coding: utf-8 -*-
"""真实几何 CT 物理引擎（LEAP-CT 后端）。

引擎能力（全部来自 mar 环境的 leapctype 1.26 / LivermorE AI Projector）：
  * 弯曲探测器（等角）扇束/锥束几何：set_conebeam + set_curvedDetector
  * 正投影（真实线积分）、FBP 解析重建、加权反投影：project / FBP / weightedBackproject
  * 滤波分离调用：preRampFiltering → rampFilterProjections → postRampFiltering → weightedBackproject
  * ramp 阶数 set_rampFilter(0/2/…/12) 与低通 set_FBPlowpass(FWHM px)
  * GPU（set_GPU(i)）与 CPU（set_GPU(-1)）

本模块自己实现的部分：
  * 扇束等角 → 平行束「插值展平」重排（θ = β + γ, t = -R·sinγ）
  * 短扫描 Parker 加权
  * 命名滤波核窗（Shepp-Logan / Cosine / Hann / Hamming / Butterworth）作为 ramp 之上的附加窗
  * 真实 sinogram 文件读取（.npy/.npz/.csv/.txt/.tif）

数组约定：
  正弦图对外统一为 (n_ch, n_views)（显示方便）；LEAP 内部为 (numAngles, numRows, numCols)。
  体数据 f 为 (numZ, numY, numX)，与 ph[y, x] 图像坐标一致。
"""

import os
from collections import OrderedDict

import numpy as np

try:
    from leapctype import tomographicModels
    LEAP_AVAILABLE = True
    LEAP_IMPORT_ERROR = ''
except Exception as exc:            # pragma: no cover
    tomographicModels = None
    LEAP_AVAILABLE = False
    LEAP_IMPORT_ERROR = f'{type(exc).__name__}: {exc}'


# =====================================================================
# 几何：由 ct_geometry 的结果导出"物理可测"参数
# =====================================================================

class FanGeometry:
    """扇形束（弧形等角探测器）真实几何参数。"""

    def __init__(self, params, results, system='A', n_views=None):
        self.system = system
        p = params
        r = results
        self.R = float(p['RA'] if system == 'A' else p['RB'])
        self.FDD = float(p['FDD'])
        self.p_iso = max(0.001, float(p.get('pixel_xy', 0.625)))
        self.p_iso_z = max(0.001, float(p.get('pixel_z', 0.625)))
        self.mag = self.FDD / self.R if self.R else 1.0
        self.p_det = self.p_iso * self.mag                   # 探测器实尺像元
        self.sfov = float(p['SFOV_A'] if system == 'A' else p['SFOV_B'])
        self.z_cov = float(p['Z_coverage'])
        self.beta = float(r['beta_A'] if system == 'A' else r['beta_B'])   # 扇形半角 (rad)
        self.fan_deg = float(np.rad2deg(2.0 * self.beta))                  # 开扇角
        self.arc = self.FDD * 2.0 * self.beta                              # 弧长
        self.n_ch = max(8, int(round(self.arc / self.p_det)))              # 通道数
        self.d_gamma = 2.0 * self.beta / self.n_ch                         # 开扇角单元 (rad)
        self.d_gamma_deg = float(np.rad2deg(self.d_gamma))
        self.n_rows = max(1, int(round(self.z_cov / self.p_iso_z)))
        self.rot_time = float(p['rotation_time'])
        self.rate = float(p.get('sampling_rate', 2000.0))
        self.rpm = 60.0 / self.rot_time if self.rot_time else 0.0
        # views 个数：采样率 × 转速（每圈）
        self.n_views = int(n_views) if n_views else max(8, int(round(self.rate * self.rot_time)))
        # 探测器高度（物理）
        self.h_det = self.FDD * self.z_cov / self.R
        self.gammas = (np.arange(self.n_ch) - (self.n_ch - 1) / 2.0) * self.d_gamma  # (n_ch,)

        # ---- Bowtie 滤过器：真正照射视野与透射率（由几何内核给出）----
        r = results or {}
        # 探测器阵列：采用几何内核"整数化自动匹配"后的值，保证
        # 阵列总数 = n_ch × n_rows 与 LEAP 的 numCols/numRows 及像元尺寸严格一致
        if 'n_ch' in r:
            self.n_ch = int(r['n_ch'])
            self.n_rows = int(r['n_rows'])
            self.p_det = float(r.get('col_pitch', self.p_det))
            self.row_pitch_det = float(r.get('row_pitch_det', self.p_det))
            self.row_pitch_iso = float(r.get('row_pitch_iso', self.p_iso_z))
            self.array_total = int(r.get('array_total', self.n_ch * self.n_rows))
            self.d_gamma = 2.0 * self.beta / self.n_ch
            self.d_gamma_deg = float(np.rad2deg(self.d_gamma))
            self.gammas = (np.arange(self.n_ch) - (self.n_ch - 1) / 2.0) * self.d_gamma
        else:
            self.row_pitch_det = self.p_iso_z * self.mag
            self.row_pitch_iso = self.p_iso_z
            self.array_total = self.n_ch * self.n_rows
        self.gamma_bt = float(r.get('gamma_bt', self.beta))
        self.sfov_bt = float(r.get('bowtie_sfov', self.sfov))
        self.sfov_bt_set = float(r.get('bowtie_sfov_set', self.sfov))
        self.bowtie_fan_deg = float(r.get('bowtie_fan_deg', self.fan_deg))
        self.bt_limits = bool(r.get('bowtie_limits', False))
        self.bt_t_edge = float(r.get('bt_t_edge', 0.0))
        self.bt_mu = float(r.get('bt_mu', 0.0196))
        self.bt_dose_ratio = float(r.get('bt_dose_ratio', 1.0))
        self.bt_T_edge = float(r.get('bt_T_edge', 1.0))

    # ---------- Bowtie ----------
    def bowtie_transmission(self):
        """逐通道透射率 T(γ)：中心 1 → 边缘 exp(-μ·t_edge)（抛物型厚度剖面）。"""
        u = np.clip(np.abs(self.gammas) / max(self.gamma_bt, 1e-9), 0.0, 1.0)
        return np.exp(-self.bt_mu * self.bt_t_edge * u ** 2).astype(np.float32)

    def bowtie_mask(self):
        """被 bowtie 照亮的通道掩膜：|γ| > γ_bt 的射线被滤过体/准直挡住 → 无信号（截断）。"""
        return (np.abs(self.gammas) <= self.gamma_bt + 1e-12).astype(np.float32)

    def angles(self, mode='full360'):
        """返回角度数组（度）。full360: 0..360；short: 180+开扇角（短扫描）"""
        if mode == 'short':
            span = 180.0 + self.fan_deg
            return np.linspace(0.0, span, self.n_views, endpoint=False).astype(np.float32)
        return np.linspace(0.0, 360.0, self.n_views, endpoint=False).astype(np.float32)

    def adapt_to_data(self, n_ch, n_views, meta=None):
        """按外部真实数据的通道/视角数派生几何（保持 R/FDD/开扇角；元数据可覆盖）。"""
        import copy as _copy
        o = _copy.copy(self)
        o.n_ch = max(8, int(n_ch))
        o.n_views = max(1, int(n_views))
        if meta:
            conv = {'fan_angle_deg': ('beta', lambda v: float(np.deg2rad(float(v) / 2.0))),
                    'source_iso_mm': ('R', float),
                    'source_detector_mm': ('FDD', float),
                    'rotation_time_s': ('rot_time', float),
                    'sampling_rate_hz': ('rate', float),
                    'pixel_iso_mm': ('p_iso', float)}
            for k, (attr, fn) in conv.items():
                if k in meta:
                    try:
                        setattr(o, attr, fn(meta[k]))
                    except Exception:
                        pass
        o.mag = o.FDD / o.R if o.R else 1.0
        o.fan_deg = float(np.rad2deg(2.0 * o.beta))
        o.arc = o.FDD * 2.0 * o.beta
        o.d_gamma = 2.0 * o.beta / o.n_ch
        o.d_gamma_deg = float(np.rad2deg(o.d_gamma))
        o.p_det = o.arc / o.n_ch
        o.p_iso = o.p_det * o.R / o.FDD if o.FDD else o.p_det
        o.h_det = o.FDD * o.z_cov / o.R if o.R else o.z_cov
        o.gammas = (np.arange(o.n_ch) - (o.n_ch - 1) / 2.0) * o.d_gamma
        o.rpm = 60.0 / o.rot_time if o.rot_time else 0.0
        return o

    def report(self, extra=None):
        """用于界面显示的参数表。"""
        d = OrderedDict()
        d['系统'] = f'{self.system} 系统（R={self.R:.0f} mm, FDD={self.FDD:.0f} mm）'
        d['探测器'] = '弯曲（等角 / equiangular），LEAP set_curvedDetector'
        d['采样率'] = f'{self.rate:.0f} Hz'
        d['转速'] = f'{self.rot_time:.3f} s/rot  ({self.rpm:.0f} rpm)'
        d['views 个数'] = f'{self.n_views}  /圈'
        d['角步进'] = f'{360.0 / self.n_views:.4f} °/view'
        d['开扇角'] = f'{self.fan_deg:.3f} °  (半角 {self.fan_deg / 2:.3f}°)'
        d['开扇角单元'] = f'{self.d_gamma_deg:.5f} °/通道  ({np.rad2deg(self.d_gamma) * 60:.4f} 角分)'
        d['通道数'] = f'{self.n_ch}'
        d['物理像素(ISO)'] = f'{self.p_iso:.3f} mm'
        d['探测器像元(实尺)'] = f'{self.p_det:.3f} mm  (放大比 {self.mag:.4f})'
        d['探测器高度/排数'] = f'{self.h_det:.1f} mm / {self.n_rows} 排'
        d['探测器阵列'] = (f'{self.n_ch} 个/排 × {self.n_rows} 排 = {self.array_total} 单元'
                       f'  (匹配后列距 {self.p_det:.4f} / 行距 {self.row_pitch_det:.4f} mm)')
        d['Bowtie 视野(设定)'] = f'{self.sfov_bt_set:.0f} mm'
        d['真正视野 SFOV'] = (f'{self.sfov_bt:.1f} mm  (开扇角 {self.bowtie_fan_deg:.2f}°)'
                          + ('  ← 受 Bowtie 限制' if self.bt_limits else ''))
        d['Bowtie 截断'] = ('是（|γ| > %.3f° 无信号，FBP 将出现截断伪影）' % np.rad2deg(self.gamma_bt)
                         if self.bt_limits else '否（探测器为限制方）')
        d['Bowtie 透射'] = f'中心 {1.0:.3f} / 边缘 {self.bt_T_edge:.3f}  (μ={self.bt_mu:.4f}/mm, t_edge={self.bt_t_edge:.0f} mm)'
        d['剂量比(边缘/中心)'] = f'{self.bt_dose_ratio:.3f}  (边缘节省 {100 * (1 - self.bt_dose_ratio):.0f}%)'
        d['弧长'] = f'{self.arc:.1f} mm'
        d['SFOV'] = f'{self.sfov:.0f} mm'
        if extra:
            d.update(extra)
        return d


def parker_weight(geom, angles_deg):
    """短扫描 Parker 冗余加权，返回 (n_ch, n_views) 权重。"""
    g = geom.gammas[None, :]                       # (1, n_ch) rad
    b = np.deg2rad(np.asarray(angles_deg, dtype=float))[:, None]   # (n_views, 1) rad
    gm = geom.beta
    w = np.ones_like(b * np.ones_like(g))
    lo = 2.0 * gm - 2.0 * g
    hi = np.pi - 2.0 * g
    m1 = b < lo
    m2 = b > hi
    with np.errstate(invalid='ignore', divide='ignore'):
        w1 = np.sin(np.pi / 4.0 * b / (gm - g)) ** 2
        w2 = np.sin(np.pi / 4.0 * (np.pi + 2.0 * gm - b) / (gm + g)) ** 2
    w = np.where(m1, w1, w)
    w = np.where(m2, w2, w)
    w = np.where((~m1) & (~m2), 1.0, w)
    return np.nan_to_num(w, nan=0.0).T.astype(np.float32)          # -> (n_ch, n_views)


# =====================================================================
# 滤波核（作为 LEAP ramp 之上的附加频率窗）
# =====================================================================

KERNELS = ['Ram-Lak', 'Shepp-Logan', 'Cosine', 'Hann', 'Hamming', 'Butterworth', '无滤波']


def window_response(name, n, d=1.0):
    """返回长度为 n 的频率窗（对 Nyquist 归一化，1.0 = 无衰减）。"""
    f = np.fft.fftfreq(n, d=d)                     # cycles/unit，|f| <= 0.5/d
    fny = 0.5 / d
    u = np.abs(f) / fny                            # 0..1
    u = np.clip(u, 0.0, 1.0)
    if name == 'Ram-Lak':
        return np.ones_like(u)
    if name == 'Shepp-Logan':
        x = np.pi * u
        return np.where(u < 1e-9, 1.0, np.sin(x) / np.where(x < 1e-9, 1.0, x))
    if name == 'Cosine':
        return np.cos(np.pi / 2.0 * u)
    if name == 'Hann':
        return 0.5 + 0.5 * np.cos(np.pi * u)
    if name == 'Hamming':
        return 0.54 + 0.46 * np.cos(np.pi * u)
    if name == 'Butterworth':
        order, cut = 4, 0.85
        return 1.0 / np.sqrt(1.0 + (u / cut) ** (2 * order))
    return np.ones_like(u)


def apply_window(sino_cv, name):
    """沿通道轴（axis 0）施加频率窗；输入/输出 (n_ch, n_views)。"""
    if name in (None, '', 'Ram-Lak'):
        return sino_cv
    n = sino_cv.shape[0]
    n2 = int(2 ** np.ceil(np.log2(max(n * 2, 16))))
    pad = np.zeros((n2, sino_cv.shape[1]), dtype=np.float32)
    pad[:n] = sino_cv
    W = window_response(name, n2).astype(np.float32)[:, None]
    out = np.real(np.fft.ifft(np.fft.fft(pad, axis=0) * W, axis=0)).astype(np.float32)
    return out[:n]


# =====================================================================
# 引擎
# =====================================================================

class LeapEngine:
    """LEAP-CT 封装：几何设置 / 正投影 / FBP / 展平重排。"""

    def __init__(self, gpu_index=0):
        if not LEAP_AVAILABLE:
            raise RuntimeError(f'LEAP-CT 不可用：{LEAP_IMPORT_ERROR}')
        self.ct = tomographicModels()
        for lvl in ('set_log_error', 'set_log_warning'):
            try:
                getattr(self.ct, lvl)(False)
            except Exception:
                pass
        self.gpu_index = gpu_index
        self.set_gpu(gpu_index)
        self.last_timing = {}

    # ---------- 设备 ----------
    def set_gpu(self, idx):
        self.gpu_index = idx
        try:
            self.ct.set_GPU(int(idx))          # -1 = CPU
            return True
        except Exception:
            return False

    def device_label(self):
        if self.gpu_index is None or self.gpu_index < 0:
            return 'CPU'
        try:
            return f'GPU {self.gpu_index}'
        except Exception:
            return f'GPU {self.gpu_index}'

    # ---------- 几何 ----------
    def _set_fan(self, geom, angles_deg, n_rows=1):
        ct = self.ct
        ct.set_conebeam(len(angles_deg), n_rows, geom.n_ch,
                        geom.p_det, geom.p_det, 0.0, 0.5 * (geom.n_ch - 1),
                        np.ascontiguousarray(angles_deg, dtype=np.float32),
                        geom.R, geom.FDD)
        try:
            ct.set_curvedDetector()
        except Exception:
            pass

    def _set_volume(self, n):
        self.ct.set_volume(int(n), int(n), 1, 0.0, 0.0)   # voxel 稍后用 set_volume 覆盖

    def configure_volume(self, n, voxel_mm):
        self.ct.set_volume(int(n), int(n), 1, float(voxel_mm), float(voxel_mm))

    # ---------- 正投影（生成真实 sinogram） ----------
    def project_fan(self, phantom2d, geom, angles_deg, mode='full360'):
        import time
        if mode == 'short':
            angles_deg = geom.angles('short')
        ct = self.ct
        n = int(phantom2d.shape[0])
        self._set_fan(geom, angles_deg)
        self.configure_volume(n, geom.sfov / n)
        f = ct.allocateVolume()
        f[0] = np.ascontiguousarray(phantom2d, dtype=np.float32)
        g = ct.allocateProjections()
        t0 = time.perf_counter()
        ct.project(g, f)
        self.last_timing['project'] = time.perf_counter() - t0
        return np.ascontiguousarray(np.asarray(g)[:, 0, :].T)      # -> (n_ch, n_views)

    # ---------- 静态多源：LEAP 模块束（任意源/探测器位形）----------
    def set_modular_static(self, geom, n_src=24, n_rows=1, row_pitch_mm=None):
        """静态多源机架：n_src 个源均布于 R 圆上，每个源对侧 FDD 处一个探测器模块。

        LEAP 的 set_modularbeam 允许逐视角指定源位置 / 模块中心 / 行列方向向量，
        因此无需把静态采集"伪装"成旋转采集——这正是静态 CT 的正确几何入口。
        """
        ct = self.ct
        n_src = int(n_src)
        th = (np.arange(n_src, dtype=np.float32) * (2.0 * np.pi / n_src))
        cz, sz = np.cos(th), np.sin(th)
        zz = np.zeros(n_src, np.float32)
        src = np.column_stack([geom.R * cz, geom.R * sz, zz]).astype(np.float32)
        ctr = np.column_stack([(geom.R - geom.FDD) * cz, (geom.R - geom.FDD) * sz,
                               zz]).astype(np.float32)
        rowv = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (n_src, 1))
        colv = np.column_stack([-sz, cz, zz]).astype(np.float32)   # 与射线方向正交
        rp = float(row_pitch_mm or (geom.p_iso_z * geom.FDD / geom.R))
        ct.set_modularbeam(n_src, int(n_rows), geom.n_ch, rp, float(geom.p_det),
                           src, ctr, rowv, colv)

    def project_modular(self, vol3d, geom, n_src=24, n_rows=1, row_pitch_mm=None):
        """模块束正投影 → (n_src, n_rows, n_ch)。"""
        import time
        ct = self.ct
        nz, ny, nx = vol3d.shape
        self.set_modular_static(geom, n_src, n_rows, row_pitch_mm)
        ct.set_volume(nx, ny, nz, geom.sfov / nx, float(getattr(geom, 'row_pitch_iso', geom.p_iso_z)))
        f = ct.allocateVolume()
        np.copyto(np.asarray(f), np.ascontiguousarray(vol3d, dtype=np.float32))
        g = ct.allocateProjections()
        t0 = time.perf_counter()
        ct.project(g, f)
        self.last_timing['project'] = time.perf_counter() - t0
        return np.asarray(g)

    def recon_modular(self, g, geom, nx, nz=1, n_src=24, n_rows=1, row_pitch_mm=None,
                      method='SART', n_iter=40, n_subsets=1, num_tv=1):
        """稀疏视角迭代重建：SART / ASDPOCS(TV) / MLEM。返回 (nz, ny, nx)。"""
        import time
        ct = self.ct
        self.set_modular_static(geom, n_src, n_rows, row_pitch_mm)
        ct.set_volume(int(nx), int(nx), int(nz), geom.sfov / nx, float(getattr(geom, 'row_pitch_iso', geom.p_iso_z)))
        gg = ct.allocateProjections()
        np.copyto(np.asarray(gg), np.ascontiguousarray(g, dtype=np.float32))
        out = ct.allocateVolume()
        t0 = time.perf_counter()
        if method == 'ASDPOCS':
            # 注意：第 5 个参数是 numTV（TV 步数，整数），不是权重
            ct.ASDPOCS(gg, out, int(n_iter), int(n_subsets), int(num_tv))
        elif method == 'MLEM':
            ct.MLEM(gg, out, int(n_iter))
        elif method == 'OSEM':
            ct.OSEM(gg, out, int(n_iter), int(n_subsets))
        else:
            ct.SART(gg, out, int(n_iter), int(n_subsets))
        self.last_timing['fbp'] = time.perf_counter() - t0
        return np.asarray(out)

    # ---------- 原生锥束螺旋（3D 体数据 + helicalPitch） ----------
    def configure_cone_helical(self, geom, phis, n_rows, row_pitch_mm, pitch):
        """numRows=n_rows 的锥束几何 + LEAP 原生螺旋螺距。

        LEAP 的 helicalPitch 取**每圈进床距离(mm)** = pitch × Z轴覆盖；
        螺旋重建由 LEAP 内部做锥束加权（FDK 类），无需任何行平均/共轭重排。

        pitch=0 → 纯轴向锥束（旋转架构的"阵列完整"模式用这条）。
        """
        ct = self.ct
        ct.set_conebeam(len(phis), int(n_rows), geom.n_ch,
                        float(row_pitch_mm), float(geom.p_det),
                        0.5 * (n_rows - 1), 0.5 * (geom.n_ch - 1),
                        np.ascontiguousarray(phis, dtype=np.float32),
                        # 注意：LEAP 的 helicalPitch 单位是 **mm/弧度**（见 leapctype 文档），
                        # 不是每圈进床 mm。每圈进床 = pitch×Z覆盖 → 需除以 2π 转成 mm/rad，
                        # 否则实际进床会大 2π≈6.28 倍（这正是此前螺旋相关只有 0.66 的根因）。
                        geom.R, geom.FDD, 0.0,
                        float(pitch) * geom.z_cov / (2.0 * np.pi))
        try:
            ct.set_curvedDetector()
        except Exception:
            pass

    def cone_params(self, geom):
        """由 **Z 轴范围 与 Z 向宽度** 推导锥束探测器参数（几何内核已整数化匹配）。

            numRows     = n_rows        = Z 轴覆盖 / ISO 像素Z
            pixelHeight = row_pitch_det = (Z 轴覆盖 / n_rows) × 放大比
            numCols     = n_ch          = 弧长 / 实尺列距
            pixelWidth  = p_det         = 弧长 / n_ch
            voxelHeight = row_pitch_iso = Z 轴覆盖 / n_rows   ← 体数据 Z 向体素
        """
        g = geom
        return {'numRows': int(g.n_rows), 'numCols': int(g.n_ch),
                'pixelHeight': float(getattr(g, 'row_pitch_det', g.p_det)),
                'pixelWidth': float(g.p_det),
                'voxelHeight': float(getattr(g, 'row_pitch_iso', g.p_iso_z)),
                'h_det': float(g.n_rows * getattr(g, 'row_pitch_det', g.p_det)),
                'z_cov': float(g.z_cov), 'nz_full': int(g.n_rows)}

    def project_cone(self, vol3d, geom, phis, n_rows, row_pitch_mm, pitch):
        """3D 体数据 → 锥束螺旋投影，返回 (n_angles, n_rows, n_ch)。"""
        import time
        ct = self.ct
        nz, ny, nx = vol3d.shape
        self.configure_cone_helical(geom, phis, n_rows, row_pitch_mm, pitch)
        ct.set_volume(nx, ny, nz, geom.sfov / nx, float(getattr(geom, 'row_pitch_iso', geom.p_iso_z)))
        f = ct.allocateVolume()
        np.copyto(np.asarray(f), np.ascontiguousarray(vol3d, dtype=np.float32))
        g = ct.allocateProjections()
        t0 = time.perf_counter()
        ct.project(g, f)
        self.last_timing['project'] = time.perf_counter() - t0
        return np.asarray(g)

    def fbp_cone_helical(self, g, geom, phis, n_rows, row_pitch_mm, pitch, nx, nz):
        """LEAP 原生锥束螺旋 FBP，返回 (nz, ny, nx)。"""
        import time
        ct = self.ct
        self.configure_cone_helical(geom, phis, n_rows, row_pitch_mm, pitch)
        ct.set_volume(int(nx), int(nx), int(nz), geom.sfov / nx,
                      float(getattr(geom, 'row_pitch_iso', geom.p_iso_z)))
        gg = ct.allocateProjections()
        np.copyto(np.asarray(gg), np.ascontiguousarray(g, dtype=np.float32))
        out = ct.allocateVolume()
        t0 = time.perf_counter()
        ct.FBP(gg, out)
        self.last_timing['fbp'] = time.perf_counter() - t0
        return np.asarray(out)

    # ---------- 直接扇形束 FBP ----------
    def fbp_fan(self, sino_cv, geom, angles_deg, kernel='Ram-Lak', lowpass=1.0,
                vol_n=None, voxel_mm=None, parker=False, manual=True):
        """直接在扇束几何上做 FBP。

        重要：LEAP 的 FBP() 对**非 360° 角度区间**会额外加权，实测短扫描只有 0.9355；
        而它自己文档给出的等价手动链
            preRampFiltering → rampFilterProjections(get_FBPscalar) → postRampFiltering
            → weightedBackproject
        在同样数据上可达 0.9996，因此默认走手动链（manual=True）。
        """
        import time
        ct = self.ct
        if vol_n is None:
            vol_n = geom.n_ch
        if voxel_mm is None:
            voxel_mm = geom.sfov / vol_n
        g_np = sino_cv.T.astype(np.float32).copy()                 # (views, ch)
        is_short = len(angles_deg) > 1 and (float(angles_deg[-1]) - float(angles_deg[0])) < 359.0
        if parker and is_short:
            g_np *= parker_weight(geom, angles_deg).T              # (views, ch)
        self._set_fan(geom, angles_deg)
        self.configure_volume(vol_n, voxel_mm)
        g = ct.allocateProjections()
        np.copyto(np.asarray(g), g_np[:, None, :])
        t0 = time.perf_counter()
        ct.set_rampFilter(12)
        if lowpass and lowpass >= 2.0:
            ct.set_FBPlowpass(float(lowpass))
        out = ct.allocateVolume()
        if kernel == '无滤波':
            ct.weightedBackproject(g, out)
        elif kernel == 'Ram-Lak' and not manual:
            ct.FBP(g, out)
        else:
            ct.preRampFiltering(g)
            ct.rampFilterProjections(g, ct.get_FBPscalar())
            ct.postRampFiltering(g)
            if kernel not in ('Ram-Lak', '无滤波'):
                win = np.ascontiguousarray(apply_window(np.asarray(g)[:, 0, :].T, kernel).T)
                np.copyto(np.asarray(g), win[:, None, :])
            ct.weightedBackproject(g, out)
        self.last_timing['fbp'] = time.perf_counter() - t0
        return np.asarray(out)[0].astype(np.float32)

    # ---------- 滤波后正弦图（仅用于显示） ----------
    def filter_only(self, sino_cv, geom, angles_deg, kernel='Ram-Lak', lowpass=1.0,
                    vol_n=None, voxel_mm=None):
        ct = self.ct
        if vol_n is None:
            vol_n = geom.n_ch
        if voxel_mm is None:
            voxel_mm = geom.sfov / vol_n
        self._set_fan(geom, angles_deg)
        self.configure_volume(vol_n, voxel_mm)          # ramp 滤波依赖已定义的体数据
        g = ct.allocateProjections()
        np.copyto(np.asarray(g), sino_cv.T.astype(np.float32)[:, None, :])
        ct.set_rampFilter(12)
        if lowpass and lowpass >= 2.0:
            ct.set_FBPlowpass(float(lowpass))
        ct.preRampFiltering(g)
        ct.rampFilterProjections(g, ct.get_FBPscalar())
        ct.postRampFiltering(g)
        arr = np.asarray(g)[:, 0, :].T.copy()
        if kernel not in ('Ram-Lak', '无滤波'):
            arr = apply_window(arr, kernel)
        return arr.astype(np.float32)

    # ---------- 插值展平：扇束(等角) → 平行束 ----------
    def rebin_fan_to_parallel(self, sino_cv, geom, angles_deg, theta_deg=None, oversample=4):
        """θ = β + γ, t = -R·sinγ；双线性插值。返回 (sino_parallel(n_t,n_θ), theta, t)

        oversample：输出 t 网格相对 ISO 像素的过采样倍率。线性插值会在重排数据里留下
        折角（二阶不连续），ramp 滤波会把它放大成条纹，因此默认过采样 4 倍。
        """
        beta = np.deg2rad(np.asarray(angles_deg, dtype=float))
        full = (beta[-1] - beta[0]) >= np.deg2rad(359.0) if len(beta) > 1 else False
        t_max = geom.R * np.sin(geom.beta)                       # = SFOV/2
        d_t = geom.p_iso / float(max(1, oversample))
        n_t = int(2 * t_max / d_t)
        t = (np.arange(n_t) - (n_t - 1) / 2.0) * d_t
        if theta_deg is None:
            span = 180.0 if full else 180.0 + geom.fan_deg
            n_th = max(16, int(round(span / (360.0 / geom.n_views))))
            theta = np.linspace(0.0, span, n_th, endpoint=False)
        else:
            theta = np.asarray(theta_deg, dtype=float)
        th = np.deg2rad(theta)[None, :]                          # (1, n_θ)
        tt = t[:, None]                                          # (n_t, 1)
        s = np.clip(tt / geom.R, -1.0, 1.0)
        gamma = -np.arcsin(s)                                    # (n_t, 1) rad
        b = th - gamma                                           # (n_t, n_θ) rad
        # 通道 / 视角索引
        c = gamma / geom.d_gamma + (geom.n_ch - 1) / 2.0
        b_span = beta[-1] - beta[0] if len(beta) > 1 else 2 * np.pi
        if full:
            b_rel = np.mod(b - beta[0], 2 * np.pi)
        else:
            b_rel = b - beta[0]
        v = b_rel / (b_span / max(1, len(beta) - 1)) if len(beta) > 1 else b_rel * 0

        out = np.zeros((n_t, theta.shape[0]), dtype=np.float32)
        valid = (c >= 0) & (c <= geom.n_ch - 1) & (v >= 0) & (v <= len(beta) - 1)
        c0 = np.clip(np.floor(c).astype(int), 0, geom.n_ch - 2)
        v0 = np.clip(np.floor(v).astype(int), 0, max(0, len(beta) - 2))
        wc = np.clip(c - c0, 0, 1)
        wv = np.clip(v - v0, 0, 1)
        src = sino_cv
        out = ((1 - wc) * (1 - wv) * src[c0, v0] + wc * (1 - wv) * src[c0 + 1, v0]
               + (1 - wc) * wv * src[c0, v0 + 1] + wc * wv * src[c0 + 1, v0 + 1])
        out = np.where(valid, out, 0.0).astype(np.float32)
        return out, theta, t, d_t

    # ---------- 平行束 FBP（作用于展平后的数据） ----------
    def fbp_parallel(self, sino_tth, theta_deg, d_t, voxel_mm, vol_n,
                     kernel='Ram-Lak', lowpass=1.0):
        import time
        ct = self.ct
        n_t = sino_tth.shape[0]
        ct.set_parallelbeam(len(theta_deg), 1, n_t, float(d_t), float(d_t),
                            0.0, 0.5 * (n_t - 1),
                            np.ascontiguousarray(theta_deg, dtype=np.float32))
        # LEAP 对平行束/扇束要求：体素高度必须等于探测器像素高度
        ct.set_volume(int(vol_n), int(vol_n), 1, float(voxel_mm), float(d_t))
        g = ct.allocateProjections()
        np.copyto(np.asarray(g), sino_tth.T.astype(np.float32)[:, None, :])
        t0 = time.perf_counter()
        ct.set_rampFilter(12)
        if lowpass and lowpass >= 2.0:
            ct.set_FBPlowpass(float(lowpass))
        out = ct.allocateVolume()
        if kernel == '无滤波':
            ct.weightedBackproject(g, out)
        elif kernel == 'Ram-Lak':
            ct.FBP(g, out)
        else:
            ct.preRampFiltering(g)
            ct.rampFilterProjections(g, ct.get_FBPscalar())
            ct.postRampFiltering(g)
            win = np.ascontiguousarray(apply_window(np.asarray(g)[:, 0, :].T, kernel).T)
            np.copyto(np.asarray(g), win[:, None, :])
            ct.weightedBackproject(g, out)
        self.last_timing['fbp'] = time.perf_counter() - t0
        return np.asarray(out)[0].astype(np.float32)


# =====================================================================
# 真实 sinogram 文件读取
# =====================================================================

def _orient_frame(arr, meta=None):
    """把单帧数据统一成 (n_ch, n_views)。"""
    a = np.asarray(arr, dtype=np.float32)
    while a.ndim > 2:
        a = a[..., 0]
    layout = str((meta or {}).get('layout', '')).lower()
    if layout in ('cv', 'channels_views'):
        pass
    elif layout in ('vc', 'views_channels'):
        a = a.T
    elif a.shape[0] < a.shape[1]:
        a = a.T                     # 行少于列 → 通常是 (views, channels)，转成 (channels, views)
    return np.ascontiguousarray(a, dtype=np.float32)


def _load_raw_array(path):
    """读取原始数组（不降维），返回 (ndarray, meta)。"""
    p = str(path)
    low = p.lower()
    meta = {}
    if low.endswith('.npz'):
        z = np.load(p, allow_pickle=True)
        key = 'sinogram' if 'sinogram' in z else ('data' if 'data' in z else list(z.keys())[0])
        arr = np.asarray(z[key])
        for k in z.keys():
            if k == key:
                continue
            try:
                meta[k] = z[k].item() if np.asarray(z[k]).ndim == 0 else np.asarray(z[k]).tolist()
            except Exception:
                pass
    elif low.endswith('.npy'):
        arr = np.asarray(np.load(p, allow_pickle=True))
    elif low.endswith(('.tif', '.tiff', '.png')):
        import imageio.v2 as imageio
        arr = np.asarray(imageio.imread(p))
    else:
        arr = np.asarray(np.loadtxt(p, delimiter=',' if low.endswith('.csv') else None))
    return np.asarray(arr, dtype=np.float32), meta


def load_sinogram_file(path):
    """读取单帧 sinogram，返回 (data(n_ch,n_views) float32, meta)。

    约定：LEAP/多数采集系统导出为 (views, channels)，其余按较短轴为 views 判断；
    npz 中若含 'sinogram'/'data' 键与几何元数据则一并采纳（见 load_sinogram_series 说明）。
    """
    arr, meta = _load_raw_array(path)
    out = _orient_frame(arr, meta)
    meta['shape_after'] = out.shape
    return out, meta


def load_sinogram_series(path, max_frames=512):
    """读取**批量目录**或**动态序列**，返回 (frames, labels, meta)。

    * path 为目录：按文件名排序读取其中所有 .npy/.npz/.csv/.txt/.tif，每文件一帧
    * path 为文件且为 3D (F, ·, ·)：按第一维拆帧（动态序列 / 时间序列）
    * 其余情况视为单帧
    """
    import glob
    p = str(path)
    frames, labels, meta = [], [], {}
    if os.path.isdir(p):
        exts = ('.npy', '.npz', '.csv', '.txt', '.tif', '.tiff')
        files = sorted(f for f in glob.glob(os.path.join(p, '*')) if f.lower().endswith(exts))
        if not files:
            raise ValueError('目录中没有可识别的 sinogram 文件')
        for f in files[:max_frames]:
            raw, m = _load_raw_array(f)
            if raw.ndim == 3:                       # 目录里也可能混有多帧文件
                for i in range(min(raw.shape[0], max_frames - len(frames))):
                    frames.append(_orient_frame(raw[i], m))
                    labels.append(f'{os.path.basename(f)}#{i + 1}')
            elif raw.ndim == 2:
                frames.append(_orient_frame(raw, m))
                labels.append(os.path.basename(f))
            else:
                continue
            for k, v in m.items():
                meta.setdefault(k, v)
        meta['series_kind'] = '批量目录'
    else:
        arr, meta = _load_raw_array(p)
        if arr.ndim == 3:
            n = min(arr.shape[0], max_frames)
            for i in range(n):
                frames.append(_orient_frame(arr[i], meta))
                labels.append(f'frame {i + 1}')
            meta['series_kind'] = '动态序列 (3D 数组)'
        else:
            frames.append(_orient_frame(arr, meta))
            labels.append(os.path.basename(p))
            meta['series_kind'] = '单帧'
    return frames, labels, meta

