# -*- coding: utf-8 -*-
"""双源 CT 锥束几何内核（纯 numpy，无任何 Qt 依赖）。

本模块由 simulate_qt.py 的 calculate_geometry() 抽取而来，供 PyQt5 / PySide6
两个前端共用，避免在同一进程内混用两套 Qt 绑定。

约定：
  ISO 为原点；系统 A 焦点位于 -alpha/2 方向、系统 B 位于 +alpha/2 方向；
  角度单位：入参 alpha 为度，返回值中 beta_* / kappa_* / alpha_rad 为弧度。
"""

import numpy as np


ARCH_TABLE = {
    # 架构 → (源数量, 能量维度[层/箱], 是否旋转, 中文标签)
    'dual_source': (2, 1, True, '双源 (Dual-Source, α 夹角)'),
    'single_wide': (1, 1, True, '单源宽体 (Single-Source Wide-Body)'),
    'dual_layer': (1, 2, True, '双层探测器 (Dual-Layer Spectral, 2 层)'),
    'pcct': (1, 8, True, '光子计数 (Photon-Counting, 8 能量箱)'),
    'static_multi': (24, 1, False, '静态多源 (Stationary 24-Source)'),
    # 第 6 种：同步辐射仿真模式，以**单元光子计数**架构改造。
    # 能量箱数不固定为 8，由 sync_bins 决定（见 calculate_geometry 中的覆盖逻辑）。
    # 物理与算法细节见 ct_synchrotron.py 及其引用的两份交接文档。
    'synchrotron': (1, 8, True, '同步辐射 (Synchrotron · 单元光子计数改造)'),
}


def calculate_geometry(alpha, RA, RB, FDD, SFOV_A, SFOV_B, Z_coverage, rotation_time, is_asymmetric=False, required_min_dist=150, required_arc_diff=120, bowtie_sfov=None, bowtie_edge_mm=30.0, scan_mode='axial', pitch=1.0, pixel_xy=0.625, pixel_z=0.625, n_ch_set=None, arch='dual_source', sampling_rate=2000.0, src_switch_us=200.0, scan_length=300.0, slice_thickness=1.0, slice_interval=1.0, shots_per_source=1, ring_sources=0,
                        # ---- 同步辐射仿真模式专用（arch='synchrotron' 时生效）----
                        # 硬件：源类型 / 焦点 / 探测器材料与像素 / 电荷云 / 整形时间
                        sync_source='metaljet', sync_focus_um=None, sync_pixel_um=55.0,
                        sync_sigma_c_um=None, sync_detector='CdTe', sync_shaping_ns=None,
                        sync_cols=2048, sync_rows=512,
                        # µCT 几何：同步辐射用**自己的** SOD/ODD（不去挤临床的 RA/FDD）
                        sync_sod_mm=100.0, sync_odd_mm=300.0,
                        # 采集：过采样帧数 / 能量箱 / 通量 / 标称能量
                        sync_oversample=4, sync_bins=8, sync_energy_kev=60.0, sync_flux=1e8,
                        # 算法：相衬 / 基底比值 / VMI 目标能量
                        sync_phase=False, sync_delta_beta=100.0, sync_vmi_kev=65.0,
                        # 目标：参考分辨率（现状基线）与目标分辨率
                        sync_ref_res_um=100.0, sync_target_res_um=15.0):
    """根据输入参数计算所有相关几何值和约束。

    几何约定：RA 即源-等中心距（SOD），FDD − RA 即等中心-探测器距（ODD），
    因此放大率 M = (SOD+ODD)/SOD = FDD/RA。同步辐射模式直接复用这一关系，
    把"样品推向源"表达为调小 RA / 调大 FDD。
    """
    
    # 转换为弧度
    alpha_rad = np.deg2rad(alpha)
    
    # 扇形半角 (Beta) - XY平面
    try:
        beta_A = np.arcsin((SFOV_A / 2) / RA)
        beta_B_orig = np.arcsin((SFOV_B / 2) / RB)
    except ValueError:
        return {"error": "SFOV/2 必须小于 F-ISO 距离 R"}, None, None

    # Handle Asymmetric Logic
    arc_B_orig = FDD * 2 * beta_B_orig
    arc_extension = 120.0 # mm
    
    if is_asymmetric:
        # Effective Arc (Virtual Left + Real Right)
        arc_B_effective = arc_B_orig + 2 * arc_extension
        # New Beta based on effective arc
        beta_B = (arc_B_effective / FDD) / 2
        # New SFOV B based on effective beta
        SFOV_B_effective = 2 * RB * np.sin(beta_B)
        
        # Inner SFOV (Double Sampled / Symmetric part)
        SFOV_B_inner = 2 * RB * np.sin(beta_B_orig)
        
        # Physical Mass Increase (Only Right side is real)
        # Assuming original mass M_det corresponds to arc_B_orig
        # Mass factor = (arc_B_orig + arc_extension) / arc_B_orig
        mass_factor_B = (arc_B_orig + arc_extension) / arc_B_orig
        
        # Physical SFOV B (For Scatter calculation)
        # Physical Arc = Original + Right Extension
        arc_B_physical = arc_B_orig + arc_extension
        beta_B_physical = (arc_B_physical / FDD) / 2
        SFOV_B_physical = 2 * RB * np.sin(beta_B_physical)
        
        # Asymmetric Sector Angle (in degrees) for temporal resolution
        angle_ext_rad = arc_extension / FDD
        angle_ext_deg = np.rad2deg(angle_ext_rad)
    else:
        beta_B = beta_B_orig
        SFOV_B_effective = SFOV_B
        SFOV_B_inner = SFOV_B
        mass_factor_B = 1.0
        SFOV_B_physical = SFOV_B
        angle_ext_deg = 0.0

    # 扇形全角
    fan_angle_A = np.rad2deg(2 * beta_A)
    fan_angle_B = np.rad2deg(2 * beta_B)
    
    # 锥角半角 (Kappa) - Z轴方向
    kappa_A = np.arctan((Z_coverage / 2) / RA)
    kappa_B = np.arctan((Z_coverage / 2) / RB)
    
    # 探测器高度 (Z方向)
    H_det_A = Z_coverage * FDD / RA
    H_det_B = Z_coverage * FDD / RB

    # 弧长 (Arc Length)
    arc_A = FDD * 2 * beta_A
    arc_B = FDD * 2 * beta_B # This is effective arc if asymmetric
    arc_diff = abs(arc_A - arc_B)
    
    # 坐标系定义: ISO 为原点 (0,0,0)
    FA_x, FA_y = RA * np.cos(-alpha_rad / 2), RA * np.sin(-alpha_rad / 2)
    FB_x, FB_y = RB * np.cos(alpha_rad / 2), RB * np.sin(alpha_rad / 2)
    
    # DetA XY 边缘点
    angle_FA_to_ISO = np.pi - alpha_rad / 2
    
    # --- 物理引擎: 碰撞检测 (Physics Engine) ---
    num_sample_points = 50
    
    # DetA 采样点
    theta_A = np.linspace(angle_FA_to_ISO - beta_A, angle_FA_to_ISO + beta_A, num_sample_points)
    DetA_x = FA_x + FDD * np.cos(theta_A)
    DetA_y = FA_y + FDD * np.sin(theta_A)
    pts_A = np.column_stack((DetA_x, DetA_y))
    
    # DetB 采样点
    angle_FB_to_ISO = np.pi + alpha_rad / 2
    
    if is_asymmetric:
        # Physical Det B: [Center - Beta_Orig, Center + Beta_Orig + Extension]
        # Left side is virtual (Center - Beta_New to Center - Beta_Orig) -> No Collision
        # Right side is real (Center + Beta_Orig to Center + Beta_Orig + Extension) -> Collision
        
        # Original Part
        theta_B_orig = np.linspace(angle_FB_to_ISO - beta_B_orig, angle_FB_to_ISO + beta_B_orig, num_sample_points)
        
        # Right Extension Part
        angle_ext = arc_extension / FDD
        theta_B_ext = np.linspace(angle_FB_to_ISO + beta_B_orig, angle_FB_to_ISO + beta_B_orig + angle_ext, 10)
        
        theta_B = np.concatenate([theta_B_orig, theta_B_ext])
    else:
        theta_B = np.linspace(angle_FB_to_ISO - beta_B, angle_FB_to_ISO + beta_B, num_sample_points)
        
    DetB_x = FB_x + FDD * np.cos(theta_B)
    DetB_y = FB_y + FDD * np.sin(theta_B)
    pts_B = np.column_stack((DetB_x, DetB_y))
    
    # 1. DetA <-> TubeB (FB) 距离
    dists_A_TubeB = np.sqrt(np.sum((pts_A - np.array([FB_x, FB_y]))**2, axis=1))
    min_dist_A_TubeB = np.min(dists_A_TubeB)
    idx_A_TubeB = np.argmin(dists_A_TubeB)
    closest_pt_A_TubeB = (pts_A[idx_A_TubeB][0], pts_A[idx_A_TubeB][1], 0)
    
    # 2. DetB <-> TubeA (FA) 距离
    dists_B_TubeA = np.sqrt(np.sum((pts_B - np.array([FA_x, FA_y]))**2, axis=1))
    min_dist_B_TubeA = np.min(dists_B_TubeA)
    idx_B_TubeA = np.argmin(dists_B_TubeA)
    closest_pt_B_TubeA = (pts_B[idx_B_TubeA][0], pts_B[idx_B_TubeA][1], 0)
    
    # 3. DetA <-> DetB 距离
    dists_Det_Det = np.sqrt(((pts_A[:, np.newaxis, :] - pts_B[np.newaxis, :, :]) ** 2).sum(axis=2))
    min_dist_Det_Det = np.min(dists_Det_Det)
    
    min_idx = np.unravel_index(np.argmin(dists_Det_Det), dists_Det_Det.shape)
    closest_pt_DetA_DetB = (pts_A[min_idx[0]][0], pts_A[min_idx[0]][1], 0)
    closest_pt_DetB_DetA = (pts_B[min_idx[1]][0], pts_B[min_idx[1]][1], 0)
    
    # 综合最小距离
    min_dist_exact = float(min(min_dist_A_TubeB, min_dist_B_TubeA, min_dist_Det_Det))
    
    # --- 离心力计算 (Centrifugal Force) ---
    M_tube = 30.0 # kg
    M_det = 15.0  # kg
    M_det_B_real = M_det * mass_factor_B
    
    omega = 2 * np.pi / rotation_time # rad/s
    # 静态多源架构：机架不旋转 → 无离心加速度，G 负载为 0
    # （"旋转采集"由各射线源的时序触发等效实现，因此没有机械离心负载）
    _arch_probe = ARCH_TABLE.get(str(arch), ARCH_TABLE['dual_source'])
    _rotating = bool(_arch_probe[2])
    if not _rotating:
        omega = 0.0
    
    # 1. 各个组件的离心力 (标量)
    F_cent_TubeA = M_tube * (omega ** 2) * (RA / 1000) # r in meters
    F_cent_TubeB = M_tube * (omega ** 2) * (RB / 1000)
    
    R_DetA = (FDD - RA)
    R_DetB = (FDD - RB)
    
    F_cent_DetA = M_det * (omega ** 2) * (R_DetA / 1000)
    F_cent_DetB = M_det_B_real * (omega ** 2) * (R_DetB / 1000)
    
    # 2. 矢量合力 (Vector Sum)
    angle_TubeA = -alpha_rad / 2
    angle_TubeB = alpha_rad / 2
    
    Fx_TubeA = F_cent_TubeA * np.cos(angle_TubeA)
    Fy_TubeA = F_cent_TubeA * np.sin(angle_TubeA)
    
    Fx_TubeB = F_cent_TubeB * np.cos(angle_TubeB)
    Fy_TubeB = F_cent_TubeB * np.sin(angle_TubeB)
    
    Fx_total = Fx_TubeA + Fx_TubeB
    Fy_total = Fy_TubeA + Fy_TubeB
    
    # 按照截图公式计算合力 (System Pressure = 2 * Vector Sum)
    F_total_mag = np.sqrt(F_cent_TubeA**2 + F_cent_TubeB**2 + 2 * F_cent_TubeA * F_cent_TubeB * np.cos(alpha_rad)) * 2
    
    # 计算系统合力的 G 值
    total_mass_tubes = 2 * M_tube
    g_force_total = F_total_mag / (total_mass_tubes * 9.8)
    
    # --- 物理时间分辨 (Physical Temporal Resolution) ---
    # Formula updated: (rotation_time / 4) * ((alpha + asymmetric_angle) / 90)
    effective_alpha = alpha + angle_ext_deg
    temporal_resolution = (rotation_time / 4) * (effective_alpha / 90)
    
    # --- 散射干扰系数 (Scatter Interference Coefficient) ---
    # MUST be based on REAL Physical Volume B
    ref_sfov = 500.0
    ref_z = 40.0
    vol_ref = (ref_sfov ** 2) * ref_z
    
    # --- Bowtie 滤过器（球管前方整形滤过）：决定**真正照射视野**与剂量分布 ---
    # 中间薄、两侧厚；超出其角跨度 gamma_bt 的射线被滤过体/准直挡住 → 数据截断。
    # 透射率 T(gamma) = exp(-mu * t(gamma))，t(gamma) = t_edge * (|gamma|/gamma_bt)^2
    MU_AL = 0.0196                                # 等效铝线衰减系数 /mm @ ~70 keV
    if bowtie_sfov is None or bowtie_sfov <= 0:
        bowtie_sfov = SFOV_A                      # 默认与标称 SFOV 一致 → 不改变原有结论
    gamma_bt = float(np.arcsin(min(1.0, (bowtie_sfov / 2.0) / RA)))
    sfov_bowtie = float(2.0 * RA * np.sin(gamma_bt))          # 真正视野（ISO 平面）
    bowtie_fan_deg = float(np.rad2deg(2.0 * gamma_bt))        # bowtie 开扇角
    t_edge = float(max(0.0, bowtie_edge_mm))
    T_center = float(np.exp(-MU_AL * 0.0))
    T_edge = float(np.exp(-MU_AL * t_edge))
    bowtie_limits = bool(gamma_bt < beta_A - 1e-9)            # 比探测器扇角更窄 → 截断
    sfov_irradiated = float(min(SFOV_A, sfov_bowtie))         # 实际被照射的视野
    dose_ratio = float(T_edge / T_center)                     # 边缘/中心剂量比

    # --- 扫描模式：轴扫 / 螺旋 + 螺距 pitch ---
    # 螺距定义(IEC 60601-2-44)：pitch = 每圈进床距离 / 总准直宽度(Z轴覆盖)
    scan_mode_s = str(scan_mode).lower()
    helical = scan_mode_s.startswith('h') or scan_mode_s.startswith('螺')
    pitch_v = float(max(0.01, pitch))
    z_per_rot = (pitch_v * Z_coverage) if helical else Z_coverage   # 每圈进床 / 轴扫每圈覆盖
    v_table = (z_per_rot / rotation_time) if helical else 0.0       # 进床速度 mm/s
    rot_per_s = 1.0 / rotation_time if rotation_time else 0.0
    # 体素在射束内期间机架转过的角度：螺旋 = 360/pitch；轴扫 = 整圈
    ang_per_voxel = (360.0 / pitch_v) if helical else 360.0
    # 重建需 ≥180°+扇角；双源叠加两系统角间隔 alpha
    need_deg = 180.0 + float(np.rad2deg(2 * beta_A))
    pitch_max_single = (360.0 / need_deg) if need_deg > 0 else float('inf')
    _den = need_deg - float(np.rad2deg(alpha_rad))
    pitch_max_dual = (360.0 / _den) if _den > 1e-6 else float('inf')
    pitch_over = bool(helical and pitch_v > pitch_max_dual + 1e-9)
    dose_factor = (1.0 / pitch_v) if helical else 1.0               # 相对剂量 ∝ 1/pitch

    # 被照射体积受 bowtie 限制 → 散射体积按真正照射视野计算
    vol_A = (sfov_irradiated ** 2) * Z_coverage
    vol_B = (SFOV_B_physical ** 2) * Z_coverage # Using Physical SFOV
    
    scatter_coeff = (vol_A + vol_B) / vol_ref
    
    # --- 受影响的FOV Sphere体积 (Volume of Affected FOV Sphere) ---
    # Volume = (4/3) * pi * (R_outer^3 - R_inner^3)
    # R = SFOV / 2
    r_outer = SFOV_B_effective / 2
    r_inner = SFOV_B_inner / 2
    affected_volume_cm3 = ((4/3) * np.pi * (r_outer**3 - r_inner**3)) / 1000.0 # Convert mm^3 to cm^3 (mL)
    
    # --- 探测器阵列：阵列总数 = 每排个数 × 排数（整数化自动匹配约束）---
    # 物理像素给的是 ISO 平面尺度：实尺列距 = p_xy·FDD/R，实尺行距 = p_z·FDD/R。
    # 通道数由"探测器弧长 / 实尺列距"决定，排数由"Z 覆盖 / ISO 行距"决定；
    # 两者都必须取整（阵列是离散器件），取整后**反解**实际列距/行距，
    # 使 阵列总数 = n_ch × n_rows 与实际弧长/高度严格自洽（残差 < 0.5 像素）。
    m_amp = FDD / RA if RA else 1.0                      # 放大比
    p_det_col_ideal = float(pixel_xy) * m_amp            # 理想实尺列距
    arc_det = FDD * 2.0 * beta_A                         # A 系统探测器弧长（真实器件长度）
    n_ch_ideal = arc_det / p_det_col_ideal
    n_rows_ideal = Z_coverage / float(pixel_z)
    n_ch_auto = max(8, int(round(n_ch_ideal)))           # 由像素/SFOV 推导的通道数
    if n_ch_set:                                         # 用户指定通道数（每排单元数）→ 以它为准
        n_ch = max(8, int(round(float(n_ch_set))))
        n_ch_manual = True
    else:
        n_ch = n_ch_auto
        n_ch_manual = False
    n_rows = max(1, int(round(n_rows_ideal)))
    col_pitch = arc_det / n_ch                           # 自动匹配后的实尺列距
    row_pitch_iso = Z_coverage / n_rows                  # 自动匹配后的 ISO 行距
    row_pitch_det = row_pitch_iso * m_amp                # 自动匹配后的实尺行距
    array_total = int(n_ch * n_rows)                     # 探测器阵列单元总数
    h_det_match = n_rows * row_pitch_det                 # 阵列高度（实尺）= 匹配后
    col_res_pct = 100.0 * (col_pitch / p_det_col_ideal - 1.0)
    row_res_pct = 100.0 * (row_pitch_iso / float(pixel_z) - 1.0)
    array_ok = (abs(col_res_pct) <= 2.0) and (abs(row_res_pct) <= 2.0)
    arc_residual_px = abs(n_ch_ideal - n_ch)
    row_residual_px = abs(n_rows_ideal - n_rows)

    # --- CT 架构层：源数量 × 能量维度 × 是否旋转 ---
    # 双源只是其中一例；单源宽体=1源无能量维；双层/光子计数=单源 + 能量维度；
    # 静态多源=多源均布 + 不旋转（views 由源数量决定，时间分辨率由源切换时间决定）。
    arch_key = str(arch)
    n_src, energy_dim, rotating, arch_label = ARCH_TABLE.get(arch_key, ARCH_TABLE['dual_source'])
    alpha_deg = float(np.rad2deg(alpha_rad))
    fan_A_deg = float(np.rad2deg(2.0 * beta_A))
    if n_src >= 2:
        src_step_deg = alpha_deg if n_src == 2 else 360.0 / n_src
    else:
        src_step_deg = 0.0
    views_per_rot = max(8, int(round(sampling_rate * rotation_time)))
    if rotating:
        views_total = views_per_rot * n_src                  # 双源两套系统同时采样
        views_per_point = views_total
        ang_step = 360.0 / views_total if views_total else 0.0
        ang_cover_deg = 360.0                                # 旋转一整圈即覆盖
        t_res_arch_ms = (rotation_time * 1000.0) / (4.0 if n_src >= 2 else 2.0)
    else:
        # 静态多源：各源扇角均以 ISO 为中心 → 中心点被**所有源**照射；
        # 真正决定质量的是"每点视角数"与"角度采样间隔"（稀疏视角问题）。
        views_total = n_src                                  # 每个源一个视角（可多次曝光）
        views_per_point = n_src
        ang_step = 360.0 / n_src if n_src else 0.0
        ang_cover_deg = ang_step * max(0, n_src - 1)
        t_res_arch_ms = float(src_switch_us) / 1000.0        # 时间分辨率 = 源切换时间
    need_cover_deg = 180.0 + fan_A_deg
    arch_ok = bool(ang_cover_deg >= need_cover_deg - 1e-9)
    sparse_view = bool(views_per_point < 128)                # 视角过少 → 稀疏视角重建问题
    data_cells = int(n_ch * n_rows * views_total * energy_dim)
    dose_note = {1: '常规（单能量）', 2: '双层：上层吸收低能、上层剂量不可回收 → 剂量效率下降',
                 8: '光子计数：无电子噪声、能量箱可加权 → 剂量效率最高'}.get(
        energy_dim, f'{energy_dim} 能量箱')
    results_arch = {
        'arch_key': arch_key,
        'arch_label': arch_label,
        'n_src': int(n_src),
        'energy_dim': int(energy_dim),
        'rotating': bool(rotating),
        'src_step_deg': float(src_step_deg),
        'views_per_rot': int(views_per_rot),
        'views_total': int(views_total),
        'ang_cover_deg': float(ang_cover_deg),
        'views_per_point': int(views_per_point),
        'ang_step_deg': float(ang_step),
        'sparse_view': bool(sparse_view),
        'arch_ok': arch_ok,
        't_res_arch_ms': float(t_res_arch_ms),
        'data_cells': data_cells,
        'dose_note': dose_note,
        'src_switch_us': float(src_switch_us),
    }

    # =====================================================================
    # 同步辐射仿真模式（第 6 种架构）：以单元光子计数架构改造
    #
    # 物理与算法全部落在 ct_synchrotron.py，依据：
    #   《PCCT 模拟同步辐射 CT · 算法交接手册》(v1.0)
    #   《PCD CT 细胞级扫描 · 工程交接文档》(v1.0)
    # 这里只负责把它的派生量并进统一的结果字典，供 GUI / Agent 读取。
    #
    # 几何映射：RA = SOD（源-等中心），FDD − RA = ODD（等中心-探测器），
    # 故 M = (SOD+ODD)/SOD = FDD/RA —— "把样品推向源"就是调小 RA / 调大 FDD。
    # =====================================================================
    sync = None
    sync_flat = {}
    if arch_key == 'synchrotron':
        try:
            import ct_synchrotron as _CS
        except Exception:                            # pragma: no cover
            _CS = None
        if _CS is not None:
            sync = _CS.analyze(
                source=sync_source, focus_um=sync_focus_um,
                # 优先用同步辐射自己的 SOD/ODD；为 0 时回退到 RA / FDD−RA
                sod_mm=(float(sync_sod_mm) if sync_sod_mm else float(RA)),
                odd_mm=(float(sync_odd_mm) if sync_odd_mm
                        else float(max(FDD - RA, 1e-6))),
                pixel_um=sync_pixel_um, sigma_c_um=sync_sigma_c_um,
                detector=sync_detector, oversample=sync_oversample,
                bins=sync_bins, energy_kev=sync_energy_kev,
                flux_per_mm2=sync_flux, shaping_ns=sync_shaping_ns,
                ref_res_um=sync_ref_res_um, target_res_um=sync_target_res_um,
                phase_enabled=sync_phase, delta_beta=sync_delta_beta,
                vmi_kev=sync_vmi_kev, n_ch=int(sync_cols))

            # 能量箱数由 sync_bins 决定（ARCH_TABLE 里固定为 8 只是占位）；
            # 数据量按**同步辐射自己的探测器阵列**算，不用临床阵列。
            results_arch['energy_dim'] = int(sync['spectrum']['bins'])
            results_arch['data_cells'] = int(int(sync_cols) * int(sync_rows)
                                             * views_total * results_arch['energy_dim'])

            g, sm, mt = sync['geometry'], sync['sampling'], sync['mtf']
            dt, sp, ph = sync['detector'], sync['spectrum'], sync['phase']
            do, rc = sync['dose'], sync['reachable']
            sync_flat = {
                'sync_source': sync['source'],
                'sync_source_label': sync['source_label'],
                'sync_focus_um': float(sync['focus_um']),
                'sync_M': g['M'], 'sync_M_star': g['M_star'],
                'sync_M_practical': g['M_practical'],
                'sync_p_eff_um': g['p_eff_um'], 'sync_penumbra_um': g['penumbra_um'],
                'sync_r_um': g['r_um'], 'sync_r_star_um': g['r_star_um'],
                'sync_limited_by': g['limited_by'], 'sync_fov_mm': g['fov_mm'],
                'sync_cols': int(sync_cols), 'sync_rows': int(sync_rows),
                # 同步辐射的重建体素：样品面有效像素 p/M 经 K 帧过采样细分。
                # 不能用临床的 iso_sampling —— 那是机架通道采样，与本模式无关，
                # 而且量级差 40 倍，会让"反卷积是否有效"的判断完全失真。
                'sync_voxel_um': float(g['p_eff_um'] / max(int(sm['K']), 1)),
                'sync_f0': sm['f0_cyc_mm'], 'sync_fnyq_single': sm['f_nyq_single'],
                'sync_fnyq_over': sm['f_nyq_over'], 'sync_fnyq_eff': sm['f_nyq_effective'],
                'sync_K': sm['K'], 'sync_grid_um': sm['grid_um'],
                'sync_oversample_saturated': sm['oversample_saturated'],
                'sync_nyquist_um': sm['nyquist_um'],
                'sync_mtf10': mt['mtf10_cyc_mm'], 'sync_mtf50': mt['mtf50_cyc_mm'],
                'sync_lp_cm': mt['lp_cm_at_mtf10'],
                'sync_res_mtf10_um': mt['res_at_mtf10_um'],
                'sync_res_mtf50_um': mt['res_at_mtf50_um'],
                'sync_detector': dt['material'], 'sync_detector_label': dt['material_label'],
                'sync_sigma_c_um': dt['sigma_c_um'], 'sync_p_over_sigma': dt['p_over_sigma'],
                'sync_spectral_ok': dt['spectral_ok'],
                'sync_lambda_K_um': dt['lambda_K_um'], 'sync_escape_px': dt['escape_pixels'],
                'sync_kescape_severe': dt['kescape_severe'],
                'sync_shaping_ns': dt['shaping_ns'], 'sync_max_rate': dt['max_rate_cps'],
                'sync_rate_per_pixel': dt['rate_per_pixel_cps'],
                'sync_pileup_pct': float(dt['pileup_fraction'] * 100.0),
                'sync_pileup_ok': dt['pileup_ok'],
                'sync_bins': sp['bins'], 'sync_vmi_kev': sp['vmi_keV'],
                'sync_vmi_ok': sp['vmi_ok'], 'sync_vmi_residual': sp['vmi_residual'],
                'sync_vmi_weights': sp['vmi_weights'], 'sync_bin_edges': sp['bin_edges_keV'],
                'sync_phase_on': ph['enabled'], 'sync_wavelength_mm': ph['wavelength_mm'],
                'sync_paganin_alpha': ph['paganin_alpha_mm2'],
                'sync_N': do['N'], 'sync_dose_n3': do['dose_x_n3'],
                'sync_dose_n4': do['dose_x_n4'],
                'sync_dose_verdict': do['clinical_verdict'],
                'sync_brightness_gap_dec': do['brightness_gap_decades'],
                'sync_swank': do['swank'], 'sync_dqe': do['dqe_at_half_nyq'],
                'sync_neq': do['neq_at_half_nyq'], 'sync_shannon': do['shannon_bits_per_mm'],
                'sync_r_best_um': rc['r_best_um'], 'sync_target_met': rc['target_met'],
                'sync_bottleneck': rc['bottleneck'],
                'sync_summary': _CS.summary_lines(sync),
            }

    # --- 静态多源的**时序触发**模型：等效旋转采集 ---
    # 各源均在同一 R 圆上、对侧各有探测器 → 第 k 个源触发时的射线路径
    # 与"旋转机架位于 θ_k"完全一致。因此顺序触发 = 等效旋转采集，其视角角
    # 就是源角度序列；覆盖 360° ≥ 180°+扇角 → 可用标准扇束 FBP（含 Parker 冗余加权）。
    shots = max(1, int(round(shots_per_source)))
    static_shots = int(shots)
    static_views = int(n_src * shots) if (not rotating) else 0
    static_step_deg = (360.0 / static_views) if static_views else 0.0
    static_acq_ms = (static_views * float(src_switch_us) / 1000.0) if static_views else 0.0
    # 探测器角间距（信息用）；经验判据：步进 ≤ 1°（每圈 ≥360 视角）时 FBP 无明显角向伪影
    # （实测 0.625° → 相关 0.993；严格"匹配采样"判据 dγ≈0.06° 过于保守）
    static_dgamma_deg = float(np.rad2deg(2.0 * beta_A) / max(1, n_ch))
    static_dense_ok = bool(static_views and static_step_deg <= 1.0)

    # 180° 采样在静态阵列中的对应：源角间距 = 360/n_src，180° 只需**一半源数**
    #   12 源 × 15° = 180°（源角跨度）→ 即 180° 采样数据
    #   24 源全部触发 = 360° 全扫描（2× 冗余）
    # 累计探测器阵列 = 参与 180° 采集的源数 × 每源阵列 (n_ch × n_rows)，含 Z 轴排数
    static_step_src_deg = (360.0 / n_src) if n_src else 0.0
    static_src_180 = int(max(1, round(n_src / 2.0))) if (not rotating) else 0
    static_views_180 = int(static_src_180 * shots) if static_src_180 else 0
    static_array_180 = int(static_src_180 * n_ch * n_rows) if static_src_180 else 0
    static_z_row_iso = float(row_pitch_iso)
    # 短扫描 180°+扇角 所需的最小源数（按源角跨度 + 探测器扇角延伸）
    static_src_short = int(np.ceil((180.0 + fan_A_deg) / max(static_step_src_deg, 1e-9))) \
        if (not rotating and static_step_src_deg > 0) else 0
    # 采集环段：实际使用的源数与角度跨度（ring_sources<=0 表示全环）
    static_src_used = (int(min(ring_sources, n_src)) if (int(ring_sources) > 0 and not rotating)
                       else (n_src if not rotating else 0))
    static_span_deg = float(static_src_used * static_step_src_deg) if static_src_used else 0.0
    static_ok_180 = bool(static_span_deg >= (180.0 + fan_A_deg)) if static_src_used else False

    # --- 扫描方案闭环：扫描长度 / 层厚 / 间隔 → 层数、圈数、曝光时间、剂量 ---
    L = float(max(10.0, scan_length))
    sw = float(max(0.1, slice_thickness))
    si = float(max(0.05, slice_interval))
    n_slices = int(max(1, np.floor(L / si + 1e-9) + 1))          # 重建层数
    row_iso = row_pitch_iso                                       # ISO 行距（已匹配）
    axial_steps = 0                                               # 轴扫步进次数（螺旋时不用）
    if rotating:
        if helical:
            rot_net = L / max(z_per_rot, 1e-9)                    # 净圈数 = 长度 / 每圈进床
            over_rot = 1.0 if n_src >= 2 else 2.0                 # 过扫描圈数（双源可少）
            rot_total = rot_net + over_rot
        else:
            over_rot = 0.0
            axial_steps = int(np.ceil(L / max(Z_coverage, 1e-9)))
            rot_net = float(axial_steps)
            rot_total = float(axial_steps)
        scan_time_s = rot_total * rotation_time
    else:
        over_rot = 0.0
        axial_steps = 1
        rot_net = rot_total = 1.0
        scan_time_s = n_src * float(src_switch_us) / 1e6          # 静态：各源顺序曝光
    total_mas_rel = scan_time_s / max(rotation_time, 1e-9)        # 相对总 mAs（mA 固定）
    ctdi_rel = (1.0 / pitch_v) if (helical and rotating) else 1.0  # CTDIvol ∝ 1/pitch
    dlp_rel = ctdi_rel * (L / 100.0)                              # DLP ∝ CTDIvol × 长度
    ssp_eff = float(np.sqrt(sw ** 2 + ((pitch_v * row_iso * 0.5) ** 2
                                       if (helical and rotating) else 0.0)))  # 有效层厚(经验)
    slice_vs_row = sw / max(row_iso, 1e-9)                        # 层厚 / 行距
    results_proto = {
        'scan_length': L, 'slice_thickness': sw, 'slice_interval': si,
        'n_slices': n_slices, 'rot_net': float(rot_net), 'rot_total': float(rot_total),
        'over_rot': float(over_rot), 'axial_steps': int(axial_steps),
        'scan_time_s': float(scan_time_s), 'total_mas_rel': float(total_mas_rel),
        'ctdi_rel': float(ctdi_rel), 'dlp_rel': float(dlp_rel),
        'ssp_eff': ssp_eff, 'slice_vs_row': float(slice_vs_row),
        'row_iso': float(row_iso),
    }

    # --- 约束检查 ---
    is_safe = min_dist_exact >= required_min_dist
    is_arc_ok = arc_diff >= required_arc_diff
    
    results = {
        "fan_angle_A": float(fan_angle_A),
        "fan_angle_B": float(fan_angle_B),
        "arc_diff": float(arc_diff),
        "min_dist": float(min_dist_exact),
        "is_safe": bool(is_safe),
        "is_arc_ok": bool(is_arc_ok),
        "RA": float(RA),
        "RB": float(RB),
        "beta_A": float(beta_A), "beta_B": float(beta_B),
        "beta_B_orig": float(beta_B_orig), # Keep track of original beta
        "kappa_A": float(kappa_A), "kappa_B": float(kappa_B),
        "H_det_A": float(H_det_A), "H_det_B": float(H_det_B),
        "R_A_coord": (float(FA_x), float(FA_y), 0),
        "R_B_coord": (float(FB_x), float(FB_y), 0),
        
        "min_dist_A_TubeB": float(min_dist_A_TubeB),
        "min_dist_B_TubeA": float(min_dist_B_TubeA),
        "min_dist_Det_Det": float(min_dist_Det_Det),
        
        "closest_pt_A_TubeB": (float(closest_pt_A_TubeB[0]), float(closest_pt_A_TubeB[1]), 0),
        "closest_pt_B_TubeA": (float(closest_pt_B_TubeA[0]), float(closest_pt_B_TubeA[1]), 0),
        "closest_pt_DetA_DetB": (float(closest_pt_DetA_DetB[0]), float(closest_pt_DetA_DetB[1]), 0),
        "closest_pt_DetB_DetA": (float(closest_pt_DetB_DetA[0]), float(closest_pt_DetB_DetA[1]), 0),
        
        "g_force_TubeA": float(F_cent_TubeA / 9.8 / M_tube),
        "g_force_TubeB": float(F_cent_TubeB / 9.8 / M_tube),
        "g_force_DetA": float(F_cent_DetA / 9.8 / M_det),
        "g_force_DetB": float(F_cent_DetB / 9.8 / M_det_B_real), # Use real mass
        "F_total_mag": float(F_total_mag),
        "g_force_total": float(g_force_total),
        "g_force_note": ('无旋转：机架静止，G 负载 = 0（源时序触发模拟旋转采集）'
                         if not _rotating else ''),
        "F_total_vector": (float(Fx_total), float(Fy_total)),
        
        "temporal_resolution": float(temporal_resolution),
        "scatter_coeff": float(scatter_coeff),
        "affected_volume_cm3": float(affected_volume_cm3), # 新增体积

        # --- Bowtie 滤过器派生量 ---
        "bowtie_sfov_set": float(bowtie_sfov),
        "bowtie_sfov": float(sfov_bowtie),            # 真正视野（ISO 平面）
        "bowtie_fan_deg": float(bowtie_fan_deg),      # bowtie 开扇角
        "gamma_bt": float(gamma_bt),                  # bowtie 半角 (rad)
        "bowtie_limits": bool(bowtie_limits),         # 是否限制视野（截断）
        "sfov_irradiated": float(sfov_irradiated),    # 实际照射视野
        "bt_mu": float(MU_AL),
        "bt_t_edge": float(t_edge),
        "bt_T_center": float(T_center),
        "bt_T_edge": float(T_edge),
        "bt_dose_ratio": float(dose_ratio),           # 边缘/中心剂量比
        "bt_dose_saving_pct": float(100.0 * (1.0 - dose_ratio)),

        # --- 扫描模式 / 螺距 ---
        "scan_mode": ('helical' if helical else 'axial'),
        "pitch": float(pitch_v),
        "z_per_rot": float(z_per_rot),
        "v_table": float(v_table),
        "rot_per_s": float(rot_per_s),
        "ang_per_voxel": float(ang_per_voxel),
        "pitch_max_single": float(pitch_max_single),
        "pitch_max_dual": float(pitch_max_dual),
        "pitch_over": bool(pitch_over),
        "dose_factor": float(dose_factor),

        # --- 探测器阵列（整数化自动匹配）---
        "n_ch": int(n_ch),                       # 每排个数（通道数）
        "n_ch_auto": int(n_ch_auto),             # 自动推导值
        "n_ch_manual": bool(n_ch_manual),        # 是否由用户指定
        "iso_sampling_mm": float(col_pitch * RA / FDD),   # 通道数决定的 ISO 平面采样间隔
        "n_rows": int(n_rows),                   # 排数
        "array_total": int(array_total),         # 阵列总数 = n_ch × n_rows
        "n_ch_ideal": float(n_ch_ideal),
        "n_rows_ideal": float(n_rows_ideal),
        "col_pitch": float(col_pitch),           # 匹配后实尺列距
        "row_pitch_det": float(row_pitch_det),   # 匹配后实尺行距
        "row_pitch_iso": float(row_pitch_iso),   # 匹配后 ISO 行距
        "h_det_match": float(h_det_match),       # 阵列高度（实尺）
        "array_col_res_pct": float(col_res_pct),
        "array_row_res_pct": float(row_res_pct),
        "array_residual_px": (float(arc_residual_px), float(row_residual_px)),
        "array_ok": bool(array_ok),
        "arch_key": results_arch["arch_key"],
        "arch_label": results_arch["arch_label"],
        "n_src": results_arch["n_src"],
        "energy_dim": results_arch["energy_dim"],
        "rotating": results_arch["rotating"],
        "src_step_deg": results_arch["src_step_deg"],
        "views_per_rot": results_arch["views_per_rot"],
        "views_total": results_arch["views_total"],
        "ang_cover_deg": results_arch["ang_cover_deg"],
        "views_per_point": results_arch["views_per_point"],
        "ang_step_deg": results_arch["ang_step_deg"],
        "sparse_view": results_arch["sparse_view"],
        "scan_length": results_proto["scan_length"],
        "slice_thickness": results_proto["slice_thickness"],
        "slice_interval": results_proto["slice_interval"],
        "n_slices": results_proto["n_slices"],
        "rot_net": results_proto["rot_net"],
        "rot_total": results_proto["rot_total"],
        "over_rot": results_proto["over_rot"],
        "axial_steps": results_proto["axial_steps"],
        "scan_time_s": results_proto["scan_time_s"],
        "total_mas_rel": results_proto["total_mas_rel"],
        "ctdi_rel": results_proto["ctdi_rel"],
        "dlp_rel": results_proto["dlp_rel"],
        "ssp_eff": results_proto["ssp_eff"],
        "slice_vs_row": results_proto["slice_vs_row"],
        "row_iso": results_proto["row_iso"],
        "shots_per_source": int(static_shots),
        "static_views": int(static_views),
        "static_step_deg": float(static_step_deg),
        "static_acq_ms": float(static_acq_ms),
        "static_dgamma_deg": float(static_dgamma_deg),
        "static_dense_ok": bool(static_dense_ok),
        "static_step_src_deg": float(static_step_src_deg),
        "static_src_180": int(static_src_180),
        "static_views_180": int(static_views_180),
        "static_array_180": int(static_array_180),
        "static_z_row_iso": float(static_z_row_iso),
        "static_src_short": int(static_src_short),
        "static_src_used": int(static_src_used),
        "static_span_deg": float(static_span_deg),
        "static_ok_180": bool(static_ok_180),
        "arch_ok": results_arch["arch_ok"],
        "t_res_arch_ms": results_arch["t_res_arch_ms"],
        "data_cells": results_arch["data_cells"],
        "dose_note": results_arch["dose_note"],
        "src_switch_us": results_arch["src_switch_us"],
        
        "alpha_rad": float(alpha_rad),
        "is_asymmetric": is_asymmetric,
        "SFOV_B_effective": float(SFOV_B_effective),
        "SFOV_B_inner": float(SFOV_B_inner)
    }
    # 同步辐射派生量（arch='synchrotron' 时非空）统一以 sync_ 前缀并入
    results.update(sync_flat)
    return results, fan_angle_A, fan_angle_B
