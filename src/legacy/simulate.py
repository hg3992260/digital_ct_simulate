import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np

# --- 1. 核心几何计算函数 ---

def calculate_geometry(alpha, RA, RB, FDD, SFOV_A, SFOV_B, Z_coverage, rotation_time, is_asymmetric=False, required_min_dist=150, required_arc_diff=120):
    """根据输入参数计算所有相关几何值和约束。"""
    
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
        mass_factor_B = (arc_B_orig + arc_extension) / arc_B_orig
        
        # Physical SFOV B (For Scatter calculation)
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
    arc_B = FDD * 2 * beta_B
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
        theta_B_orig = np.linspace(angle_FB_to_ISO - beta_B_orig, angle_FB_to_ISO + beta_B_orig, num_sample_points)
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
    
    # 1. 各个组件的离心力 (标量)
    F_cent_TubeA = M_tube * (omega ** 2) * (RA / 1000)
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
    
    F_total_mag = np.sqrt(F_cent_TubeA**2 + F_cent_TubeB**2 + 2 * F_cent_TubeA * F_cent_TubeB * np.cos(alpha_rad)) * 2
    
    total_mass_tubes = 2 * M_tube
    g_force_total = F_total_mag / (total_mass_tubes * 9.8)
    
    # --- 物理时间分辨 (Physical Temporal Resolution) ---
    # Formula updated: (rotation_time / 4) * ((alpha + asymmetric_angle) / 90)
    effective_alpha = alpha + angle_ext_deg
    temporal_resolution = (rotation_time / 4) * (effective_alpha / 90)
    
    # --- 散射干扰系数 (Scatter Interference Coefficient) ---
    ref_sfov = 500.0
    ref_z = 40.0
    vol_ref = (ref_sfov ** 2) * ref_z
    
    vol_A = (SFOV_A ** 2) * Z_coverage
    vol_B = (SFOV_B_physical ** 2) * Z_coverage
    
    scatter_coeff = (vol_A + vol_B) / vol_ref
    
    # --- 受影响的FOV Sphere体积 (Volume of Affected FOV Sphere) ---
    # Volume = (4/3) * pi * (R_outer^3 - R_inner^3)
    # R = SFOV / 2
    r_outer = SFOV_B_effective / 2
    r_inner = SFOV_B_inner / 2
    affected_volume_cm3 = ((4/3) * np.pi * (r_outer**3 - r_inner**3)) / 1000.0 # Convert mm^3 to cm^3 (mL)
    
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
        "kappa_A": float(kappa_A), "kappa_B": float(kappa_B),
        "H_det_A": float(H_det_A), "H_det_B": float(H_det_B),
        "R_A_coord": (float(FA_x), float(FA_y), 0),
        "R_B_coord": (float(FB_x), float(FB_y), 0),
        
        # 物理引擎详细数据
        "min_dist_A_TubeB": float(min_dist_A_TubeB),
        "min_dist_B_TubeA": float(min_dist_B_TubeA),
        "min_dist_Det_Det": float(min_dist_Det_Det),
        
        "closest_pt_A_TubeB": (float(closest_pt_A_TubeB[0]), float(closest_pt_A_TubeB[1]), 0),
        "closest_pt_B_TubeA": (float(closest_pt_B_TubeA[0]), float(closest_pt_B_TubeA[1]), 0),
        "closest_pt_DetA_DetB": (float(closest_pt_DetA_DetB[0]), float(closest_pt_DetA_DetB[1]), 0),
        "closest_pt_DetB_DetA": (float(closest_pt_DetB_DetA[0]), float(closest_pt_DetB_DetA[1]), 0),
        
        # 离心力数据
        "g_force_TubeA": float(F_cent_TubeA / 9.8 / M_tube),
        "g_force_TubeB": float(F_cent_TubeB / 9.8 / M_tube),
        "g_force_DetA": float(F_cent_DetA / 9.8 / M_det),
        "g_force_DetB": float(F_cent_DetB / 9.8 / M_det_B_real),
        "F_total_mag": float(F_total_mag),
        "g_force_total": float(g_force_total),
        "F_total_vector": (float(Fx_total), float(Fy_total)),
        
        "temporal_resolution": float(temporal_resolution),
        "scatter_coeff": float(scatter_coeff),
        "affected_volume_cm3": float(affected_volume_cm3), # 新增体积
        
        "alpha_rad": float(alpha_rad),
        "is_asymmetric": is_asymmetric,
        "SFOV_B_effective": float(SFOV_B_effective),
        "SFOV_B_inner": float(SFOV_B_inner)
    }
    return results, fan_angle_A, fan_angle_B

# --- 2. 3D 绘图函数 ---

def generate_cone_beam_mesh(F, R, FDD, beta, H_det, angle_iso, color, name_prefix):
    """
    生成锥束和探测器的3D网格数据
    F: 焦点坐标 (x, y, z)
    R: 焦-中心距离
    FDD: 焦-探测器距离
    beta: 扇角半角 (radians)
    H_det: 探测器总高度
    angle_iso: 焦点指向ISO的角度 (radians)
    """
    Fx, Fy, Fz = F
    
    # --- 1. 探测器曲面 (Cylindrical Surface) ---
    # 探测器是以 F 为圆心，FDD 为半径的圆柱面的一部分
    # 角度范围: angle_iso - beta 到 angle_iso + beta
    # 高度范围: -H_det/2 到 H_det/2
    
    theta = np.linspace(angle_iso - beta, angle_iso + beta, 20)
    z = np.linspace(-H_det/2, H_det/2, 5)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    # 探测器表面坐标
    Det_X = Fx + FDD * np.cos(theta_grid)
    Det_Y = Fy + FDD * np.sin(theta_grid)
    Det_Z = Fz + z_grid
    
    det_surface = go.Surface(
        x=Det_X, y=Det_Y, z=Det_Z,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        opacity=0.8,
        name=f'{name_prefix} 探测器'
    )
    
    # --- 2. 锥束光路 (Cone Beam Volume) ---
    # 为了不遮挡，我们只画棱锥的边框线和几条代表性的光线
    
    lines = []
    
    # 探测器四个角点
    corners = [
        (Fx + FDD * np.cos(angle_iso - beta), Fy + FDD * np.sin(angle_iso - beta), -H_det/2),
        (Fx + FDD * np.cos(angle_iso + beta), Fy + FDD * np.sin(angle_iso + beta), -H_det/2),
        (Fx + FDD * np.cos(angle_iso + beta), Fy + FDD * np.sin(angle_iso + beta), H_det/2),
        (Fx + FDD * np.cos(angle_iso - beta), Fy + FDD * np.sin(angle_iso - beta), H_det/2),
    ]
    
    # 2.1 从焦点到四个角点的连线
    for corner in corners:
        lines.append(go.Scatter3d(
            x=[Fx, corner[0]], y=[Fy, corner[1]], z=[Fz, corner[2]],
            mode='lines', line=dict(color=color, width=2), showlegend=False
        ))
        
    # 2.2 连接探测器边缘的线
    # 上边缘
    theta_line = np.linspace(angle_iso - beta, angle_iso + beta, 20)
    top_x = Fx + FDD * np.cos(theta_line)
    top_y = Fy + FDD * np.sin(theta_line)
    top_z = np.full_like(top_x, H_det/2)
    lines.append(go.Scatter3d(x=top_x, y=top_y, z=top_z, mode='lines', line=dict(color=color, width=2), showlegend=False))
    
    # 下边缘
    bottom_z = np.full_like(top_x, -H_det/2)
    lines.append(go.Scatter3d(x=top_x, y=top_y, z=bottom_z, mode='lines', line=dict(color=color, width=2), showlegend=False))
    
    # 左右边缘
    lines.append(go.Scatter3d(x=[corners[0][0], corners[3][0]], y=[corners[0][1], corners[3][1]], z=[corners[0][2], corners[3][2]], mode='lines', line=dict(color=color, width=2), showlegend=False))
    lines.append(go.Scatter3d(x=[corners[1][0], corners[2][0]], y=[corners[1][1], corners[2][1]], z=[corners[1][2], corners[2][2]], mode='lines', line=dict(color=color, width=2), showlegend=False))

    # --- 3. FOV 覆盖区域 (ISO处的球体) ---
    # SFOV 半径 = R * sin(beta)
    sfov_radius = R * np.sin(beta)
    
    # 生成球体网格
    phi = np.linspace(0, np.pi, 20)
    theta_sphere = np.linspace(0, 2 * np.pi, 40)
    phi_grid, theta_grid = np.meshgrid(phi, theta_sphere)
    
    sfov_x = sfov_radius * np.sin(phi_grid) * np.cos(theta_grid)
    sfov_y = sfov_radius * np.sin(phi_grid) * np.sin(theta_grid)
    sfov_z = sfov_radius * np.cos(phi_grid)
    
    sphere = go.Surface(
        x=sfov_x, y=sfov_y, z=sfov_z,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        opacity=0.1,
        name=f'{name_prefix} SFOV Sphere',
        hoverinfo='skip'
    )
    
    # 也可以保留一个赤道圆圈作为辅助线
    theta_circle = np.linspace(0, 2*np.pi, 60)
    circle_x = sfov_radius * np.cos(theta_circle)
    circle_y = sfov_radius * np.sin(theta_circle)
    lines.append(go.Scatter3d(
        x=circle_x, y=circle_y, z=np.zeros_like(circle_x),
        mode='lines', line=dict(color=color, width=2, dash='dash'), 
        name=f'{name_prefix} SFOV Equator'
    ))

    return [det_surface, sphere] + lines

def create_fan_sector_mesh_plotly(origin, corners, color, name, opacity=0.5):
    """创建扇区Mesh (Pyramid shape)"""
    # Vertices: Origin + 4 Corners
    x = [origin[0]] + [c[0] for c in corners]
    y = [origin[1]] + [c[1] for c in corners]
    z = [origin[2]] + [c[2] for c in corners]
    
    # Faces (Triangles) connecting Origin to base edges
    # Indices: 0 is origin, 1-4 are corners
    i = [0, 0, 0, 0]
    j = [1, 2, 3, 4]
    k = [2, 3, 4, 1]
    
    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color=color,
        opacity=opacity,
        name=name,
        showlegend=True,
        flatshading=True
    )

def create_fov_analysis_figure(results, FDD):
    """创建FOV影响分析的专用3D视图 (FOV Impact Analysis)"""
    if "error" in results:
        return go.Figure()

    fig = go.Figure()
    
    # 参数提取
    RB = results["RB"]
    FB = results["R_B_coord"]
    H_det_B = results["H_det_B"]
    alpha_rad = results["alpha_rad"]
    angle_iso_B = np.pi + alpha_rad / 2
    
    is_asymmetric = results.get("is_asymmetric", False)
    
    # 重新推导 beta 角 (为了绘图精确性)
    # SFOV_B_inner = 2 * RB * sin(beta_orig)
    beta_B_orig = np.arcsin(results["SFOV_B_inner"] / (2 * RB))
    
    # 绘制 ISO 中心
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers', marker=dict(size=5, color='white'),
        name='ISO'
    ))
    
    # 1. Inner Safe Sphere (Cyan) - Always present
    r_inner = results["SFOV_B_inner"] / 2
    
    # 使用参数方程生成球体
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    
    inner_x = r_inner * np.sin(phi_grid) * np.cos(theta_grid)
    inner_y = r_inner * np.sin(phi_grid) * np.sin(theta_grid)
    inner_z = r_inner * np.cos(phi_grid)
    
    fig.add_trace(go.Surface(
        x=inner_x, y=inner_y, z=inner_z,
        colorscale=[[0, 'cyan'], [1, 'cyan']],
        showscale=False, opacity=0.3,
        name='Safe FOV (Inner)', hoverinfo='skip'
    ))

    # 2. Fan Geometries
    # 计算 Real Fan 的角度范围
    if is_asymmetric:
        arc_extension = 120.0
        angle_ext = arc_extension / FDD # radians
        
        # Real Fan: [Center - Beta_Orig, Center + Beta_Orig + Extension]
        # 注意: 这里的 Center 是指 angle_iso_B
        fan_real_start = angle_iso_B - beta_B_orig
        fan_real_end = angle_iso_B + beta_B_orig + angle_ext
        
        # Missing/Virtual Fan: [Center - Beta_Orig - Extension, Center - Beta_Orig]
        fan_virt_start = angle_iso_B - beta_B_orig - angle_ext
        fan_virt_end = angle_iso_B - beta_B_orig
        
        # Affected Shell (Orange)
        r_outer = results["SFOV_B_effective"] / 2
        outer_x = r_outer * np.sin(phi_grid) * np.cos(theta_grid)
        outer_y = r_outer * np.sin(phi_grid) * np.sin(theta_grid)
        outer_z = r_outer * np.cos(phi_grid)
        
        fig.add_trace(go.Surface(
            x=outer_x, y=outer_y, z=outer_z,
            colorscale=[[0, 'orange'], [1, 'orange']],
            showscale=False, opacity=0.2,
            name='Affected Shell', hoverinfo='skip'
        ))
        
    else:
        # Symmetric
        fan_real_start = angle_iso_B - beta_B_orig
        fan_real_end = angle_iso_B + beta_B_orig
        fan_virt_start = None
        fan_virt_end = None

    # Helper to get corners
    def get_corners(start_angle, end_angle):
        return [
            (FB[0] + FDD * np.cos(start_angle), FB[1] + FDD * np.sin(start_angle), -H_det_B/2),
            (FB[0] + FDD * np.cos(end_angle), FB[1] + FDD * np.sin(end_angle), -H_det_B/2),
            (FB[0] + FDD * np.cos(end_angle), FB[1] + FDD * np.sin(end_angle), H_det_B/2),
            (FB[0] + FDD * np.cos(start_angle), FB[1] + FDD * np.sin(start_angle), H_det_B/2)
        ]

    # Draw Real Fan (Blue)
    corners_real = get_corners(fan_real_start, fan_real_end)
    fig.add_trace(create_fan_sector_mesh_plotly(FB, corners_real, 'blue', 'Real Fan B', opacity=0.3))
    
    # Draw Missing Fan (Red) if asymmetric
    if is_asymmetric and fan_virt_start is not None:
        corners_virt = get_corners(fan_virt_start, fan_virt_end)
        fig.add_trace(create_fan_sector_mesh_plotly(FB, corners_virt, 'red', 'Missing Fan Sector', opacity=0.6))
        
        # High Contrast Edges (Yellow Lines) for Missing Fan
        # Edges: FB -> Corners
        for corner in corners_virt:
            fig.add_trace(go.Scatter3d(
                x=[FB[0], corner[0]], y=[FB[1], corner[1]], z=[FB[2], corner[2]],
                mode='lines', line=dict(color='yellow', width=5),
                showlegend=False
            ))
        # Detector Rectangle Edges
        rect_x = [c[0] for c in corners_virt] + [corners_virt[0][0]]
        rect_y = [c[1] for c in corners_virt] + [corners_virt[0][1]]
        rect_z = [c[2] for c in corners_virt] + [corners_virt[0][2]]
        fig.add_trace(go.Scatter3d(
            x=rect_x, y=rect_y, z=rect_z,
            mode='lines', line=dict(color='yellow', width=5),
            showlegend=False
        ))
        
        # Add Text Annotation
        mid_x = (corners_virt[0][0] + corners_virt[2][0]) / 2
        mid_y = (corners_virt[0][1] + corners_virt[2][1]) / 2
        fig.add_trace(go.Scatter3d(
            x=[mid_x], y=[mid_y], z=[0],
            mode='text', text=['Missing Data'],
            textfont=dict(color='yellow', size=12, check_contrast=True),
            name='Missing Label'
        ))

    # Layout Setup
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (mm)', range=[-600, 600], backgroundcolor="rgb(30, 30, 30)"),
            yaxis=dict(title='Y (mm)', range=[-600, 600], backgroundcolor="rgb(30, 30, 30)"),
            zaxis=dict(title='Z (mm)', range=[-300, 300], backgroundcolor="rgb(30, 30, 30)"),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),
            bgcolor="rgb(0, 0, 0)"
        ),
        title='FOV 影响分析 (FOV Impact Analysis)',
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="rgb(0, 0, 0)",
        font=dict(color="white")
    )
    
    return fig

def create_3d_figure(results, FDD):
    """创建双源CT扇区和焦点、探测器的3D可视化图形。"""
    
    if "error" in results:
        return go.Figure().add_annotation(text=results["error"], showarrow=False)

    RA = results["RA"]
    RB = results["RB"]
    alpha_rad = results["alpha_rad"]

    # 焦点坐标 (FA, FB)
    FA = results["R_A_coord"]
    FB = results["R_B_coord"]
    
    fig = go.Figure()
    
    # 绘制焦点
    fig.add_trace(go.Scatter3d(
        x=[FA[0], FB[0]], y=[FA[1], FB[1]], z=[FA[2], FB[2]], 
        mode='markers', marker=dict(size=5, color=['red', 'blue']), 
        name='焦点 FA/FB'
    ))
    
    # 绘制 ISO 中心
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], 
        mode='markers', marker=dict(size=3, color='black'), 
        name='ISO 中心'
    ))

    # --- 绘制离心力矢量 ---
    # 总合力矢量
    F_vec = results["F_total_vector"]
    F_mag = results["F_total_mag"]
    
    # 缩放矢量长度以便可视化 (例如，1000N 对应 100mm)
    scale_factor = 0.1 
    vec_end_x = float(F_vec[0] * scale_factor)
    vec_end_y = float(F_vec[1] * scale_factor)
    
    fig.add_trace(go.Scatter3d(
        x=[0, vec_end_x], y=[0, vec_end_y], z=[0, 0],
        mode='lines+markers', 
        line=dict(color='purple', width=5),
        marker=dict(size=5, symbol='diamond', color='purple'), # Change symbol to diamond just in case arrow is weird
        name='系统合力离心力'
    ))
    
    # 在矢量末端标注大小
    fig.add_trace(go.Scatter3d(
        x=[vec_end_x], y=[vec_end_y], z=[0],
        mode='text',
        text=[f'System Pressure: {F_mag:.0f} N ({results["g_force_total"]:.1f} G)'],
        textposition="top center",
        textfont=dict(color='purple', size=12),
        showlegend=False
    ))
    
    # --- 绘制系统夹角标注 ---
    # 画一条圆弧连接 FA 和 FB 的方向向量，并在中间标注角度
    
    # 角度范围: 从 -alpha/2 到 alpha/2 (注意 FA 在 -alpha/2, FB 在 alpha/2)
    # 半径: 取 min(RA, RB) * 0.4，画在离 ISO 较近的位置
    r_arc = min(RA, RB) * 0.4
    theta_arc = np.linspace(-alpha_rad / 2, alpha_rad / 2, 20)
    arc_x = r_arc * np.cos(theta_arc)
    arc_y = r_arc * np.sin(theta_arc)
    arc_z = np.zeros_like(arc_x)
    
    fig.add_trace(go.Scatter3d(
        x=arc_x, y=arc_y, z=arc_z,
        mode='lines', line=dict(color='black', width=4), # 加粗黑色线条
        name='夹角弧线'
    ))
    
    # 在弧线中间添加文字标注
    mid_idx = len(theta_arc) // 2
    angle_deg = np.rad2deg(alpha_rad)
    fig.add_trace(go.Scatter3d(
        x=[arc_x[mid_idx]], y=[arc_y[mid_idx]], z=[0],
        mode='text',
        text=[f'{angle_deg:.1f}°'],
        textposition="middle center",
        textfont=dict(size=14, color='black', family="Arial Black"), # 加大加粗字体
        name='夹角数值'
    ))
    
    # 添加两条辅助虚线，从 ISO 指向 FA 和 FB 的方向，辅助显示夹角
    fig.add_trace(go.Scatter3d(
        x=[0, FA[0]*0.5], y=[0, FA[1]*0.5], z=[0, 0],
        mode='lines', line=dict(color='gray', width=1, dash='dash'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, FB[0]*0.5], y=[0, FB[1]*0.5], z=[0, 0],
        mode='lines', line=dict(color='gray', width=1, dash='dash'),
        showlegend=False
    ))

    # 生成并绘制 A 系统几何
    # A 系统角度: pi - alpha/2 (指向 ISO)
    angle_iso_A = np.pi - alpha_rad / 2
    traces_A = generate_cone_beam_mesh(FA, RA, FDD, results["beta_A"], results["H_det_A"], angle_iso_A, 'red', 'A系统')
    for trace in traces_A:
        fig.add_trace(trace)
        
    # 生成并绘制 B 系统几何
    # B 系统角度: pi + alpha/2 (指向 ISO)
    angle_iso_B = np.pi + alpha_rad / 2
    
    # B 系统颜色处理
    color_B = 'blue'
    
    traces_B = generate_cone_beam_mesh(FB, RB, FDD, results["beta_B"], results["H_det_B"], angle_iso_B, color_B, 'B系统')
    
    # 如果是非对称模式，修改 SFOV 显示逻辑
    if results.get("is_asymmetric", False):
        # traces_B[1] 是 SFOV Sphere (基于 beta_B，即 Effective SFOV)
        # 将其改为 "Affected Shell" (Orange)
        sfov_outer = traces_B[1]
        sfov_outer.colorscale = [[0, 'orange'], [1, 'orange']]
        sfov_outer.name = 'B系统 Affected Shell'
        sfov_outer.opacity = 0.2
        
        # 添加 Inner Safe Sphere (Cyan)
        r_inner = results["SFOV_B_inner"] / 2
        
        phi = np.linspace(0, np.pi, 20)
        theta_sphere = np.linspace(0, 2 * np.pi, 40)
        phi_grid, theta_grid = np.meshgrid(phi, theta_sphere)
        
        sfov_inner_x = r_inner * np.sin(phi_grid) * np.cos(theta_grid)
        sfov_inner_y = r_inner * np.sin(phi_grid) * np.sin(theta_grid)
        sfov_inner_z = r_inner * np.cos(phi_grid)
        
        sfov_inner = go.Surface(
            x=sfov_inner_x, y=sfov_inner_y, z=sfov_inner_z,
            colorscale=[[0, 'cyan'], [1, 'cyan']],
            showscale=False,
            opacity=0.3,
            name='B系统 Safe Zone',
            hoverinfo='skip'
        )
        fig.add_trace(sfov_inner)
        
    for trace in traces_B:
        fig.add_trace(trace)

    # --- 绘制碰撞安全距离连线 ---
    
    # 1. DetA -> TubeB
    closest_A_TubeB = results["closest_pt_A_TubeB"]
    fig.add_trace(go.Scatter3d(
        x=[closest_A_TubeB[0], FB[0]], y=[closest_A_TubeB[1], FB[1]], z=[closest_A_TubeB[2], FB[2]],
        mode='lines+text', line=dict(color='orange', width=3, dash='dot'),
        text=[f'{results["min_dist_A_TubeB"]:.1f}'], textposition="middle center",
        name='DetA-TubeB 距离'
    ))
    
    # 2. DetB -> TubeA
    closest_B_TubeA = results["closest_pt_B_TubeA"]
    fig.add_trace(go.Scatter3d(
        x=[closest_B_TubeA[0], FA[0]], y=[closest_B_TubeA[1], FA[1]], z=[closest_B_TubeA[2], FA[2]],
        mode='lines+text', line=dict(color='orange', width=3, dash='dot'),
        text=[f'{results["min_dist_B_TubeA"]:.1f}'], textposition="middle center",
        name='DetB-TubeA 距离'
    ))
    
    # 3. DetA -> DetB (如果距离小于某个阈值才显示，或者总是显示但颜色不同)
    closest_DetA = results["closest_pt_DetA_DetB"]
    closest_DetB = results["closest_pt_DetB_DetA"]
    det_det_dist = results["min_dist_Det_Det"]
    
    det_det_color = 'red' if det_det_dist < 200 else 'green' # 设定一个显示阈值
    
    fig.add_trace(go.Scatter3d(
        x=[closest_DetA[0], closest_DetB[0]], y=[closest_DetA[1], closest_DetB[1]], z=[closest_DetA[2], closest_DetB[2]],
        mode='lines+text', line=dict(color=det_det_color, width=4), # 实线强调
        text=[f'{det_det_dist:.1f}'], textposition="middle center",
        name='DetA-DetB 距离'
    ))

    # 调整布局
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (mm)', range=[-800, 800]),
            yaxis=dict(title='Y (mm)', range=[-800, 800]),
            zaxis=dict(title='Z (mm)', range=[-400, 400]), # Z轴范围根据实际探测器和FOV大小调整，不需要那么大，避免空旷
            aspectmode='manual', # 手动控制比例
            aspectratio=dict(x=1, y=1, z=0.5) # Z轴显示长度是XY的一半，但因为range也是一半，所以视觉比例是1:1:1
        ),
        title='双源 CT 真实锥束几何模拟 (Cone Beam)',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

# --- 3. Dash 应用和布局 ---

app = dash.Dash(__name__, title="双源 CT 真实几何模拟器 (Cone Beam)")

app.layout = html.Div([
    # 页头：LOGO + 标题（LOGO 由 Dash 自动从 assets/ 目录提供）
    html.Div([
        html.Img(src='/assets/logo.png',
                 style={'height': '80px', 'marginRight': '18px', 'verticalAlign': 'middle',
                        'borderRadius': '10px'}),
        html.H1("双源 CT 真实几何模拟器 (Cone Beam)",
                style={'textAlign': 'center', 'display': 'inline-block', 'verticalAlign': 'middle',
                       'margin': '0'}),
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
              'padding': '12px 0'}),
    
    # 参数调整区域
    html.Div([
        html.Div([
            html.H3("系统几何参数", style={'textAlign': 'center'}),
            
            # 页脚声明 (移动到这里，并改为显示框样式)
            html.Div([
                html.P("Developed by Christ.paul90@gmail.com", style={'margin': '5px 0', 'fontStyle': 'italic'}),
                html.P("All the results have been rigorously supported by geometric parameters and the physics engine.", style={'margin': '5px 0', 'fontWeight': 'bold'})
            ], style={
                'backgroundColor': '#e8f4f8', 
                'padding': '10px', 
                'borderRadius': '5px', 
                'border': '1px solid #bce8f1',
                'color': '#31708f',
                'fontSize': '12px',
                'marginBottom': '20px',
                'textAlign': 'left'
            }),
            
            html.Label("中心线夹角 α (°)", style={'marginTop': '10px'}),
            dcc.Slider(id='slider-alpha', min=45, max=150, step=0.5, value=95, marks={i: str(i) for i in range(45, 151, 10)}, updatemode='drag'),
            
            html.Label("F-ISO A (RA, mm)", style={'marginTop': '10px'}),
            dcc.Slider(id='slider-RA', min=400, max=700, step=1, value=600, marks={i: str(i) for i in range(400, 701, 50)}, updatemode='drag'),
            
            html.Label("F-ISO B (RB, mm)", style={'marginTop': '10px'}),
            dcc.Slider(id='slider-RB', min=400, max=700, step=1, value=600, marks={i: str(i) for i in range(400, 701, 50)}, updatemode='drag'),
            
            html.Label("FDD (mm)", style={'marginTop': '10px'}),
            dcc.Slider(id='slider-FDD', min=800, max=1200, step=1, value=1100, marks={i: str(i) for i in range(800, 1201, 50)}, updatemode='drag'),
            
            html.Label("SFOV A (mm)", style={'marginTop': '10px'}),
            dcc.Slider(id='slider-SFOV_A', min=400, max=600, step=1, value=500, marks={i: str(i) for i in range(400, 601, 25)}, updatemode='drag'),
            
            html.Label("SFOV B (mm)", style={'marginTop': '10px'}),
            dcc.Slider(id='slider-SFOV_B', min=300, max=600, step=1, value=350, marks={i: str(i) for i in range(300, 601, 25)}, updatemode='drag'),
            
            html.Label("Z轴覆盖 (at ISO, mm)", style={'marginTop': '10px', 'fontWeight': 'bold', 'color': 'blue'}),
            dcc.Slider(id='slider-Z-coverage', min=10, max=160, step=1, value=80, marks={i: str(i) for i in range(0, 161, 20)}, updatemode='drag'),
            
            html.Label("旋转时间 (s)", style={'marginTop': '10px', 'fontWeight': 'bold', 'color': 'red'}),
            dcc.Slider(id='slider-rotation-time', min=0.2, max=0.5, step=0.01, value=0.28, marks={i/100: str(i/100) for i in range(20, 51, 5)}, updatemode='drag'),

            html.Div([
                dcc.Checklist(
                    id='check-asymmetric',
                    options=[{'label': ' 非对称扇区模式 (Asymmetric Mode)', 'value': 'on'}],
                    value=[],
                    style={'marginTop': '20px', 'fontWeight': 'bold', 'color': '#D35400', 'fontSize': '16px'}
                )
            ])

        ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top', 'backgroundColor': '#f9f9f9'}),

        # 结果显示区域
        html.Div([
            dcc.Graph(id='3d-geometry', style={'height': '45vh', 'marginBottom': '10px'}),
            dcc.Graph(id='3d-fov-analysis', style={'height': '45vh'}),
            html.Div(id='output-results', style={'padding': '10px', 'borderTop': '1px solid #ccc', 'marginTop': '10px'})
        ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0px'}),
    ]),
])

# --- 4. Dash 回调函数（实现逻辑联动）---

@app.callback(
    [Output('output-results', 'children'),
     Output('3d-geometry', 'figure'),
     Output('3d-fov-analysis', 'figure')],
    [Input('slider-alpha', 'value'),
     Input('slider-RA', 'value'),
     Input('slider-RB', 'value'),
     Input('slider-FDD', 'value'),
     Input('slider-SFOV_A', 'value'),
     Input('slider-SFOV_B', 'value'),
     Input('slider-Z-coverage', 'value'),
     Input('slider-rotation-time', 'value'),
     Input('check-asymmetric', 'value')]
)
def update_simulation(alpha, RA, RB, FDD, SFOV_A, SFOV_B, Z_coverage, rotation_time, asymmetric_val):
    is_asymmetric = 'on' in (asymmetric_val or [])
    results, fan_angle_A, fan_angle_B = calculate_geometry(alpha, RA, RB, FDD, SFOV_A, SFOV_B, Z_coverage, rotation_time, is_asymmetric)
    
    if "error" in results:
        results_div = html.Div([
            html.P("❌ 几何错误: {}".format(results["error"]), style={'color': 'red', 'fontWeight': 'bold'})
        ])
        fig = create_3d_figure(results, FDD)
        fig_fov = go.Figure()
        return results_div, fig, fig_fov

    # 格式化结果文本
    min_dist_color = 'green' if results['is_safe'] else 'red'
    arc_ok_color = 'green' if results['is_arc_ok'] else 'red'
    
    # Asymmetric Info
    asym_info = html.Div()
    if is_asymmetric:
        asym_info = html.Div([
            html.Hr(),
            html.P("非对称模式影响 (Negative Impact):", style={'color': 'red', 'fontWeight': 'bold'}),
            html.P("1. 受影响FOV体积: {:.1f} cm³ (Outer Shell)".format(results["affected_volume_cm3"])),
            html.P("2. 有效 SFOV B: {:.1f} mm (Inner: {:.1f} mm)".format(results["SFOV_B_effective"], results["SFOV_B_inner"])),
            html.P("3. 散射干扰系数: {:.2f} (因体积增加)".format(results["scatter_coeff"])),
        ])

    results_div = html.Div([
        html.Div([
            html.P("A 系统扇区角: {:.2f}°".format(results["fan_angle_A"])),
            html.P("B 系统扇区角: {:.2f}°".format(results["fan_angle_B"])),
            html.P("A 探测器高度: {:.1f} mm".format(results["H_det_A"])),
            html.P("B 探测器高度: {:.1f} mm".format(results["H_det_B"])),
        ], style={'display': 'inline-block', 'width': '45%'}),
        
        html.Div([
            html.P([
                "最小安全距离: {:.1f} mm ".format(results["min_dist"]),
                html.Span(" (安全 ✅)" if results['is_safe'] else " (碰撞预警 ⚠️)", style={'color': min_dist_color, 'fontWeight': 'bold'})
            ]),
            html.P("DetA - DetB 间距: {:.1f} mm".format(results["min_dist_Det_Det"])),
            html.P([
                "弧长差: {:.1f} mm ".format(results["arc_diff"]),
                html.Span(" (OK)" if results['is_arc_ok'] else " (Diff < 120)", style={'color': arc_ok_color, 'fontWeight': 'bold'})
            ]),
            html.P("物理时间分辨: {:.0f} ms".format(results["temporal_resolution"] * 1000), style={'fontWeight': 'bold', 'color': '#0074D9'}),
            html.P("散射干扰系数: {:.2f}".format(results["scatter_coeff"]), style={'fontWeight': 'bold', 'color': '#FF851B'}),
            asym_info,
            html.Hr(),
            html.P("离心力 (G-Force):", style={'fontWeight': 'bold'}),
            html.P("Tube A: {:.1f} G | Tube B: {:.1f} G".format(results["g_force_TubeA"], results["g_force_TubeB"])),
            html.P("Det A: {:.1f} G | Det B: {:.1f} G".format(results["g_force_DetA"], results["g_force_DetB"])),
            html.P("系统固定压力 (双球管合力): {:.0f} N ({:.1f} G)".format(results["F_total_mag"], results["g_force_total"]), style={'color': 'purple', 'fontWeight': 'bold'}),
        ], style={'display': 'inline-block', 'width': '45%', 'verticalAlign': 'top'}),
    ])
    
    # 创建 3D 图形
    fig = create_3d_figure(results, FDD)
    fig_fov = create_fov_analysis_figure(results, FDD)
    
    return results_div, fig, fig_fov

# --- 5. 运行应用 ---

if __name__ == '__main__':
    import webbrowser
    from threading import Timer
    
    def open_browser():
        webbrowser.open_new("http://127.0.0.1:8050/")

    Timer(1, open_browser).start()
    app.run(debug=False)
