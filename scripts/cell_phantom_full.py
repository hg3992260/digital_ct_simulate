# -*- coding: utf-8 -*-
"""15 µm 细胞体模 · 全链路仿真 + 多算法重建 + 客观评估（含出图）。

链路：体模(μ,δ) → M1 正向投影 → M4 K 帧采集 → 泊松噪声 → M5 重建 → 评估

体模（15 µm 尺度）：
  · 背景 白质
  · 10 个**细胞体，直径 15 µm**（灰质值）—— 这就是被检测的目标
  · 10 个细胞核，直径 5 µm（致密）—— 亚采样特征
  · 8 段髓鞘，40 × 8 µm（相位对比最强）

评估指标：
  · 逐像素 CNR（细胞体 vs 背景）—— 注意 15 µm 采样下目标仅约 1 像素
  · **理想观察者 SNR**（预白化匹配滤波）—— 处理亚像素信号的正规工具
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, r'I:\dsw\_gh_repo\src')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import ct_forward as F
import ct_detector as CD
from skimage.transform import iradon

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei UI', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

E_KEV, R_MM = 33.0, 500.0
N0 = 1.0e7
MU = {k: CD.mu_from_beta(CD.delta_beta(k, E_KEV)[1], E_KEV)
      for k in ('white_matter', 'gray_matter', 'myelin')}
DL = {k: CD.delta_beta(k, E_KEV)[0] for k in ('white_matter', 'gray_matter', 'myelin')}
MU_WM, MU_GM, MU_MY = MU['white_matter'], MU['gray_matter'], MU['myelin']
D_WM, D_GM, D_MY = DL['white_matter'], DL['gray_matter'], DL['myelin']
MU_NU, D_NU = MU_WM * 1.030, D_WM * 1.045

# 特征几何（mm）
R_CELL, R_NUC = 0.0075, 0.0025          # 15 µm 细胞体 / 5 µm 核
MY_A, MY_B = 0.020, 0.004               # 髓鞘 40 × 8 µm


def build(field_mm, bg_r_mm=None, seed=7):
    """背景盘半径必须**小于视场**，否则物体被探测器截断 → 重建出现巨大亮环
    （截断伪影），把真实结构完全淹没。"""
    rng = np.random.default_rng(seed)
    R = field_mm / 2.0
    BG = float(bg_r_mm) if bg_r_mm else R * 1.15
    e_mu = [(MU_WM, 0, 0, BG, BG, 0.0)]
    e_de = [(D_WM, 0, 0, BG, BG, 0.0)]
    cells, feats = [], []
    for _ in range(10):
        for _t in range(300):
            x, y = rng.uniform(-R + R_CELL, R - R_CELL, 2)
            if all((x - a) ** 2 + (y - b) ** 2 > (R_CELL + r + 0.06 * R) ** 2
                   for a, b, r in feats):
                break
        else:
            continue
        feats.append((x, y, R_CELL)); cells.append((x, y))
        a = R_CELL / R
        e_mu.append((MU_GM - MU_WM, x, y, a, a, 0.0))
        e_de.append((D_GM - D_WM, x, y, a, a, 0.0))
        b = R_NUC / R
        e_mu.append((MU_NU - MU_GM, x, y, b, b, 0.0))
        e_de.append((D_NU - D_GM, x, y, b, b, 0.0))
    myelin = []
    for _ in range(8):
        x, y = rng.uniform(-R * 0.8, R * 0.8, 2)
        ang = rng.uniform(0, 180)
        myelin.append((x, y, ang))
        e_mu.append((MU_MY - MU_WM, x, y, MY_A / R, MY_B / R, ang))
        e_de.append((D_MY - D_WM, x, y, MY_A / R, MY_B / R, ang))
    return e_mu, e_de, cells, myelin


def mask_cells(field_mm, n, vox_um, cells, frac=1.0):
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    X, Y = (xx - c) * vox_um / 1000.0, (yy - c) * vox_um / 1000.0
    m = np.zeros((n, n), dtype=bool)
    for x, y in cells:
        m |= (X - x) ** 2 + (Y - y) ** 2 <= (R_CELL * frac) ** 2
    return m


def run(pe, K, field, mech, seed=1, n_ang=360, n_cols=128, dose=N0):
    p = pe / 1000.0
    fov = n_cols * p
    ang = np.linspace(0, 180, n_ang, endpoint=False)
    e_mu, e_de, cells, _ = build(field, bg_r_mm=0.45 * fov)
    if mech == 'abs':
        s = F.analytic_sinogram(e_mu, n_cols, p, ang, 1.0, 1.0)
    elif mech == 'tie':
        _, s = F.phase_contrast_sinogram(e_mu, e_de, ang, n_cols, p, 1.0, E_KEV, R_MM)
    else:
        _, _, s = F.phase_paganin_sinogram(e_mu, e_de, ang, n_cols, p, 1.0, E_KEV,
                                           R_MM, D_WM, CD.delta_beta('white_matter', E_KEV)[1])
    n = n_cols * K
    rt = lambda sn: iradon(sn.T, theta=ang, circle=False, output_size=n,
                           preserve_range=True).astype(np.float64)
    rec = rt(F.add_poisson_noise(s, dose, seed=seed))
    rec2 = rt(F.add_poisson_noise(s, dose, seed=seed + 1))
    clean = rt(s)
    vox = pe / K
    m = mask_cells(field, n, vox, cells)
    bg = (np.hypot(*np.mgrid[0:n, 0:n][::-1]) if False else None)
    c0 = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    rad = np.hypot(xx - c0, yy - c0) * vox / 1000.0
    bgm = (rad < field * 0.48) & (~m)
    cnr = None
    if m.any() and bgm.any():
        d = abs(float(rec[m].mean()) - float(rec[bgm].mean()))
        nn = float(np.std(rec - rec2) / np.sqrt(2.0))
        cnr = d / nn if nn > 0 else None
    return dict(rec=rec, clean=clean, truth=None, vox=vox, cnr=cnr,
                cells=cells, field=field, pe=pe, K=K, mech=mech)


def ideal_snr(pe, K, field, mech, seed=1, n_ang=360, n_cols=128, dose=N0):
    """理想观察者 SNR：信号取无噪重建之差，NPS 取差值法估计。"""
    p = pe / 1000.0
    fov = n_cols * p
    ang = np.linspace(0, 180, n_ang, endpoint=False)
    e_mu, e_de, cells, _ = build(field, bg_r_mm=0.45 * fov)
    # 背景-only 体模（去掉细胞与核，保留髓鞘不动 → 只检测细胞）
    e_mu_bg = [e_mu[0]] + [e for e in e_mu if abs(e[0] - (MU_MY - MU_WM)) < 1e-12]
    e_de_bg = [e_de[0]] + [e for e in e_de if abs(e[0] - (D_MY - D_WM)) < 1e-12]

    def sino(em, ed):
        if mech == 'abs':
            return F.analytic_sinogram(em, n_cols, p, ang, 1.0, 1.0)
        if mech == 'tie':
            return F.phase_contrast_sinogram(em, ed, ang, n_cols, p, 1.0, E_KEV, R_MM)[1]
        return F.phase_paganin_sinogram(em, ed, ang, n_cols, p, 1.0, E_KEV, R_MM,
                                        D_WM, CD.delta_beta('white_matter', E_KEV)[1])[2]
    s_sig, s_bg = sino(e_mu, e_de), sino(e_mu_bg, e_de_bg)
    n = n_cols * K
    rt = lambda sn: iradon(sn.T, theta=ang, circle=False, output_size=n,
                           preserve_range=True).astype(np.float64)
    sig = rt(s_sig) - rt(s_bg)
    acc = np.zeros((n, n))
    for i in range(4):
        acc += np.abs(np.fft.fft2(rt(F.add_poisson_noise(s_bg, dose, seed=seed + 2 * i))
                                  - rt(F.add_poisson_noise(s_bg, dose, seed=seed + 2 * i + 1)))) ** 2
    nps = np.maximum(acc / 8.0, 1e-30)
    fy = np.fft.fftfreq(n)[:, None]; fx = np.fft.fftfreq(n)[None, :]
    fr = np.sqrt(fy ** 2 + fx ** 2); nb = max(8, n // 24)
    idx = np.clip((fr / 0.5 * nb).astype(int), 0, nb - 1)
    for b in range(nb):
        mm = idx == b
        if mm.any():
            nps[mm] = nps[mm].mean()
    return float(np.sqrt(max(np.sum(np.abs(np.fft.fft2(sig)) ** 2 / nps), 0.0)))


print('=' * 92)
print('15 µm 细胞体模 · 全链路仿真     E=%.0f keV   R=%.0f mm   N₀=%.0e 光子/像素' % (E_KEV, R_MM, N0))
print('=' * 92)
CONF = [('15µm 采样 K=1', 15.0, 1, 1.44), ('15µm 采样 K=2', 15.0, 2, 1.44),
        ('5µm 采样 K=2', 5.0, 2, 0.48)]
MECHS = [('abs', '纯衰减'), ('tie', '相衬 TIE'), ('paganin', 'Paganin')]
rows = []
print('%-14s %-10s %8s %10s %12s %14s' % ('采样', '机制', '体素µm', 'CNR', '理想SNR', 'SNR=3 需光子'))
print('-' * 92)
data = {}
for tag, pe, K, field in CONF:
    for mech, mname in MECHS:
        r = run(pe, K, field, mech)
        snr = ideal_snr(pe, K, field, mech)
        need = N0 * (3.0 / snr) ** 2 if snr > 0 else float('inf')
        data[(tag, mname)] = (r, snr, need)
        rows.append((tag, mname, r['vox'], r['cnr'], snr, need))
        print('%-14s %-10s %8.2f %10s %12.4f %14.3e'
              % (tag, mname, r['vox'],
                 ('%.3f' % r['cnr']) if r['cnr'] else 'n/a', snr, need))

# ---------------- 出图 ----------------
fig, axes = plt.subplots(3, 4, figsize=(16, 12.6), facecolor='#0f1216')
for ax in axes.ravel():
    ax.set_facecolor('#0f1216')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#2b343d')
truth_img = data[(CONF[0][0], '纯衰减')][0]
e_mu, e_de, cells, _ = build(truth_img['field'], bg_r_mm=0.45 * 128 * 0.015)

def win(img, frac=0.55):
    """按**中心区域**取显示窗：整图百分位会被外边界/截断残留主导，
    把真实的微小细胞结构压成一片黑。"""
    n = img.shape[0]
    h = max(4, int(n * frac / 2))
    c = n // 2
    p = img[c - h:c + h, c - h:c + h]
    return float(np.percentile(p, 2)), float(np.percentile(p, 98))

def truth_at(n, vox, field, bg_r):
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    X, Y = (xx - c) * vox / 1000.0, (yy - c) * vox / 1000.0
    ref = np.zeros((n, n))
    for A, x0, y0, a, b, phi in e_mu:
        if A == MU_WM:
            continue
        t = np.deg2rad(phi); ct, st = np.cos(t), np.sin(t)
        dx, dy = X - x0, Y - y0
        u, w = dx * ct + dy * st, -dx * st + dy * ct
        ref[(u / a) ** 2 + (w / b) ** 2 <= 1] += A
    return ref

# 列0：参考体模（把细胞/核/髓鞘突出显示）
for i, (tag, pe, K, field) in enumerate(CONF):
    ax = axes[i][0]
    vox = pe / K
    n = data[(tag, '纯衰减')][0]['rec'].shape[0]
    ref = truth_at(n, vox, field, 0.45 * 128 * pe / 1000.0)
    v = np.abs(ref); v = v / (v.max() or 1)
    ax.imshow(v, cmap='inferno', interpolation='nearest')
    ax.set_title('参考体模 · %s\n体素 %.1f um（目标：15 um 细胞体）' % (tag, vox),
                 color='#e6edf3', fontsize=10)

for j, (mech, mname) in enumerate(MECHS):
    for i, (tag, pe, K, field) in enumerate(CONF):
        r = data[(tag, mname)][0]
        snr = data[(tag, mname)][1]
        ax = axes[i][j + 1]
        lo, hi = win(r['rec'])
        ax.imshow(r['rec'], cmap='gray', vmin=lo, vmax=hi, interpolation='bilinear')
        ax.set_title('%s · %s\n体素 %.1f um   理想SNR %.3f'
                     % (tag, mname, r['vox'], snr), color='#e6edf3', fontsize=10)

fig.suptitle('N0=1e7 · 15 um 细胞尺度数字体模 · 全链路仿真（M1 正向投影 -> M4 K 帧采集 -> M5 重建）',
             color='#e6edf3', fontsize=15, y=0.985)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = r'I:\dsw\_shots\cell_phantom_algorithms.png'
plt.savefig(out, dpi=110, facecolor='#0f1216')
print()
print('图已保存:', out)
json.dump([[t, m, v, c, s, nd] for (t, m, v, c, s, nd) in rows],
          open(r'I:\dsw\_ci\cell_metrics.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
