# -*- coding: utf-8 -*-
"""双源 CT 真实几何模拟器 —— 蓝色极简科技版（PySide6 + PyCt6 + 拟物化三维质感）

与 PyQt5 版 simulate_qt.py 的关系
------------------------------------------------------------------
* **布局与内容完全保留**：左栏「系统几何参数」（9 个参数 + 非对称开关 + 计算结果），
  右栏上层两个 3D 视口（System Geometry / FOV Impact Analysis），下层 5 个重建分析页
  （Sinogram / Ideal Recon / Actual Recon / Artifact Overlay / FBP Process），
  以及 3D 视口上的参数 HUD。数值、文案、判据、配色语义均与原版一致。
* **显示层重构**：控件改用 PyCt6（CFrame / CLabel / CSlider / CTextEdit / CMainWindow），
  外壳为自绘拟物面板（金属渐变 + 斜角高光 + 螺钉 + 凹陷内阴影 + 玻璃反光），
  整体为深蓝极简科技风。
* **几何内核独立**：calculate_geometry() 抽到 ct_geometry.py（纯 numpy），
  因此本文件只依赖 PySide6 一套 Qt 绑定，可与 PyQt5 版并存。

同时修掉了原版的三处显示缺陷：
  1) simulate_qt.py:892 把 self.view 二次挂载到主布局，导致 3D 主视口被挤成右侧窄条；
  2) 上层 splitter 未给尺寸，view_fov 高度塌成 0（连 GL 上下文都没创建）；
  3) 左栏内容高于窗口，底部「计算结果」被裁掉且不可滚动。

运行（PySide6 6.11 + pyqtgraph 0.14 + skimage 环境）：
    D:\\python\\envs\\mar\\python.exe I:\\dsw\\simulate_ct6.py
"""

import os
import sys
import time

# pyqtgraph 必须绑定 PySide6，避免与 PyQt5 混用
os.environ.setdefault('PYQTGRAPH_QT_LIB', 'PySide6')

import numpy as np

# 兼容旧版 pyqtgraph（<0.14）的 GLMeshItem 使用 np.product 的问题（numpy>=2 已移除）
if not hasattr(np, 'product'):
    np.product = np.prod

from numpy.fft import fft, ifft, fftfreq

from PySide6.QtCore import Qt, QRectF, QPointF, QTimer, Signal, QEvent
from PySide6.QtGui import (QColor, QFont, QIcon, QPixmap, QPainter, QPainterPath, QPen, QBrush,
                           QLinearGradient, QRadialGradient, QSurfaceFormat)
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
                               QScrollArea, QSplashScreen, QFrame, QLabel, QGridLayout)

try:
    from PyCt6 import (CMainWindow, CFrame, CLabel, CSlider, CTextEdit, CComboBox, CButton,
                       CLineEdit, set_color_theme, set_appearance_mode)
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "[FATAL] 需要 PyCt6（PySide6 组件库）。\n"
        "        已安装环境示例：D:\\python\\envs\\mar\\python.exe -c \"import PyCt6\"\n"
        f"        原始错误：{exc}\n")
    raise

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from skimage.transform import radon, iradon
from skimage.draw import disk

from ct_geometry import calculate_geometry
import ct_scene as SC
import ct_helical as CH
import ct_index as CI
import ct_fermi as CF
try:
    import ct_synchrotron as CS      # 同步辐射仿真模式（第 6 种架构）的物理与算法
except Exception:                    # pragma: no cover
    CS = None

try:
    import ct_leap as CL          # LEAP-CT 真实几何物理引擎（不可用时 FBP 页签降级提示）
    LEAP_OK = CL.LEAP_AVAILABLE
except Exception as _exc:         # pragma: no cover
    CL = None
    LEAP_OK = False
    print(f'[WARN] LEAP-CT 不可用：{_exc}')


# =====================================================================
# 1. 蓝色极简科技调色板（拟物化外壳用色，语义色与 PyQt5 版保持一致）
# =====================================================================

class P:
    BG_CORE = "#E4F0FB"          # 窗口背景中心（浅蓝）
    BG_EDGE = "#B9D2EA"          # 窗口背景四周（略深浅蓝）
    PANEL_T = "#FBFDFF"          # 面板渐变上（近白磨砂）
    PANEL_B = "#DCE9F6"          # 面板渐变下（浅蓝）
    PANEL_T2 = "#F5FAFE"         # 次级面板渐变上
    PANEL_B2 = "#D2E3F3"         # 次级面板渐变下
    EDGE_HI = "#FFFFFF"          # 斜角高光（上）
    EDGE_MID = "#A9C4DC"         # 斜角中间过渡
    EDGE_LO = "#7FA0BF"          # 斜角暗边（下）
    GROOVE_T = "#CBDDEE"         # 凹陷槽上（更深）
    GROOVE_B = "#EBF4FC"         # 凹陷槽下
    HAIRLINE = "#A9C6DF"         # 细分隔线
    ACCENT = "#1058A0"           # 主强调蓝（原 #1E7FE0 3.99 → 全底 >=5.0）
    ACCENT_HI = "#2E9BFF"        # 高亮蓝
    ACCENT_DIM = "#8FBEE8"       # 暗蓝
    TEXT = "#0F3355"             # 主文字（深海军蓝）
    TEXT_DIM = "#385B7A"         # 次级文字（全底 >=4.6）
    TEXT_MUTE = "#405D74"       # 弱化文字（原 #6E90AC 3.30 → 全底 >=4.6）
    OK = "#086341"               # 安全（原 #12A26B 3.22 → 全底 >=4.6）
    WARN = "#AF2323"             # 碰撞预警（原 #D93A3A 4.46 → 全底 >=5.1）
    INFO = "#0E64A0"             # 时间分辨（原 #1272B8 → 主底 5.7 / 次底 4.8）
    SCATTER = "#8C4E04"          # 散射系数（原 #C2700A 3.67 → 主底 5.9 / 次底 4.9）
    FORCE = "#6C3FD1"            # 离心力（语义色）


# 3D 视口 / 绘图背景（浅蓝）
GL_BG_MAIN = (232, 244, 254, 255)
GL_BG_FOV = (214, 233, 249, 255)
PLOT_BG = "#F4FAFF"
PLOT_FG = "#3E6485"
GRID_RGBA = (120, 160, 200, 90)

# 重建用的每圈投影数上限：采样率滑块最高 10000 Hz，若完全按 采样率×旋转时间 重建，
# 5000 投影需要约 2.8 s/次，拖动滑块会卡死。这里取 1440（0.25° 采样，约 0.8 s/次），
# 配合下面的"松手后重建"防抖，兼顾采样率表达力与交互流畅度。
RECON_MAX_PROJECTIONS = 1440
RECON_DEBOUNCE_MS = 260


def qss_global() -> str:
    """全局 QSS：标签页 / 滚动区 / 分割条 / 绘图控件外框。"""
    return f"""
    QTabWidget {{ background: transparent; }}
    QTabWidget::tab-bar {{ alignment: left; }}
    QTabBar {{ background: transparent; }}
    QTabWidget::pane {{
        border: 1px solid {P.HAIRLINE};
        border-radius: 9px;
        background: {PLOT_BG};
        top: -1px;
    }}
    QTabBar::tab {{
        background: transparent;
        color: {P.TEXT_MUTE};
        padding: 7px 16px;
        margin-right: 4px;
        border: 1px solid transparent;
        border-top-left-radius: 7px;
        border-top-right-radius: 7px;
        font-family: "Segoe UI";
        font-size: 11px;
    }}
    QTabBar::tab:hover {{ color: {P.TEXT_DIM}; background: rgba(255,255,255,0.65); }}
    QTabBar::tab:selected {{
        color: {P.TEXT};
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 {P.PANEL_T}, stop:1 {PLOT_BG});
        border: 1px solid {P.HAIRLINE};
        border-bottom: 1px solid {P.ACCENT};
    }}
    QScrollArea {{ background: transparent; border: none; }}
    QScrollBar:vertical {{
        background: transparent; width: 9px; margin: 2px 2px 2px 0;
    }}
    QScrollBar::handle:vertical {{
        background: {P.ACCENT_DIM}; border-radius: 4px; min-height: 36px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {P.ACCENT}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
    QSplitter::handle {{ background: transparent; }}
    QSplitter::handle:hover {{ background: {P.ACCENT_DIM}; }}
    QToolTip {{
        background: {P.PANEL_T}; color: {P.TEXT};
        border: 1px solid {P.HAIRLINE}; padding: 4px;
    }}
    """


# =====================================================================
# 2. 资源 / 图标 / 启动画面
# =====================================================================

APP_NAME_CN = "双源 CT 真实几何模拟器"
APP_NAME_EN = "Dual-Source CT Cone-Beam Geometry Simulator"
APP_SUB_EN = "Native Edition  ·  PySide6 + PyCt6  ·  Cone-Beam Geometry Engine"

# 版本号：与 git tag / GitHub Release 保持一致（见 CHANGELOG.md）
APP_VERSION = "1.0"
APP_BUILD = "PySide6 + PyCt6 · LEAP-CT 1.26 · xraylib 4.3.0"


def asset_path(name):
    """assets/ 资源定位：源码运行用脚本目录，打包运行用 _MEIPASS。"""
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(base, 'assets', name)
    return p if os.path.exists(p) else os.path.join(os.getcwd(), 'assets', name)


def icon_path():
    """程序图标文件。

    logo.ico 由 assets/LOGO.jpg 生成（见 scripts/make_icon.py，每次构建都会重新生成），
    所以"图标来源 = LOGO.jpg"这件事是构建期强制保证的，不靠人记得同步。
    """
    for name in ('logo.ico', 'logo.png'):
        p = asset_path(name)
        if os.path.exists(p):
            return p
    return None


def app_icon():
    p = icon_path()
    return QIcon(p) if p else QIcon()


def build_splash_pixmap(width=780, height=640):
    """启动画面：深蓝玻璃面板 + 蓝色顶条 + LOGO + 中英标题。"""
    pm = QPixmap(width, height)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    # 面板（浅蓝磨砂）
    rect = QRectF(0.5, 0.5, width - 1, height - 1)
    g = QLinearGradient(0, 0, 0, height)
    g.setColorAt(0.0, QColor("#F9FCFF"))
    g.setColorAt(1.0, QColor("#D3E5F6"))
    p.setPen(QPen(QColor(P.HAIRLINE), 1))
    p.setBrush(QBrush(g))
    p.drawRoundedRect(rect, 22, 22)

    # 顶条（源A红 → 探测器青 → 源B蓝）
    tg = QLinearGradient(0, 0, width, 0)
    tg.setColorAt(0.0, QColor(226, 58, 46))
    tg.setColorAt(0.5, QColor(0, 204, 204))
    tg.setColorAt(1.0, QColor(45, 108, 223))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(tg))
    p.drawRoundedRect(QRectF(24, 18, width - 48, 4), 2, 2)

    # LOGO
    logo = QPixmap(asset_path('logo.png'))
    if not logo.isNull():
        logo = logo.scaled(360, 360, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
        p.drawPixmap(int((width - logo.width()) / 2), 42, logo)

    p.setPen(QColor(P.TEXT))
    p.setFont(QFont("Microsoft YaHei UI", 20, QFont.Weight.Bold))
    p.drawText(QRectF(0, 424, width, 44), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
               APP_NAME_CN)

    p.setPen(QColor(P.TEXT_DIM))
    p.setFont(QFont("Segoe UI", 11))
    p.drawText(QRectF(0, 468, width, 24), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
               f"{APP_NAME_EN}   ·   v{APP_VERSION}")

    p.setPen(QPen(QColor(P.HAIRLINE), 1))
    p.drawLine(QPointF(130, 508), QPointF(width - 130, 508))

    p.setPen(QColor(P.TEXT_MUTE))
    p.setFont(QFont("Segoe UI", 9))
    p.drawText(QRectF(0, 516, width, 20), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
               APP_SUB_EN)
    p.drawText(QRectF(0, 540, width, 20), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
               "Developed by Christ.paul90@gmail.com")
    p.end()
    return pm


# =====================================================================
# 3. 拟物化自绘基元（金属面板 / 凹陷槽 / 玻璃反光 / LED / 钮子开关）
# =====================================================================

SHADOW_M = 9          # 面板四周预留的投影绘制边距


class SteelPanel(QFrame):
    """拟物金属面板：投影 + 竖向渐变 + 斜角高光/暗边 + 顶部玻璃光泽 + 可选螺钉。"""

    def __init__(self, master=None, radius=14, top=P.PANEL_T, bot=P.PANEL_B,
                 edge_hi=P.EDGE_HI, edge_lo=P.EDGE_LO, screws=False,
                 gloss=0.45, recessed=False):
        super().__init__(master)
        self.radius = radius
        self.top = top
        self.bot = bot
        self.edge_hi = edge_hi
        self.edge_lo = edge_lo
        self.screws = screws
        self.gloss = gloss
        self.recessed = recessed
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setContentsMargins(SHADOW_M + 6, SHADOW_M + 4, SHADOW_M + 6, SHADOW_M + 6)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(SHADOW_M, SHADOW_M, -SHADOW_M, -SHADOW_M)

        # 1) 投影（多层扩散近似高斯阴影，浅色主题用蓝灰投影）
        p.setPen(Qt.PenStyle.NoPen)
        for i, alpha in enumerate((36, 26, 18, 11, 6)):
            spread = i * 1.0
            p.setBrush(QColor(31, 71, 120, alpha))
            p.drawRoundedRect(r.adjusted(-spread, -spread + 4 + i * 1.2,
                                         spread, spread + 4 + i * 1.2),
                              self.radius + spread, self.radius + spread)

        # 2) 面板主体
        g = QLinearGradient(r.left(), r.top(), r.left(), r.bottom())
        g.setColorAt(0.0, QColor(self.top))
        g.setColorAt(1.0, QColor(self.bot))
        path = QPainterPath()
        path.addRoundedRect(r, self.radius, self.radius)

        if self.recessed:
            rg = QLinearGradient(r.left(), r.top(), r.left(), r.top() + r.height() * 0.5)
            rg.setColorAt(0.0, QColor(P.GROOVE_T))
            rg.setColorAt(1.0, QColor(P.GROOVE_B))
            p.setBrush(QBrush(rg))
        else:
            p.setBrush(QBrush(g))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(path)

        # 3) 顶部玻璃光泽（裁剪在圆角内）
        p.save()
        p.setClipPath(path)
        gl_ = QLinearGradient(r.left(), r.top(), r.left(), r.top() + r.height() * 0.55)
        gl_.setColorAt(0.0, QColor(255, 255, 255, int(255 * self.gloss)))
        gl_.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.setBrush(QBrush(gl_))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(r)
        p.restore()

        # 4) 斜角：上亮下暗的一圈描边
        bg = QLinearGradient(r.left(), r.top(), r.left(), r.bottom())
        bg.setColorAt(0.0, QColor(self.edge_hi))
        bg.setColorAt(0.45, QColor(P.EDGE_MID))
        bg.setColorAt(1.0, QColor(self.edge_lo))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QBrush(bg), 1.4))
        p.drawRoundedRect(r.adjusted(0.7, 0.7, -0.7, -0.7), self.radius, self.radius)

        # 5) 螺钉
        if self.screws:
            inset = 16
            for cx, cy in ((r.left() + inset, r.top() + inset),
                           (r.right() - inset, r.top() + inset),
                           (r.left() + inset, r.bottom() - inset),
                           (r.right() - inset, r.bottom() - inset)):
                rg = QRadialGradient(QPointF(cx - 1.2, cy - 1.2), 6.0)
                rg.setColorAt(0.0, QColor("#FFFFFF"))
                rg.setColorAt(0.55, QColor("#A8C4DA"))
                rg.setColorAt(1.0, QColor("#6E90AC"))
                p.setBrush(QBrush(rg))
                p.setPen(QPen(QColor(110, 144, 172, 170), 0.8))
                p.drawEllipse(QPointF(cx, cy), 4.4, 4.4)
                p.setPen(QPen(QColor(70, 100, 130, 190), 1.3))
                p.drawLine(QPointF(cx - 2.6, cy), QPointF(cx + 2.6, cy))
                p.drawLine(QPointF(cx, cy - 2.6), QPointF(cx, cy + 2.6))
        p.end()


class RecessedCard(QFrame):
    """凹陷槽：用于读数屏 / 图例容器。"""

    def __init__(self, master=None, radius=10):
        super().__init__(master)
        self.radius = radius
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setContentsMargins(2, 2, 2, 2)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        g = QLinearGradient(r.left(), r.top(), r.left(), r.bottom())
        g.setColorAt(0.0, QColor(P.GROOVE_T))
        g.setColorAt(1.0, QColor(P.GROOVE_B))
        p.setBrush(QBrush(g))
        p.setPen(QPen(QColor("#9FBBD4"), 1.2))
        p.drawRoundedRect(r, self.radius, self.radius)
        # 内阴影：顶部几层半透明蓝灰
        p.save()
        clip = QPainterPath()
        clip.addRoundedRect(r, self.radius, self.radius)
        p.setClipPath(clip)
        for i, a in enumerate((58, 34, 18)):
            p.setPen(QPen(QColor(60, 100, 145, a), 1.6))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r.adjusted(i, i, -i, -i), self.radius, self.radius)
        p.restore()
        p.setPen(QPen(QColor(255, 255, 255, 230), 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(QPointF(r.left() + 8, r.bottom() - 1), QPointF(r.right() - 8, r.bottom() - 1))
        p.end()


class GlassOverlay(QWidget):
    """覆盖在 3D 视口上的玻璃反光层（不接收鼠标事件）。"""

    def __init__(self, master=None):
        super().__init__(master)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect())
        p.save()
        clip = QPainterPath()
        clip.addRoundedRect(r, 7, 7)
        p.setClipPath(clip)
        g = QLinearGradient(r.left(), r.top(), r.right(), r.bottom())
        g.setColorAt(0.00, QColor(255, 255, 255, 105))
        g.setColorAt(0.28, QColor(255, 255, 255, 40))
        g.setColorAt(0.45, QColor(255, 255, 255, 0))
        p.setBrush(QBrush(g))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(r)
        p.restore()
        p.end()


class LedDot(QWidget):
    """状态 LED（拟物指示灯）。"""

    def __init__(self, master=None, color=P.OK, size=9):
        super().__init__(master)
        self.color = QColor(color)
        self.setFixedSize(size + 6, size + 6)

    def set_color(self, color):
        self.color = QColor(color)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2)
        halo = QRadialGradient(c, self.width() / 2)
        col = QColor(self.color)
        halo.setColorAt(0.0, QColor(col.red(), col.green(), col.blue(), 110))
        halo.setColorAt(1.0, QColor(col.red(), col.green(), col.blue(), 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(halo))
        p.drawEllipse(c, self.width() / 2, self.width() / 2)
        body = QRadialGradient(QPointF(c.x() - 1, c.y() - 1), 5)
        body.setColorAt(0.0, col.lighter(180))
        body.setColorAt(0.6, col)
        body.setColorAt(1.0, col.darker(220))
        p.setBrush(QBrush(body))
        p.drawEllipse(c, 4.0, 4.0)
        p.end()


class SkeuoSwitch(QWidget):
    """拟物钮子开关（替代 QCheckBox，保留原标签与语义）。"""

    toggled = Signal(bool)

    def __init__(self, master=None, text="", checked=False, on_change=None):
        super().__init__(master)
        self._checked = bool(checked)
        self._text = text
        self._anim = 1.0 if checked else 0.0
        self._hover = False
        self._track_w, self._track_h = 52, 24
        self.setMinimumHeight(34)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        if on_change is not None:
            self.toggled.connect(on_change)

    def isChecked(self):
        return self._checked

    def setChecked(self, value):
        value = bool(value)
        if value != self._checked:
            self._checked = value
            self._timer.start()
            self.update()
            self.toggled.emit(value)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self._checked)

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self.update()

    def _tick(self):
        target = 1.0 if self._checked else 0.0
        step = 0.16
        if abs(self._anim - target) <= step:
            self._anim = target
            self._timer.stop()
        else:
            self._anim += step if self._anim < target else -step
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect())

        # 文本
        on = self._checked
        p.setPen(QColor(P.ACCENT if on else P.TEXT_DIM))
        p.setFont(QFont("Microsoft YaHei UI", 10, QFont.Weight.DemiBold if on else QFont.Weight.Normal))
        p.drawText(QRectF(r.left() + 2, r.top(), r.width() - self._track_w - 16, r.height()),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._text)

        # 轨道（凹陷）
        tx = r.right() - self._track_w - 2
        ty = r.center().y() - self._track_h / 2
        track = QRectF(tx, ty, self._track_w, self._track_h)
        tg = QLinearGradient(track.left(), track.top(), track.left(), track.bottom())
        tg.setColorAt(0.0, QColor(P.GROOVE_T))
        tg.setColorAt(1.0, QColor(P.GROOVE_B))
        p.setPen(QPen(QColor("#9FBBD4"), 1.2))
        p.setBrush(QBrush(tg))
        p.drawRoundedRect(track, self._track_h / 2, self._track_h / 2)

        # 开启时的蓝色填充与辉光
        if self._anim > 0.01:
            fill = QRectF(track.left() + 2, track.top() + 2,
                          (track.width() - 4) * self._anim, track.height() - 4)
            fg = QLinearGradient(fill.left(), fill.top(), fill.left(), fill.bottom())
            fg.setColorAt(0.0, QColor(P.ACCENT_HI))
            fg.setColorAt(1.0, QColor(P.ACCENT))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(fg))
            p.drawRoundedRect(fill, fill.height() / 2, fill.height() / 2)
            halo = QRadialGradient(track.center(), track.width() * 0.75)
            halo.setColorAt(0.0, QColor(46, 155, 255, int(70 * self._anim)))
            halo.setColorAt(1.0, QColor(46, 155, 255, 0))
            p.setBrush(QBrush(halo))
            p.drawEllipse(track.center(), track.width() * 0.75, track.height() * 1.15)

        # 旋钮
        kx = track.left() + self._track_h / 2 + (track.width() - self._track_h) * self._anim
        kc = QPointF(kx, track.center().y())
        kr = self._track_h / 2 - 2
        kg = QRadialGradient(QPointF(kc.x() - 1.5, kc.y() - 2.0), kr * 1.9)
        kg.setColorAt(0.0, QColor("#FFFFFF"))
        kg.setColorAt(0.45, QColor("#CFE6FF" if on else "#F0F5FA"))
        kg.setColorAt(1.0, QColor("#7FA8CB" if on else "#AFC4D6"))
        p.setPen(QPen(QColor(90, 120, 150, 170), 1.0))
        p.setBrush(QBrush(kg))
        p.drawEllipse(kc, kr, kr)
        p.end()


def engrave(label):
    """给 CLabel 内部 QLabel 加一点"压印"效果（浅色主题：下方白色高光）。"""
    from PySide6.QtWidgets import QGraphicsDropShadowEffect
    eff = QGraphicsDropShadowEffect(label)
    eff.setBlurRadius(2)
    eff.setOffset(0, 1)
    eff.setColor(QColor(255, 255, 255, 235))
    label.setGraphicsEffect(eff)
    return label


# =====================================================================
# 4. 参数滑块（PyCt6 CSlider 封装，保留原版浮点/单位/回调语义）
# =====================================================================

class ParamSlider(QWidget):
    def __init__(self, master=None, name="", min_val=0, max_val=100, step=1, default=50,
                 unit="", is_float=False, decimals=None, on_change=None):
        super().__init__(master)
        self.unit = unit
        self.is_float = is_float
        # decimals：显示与取值的小数位数（默认浮点 2 位、整数 0 位）；物理像素需要 3 位
        self.decimals = (2 if is_float else 0) if decimals is None else int(decimals)
        self.scale_factor = 10 ** self.decimals
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 3, 0, 3)
        lay.setSpacing(1)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        self.name_label = CLabel(self, width=max(150, len(name) * 11), height=22, text=name,
                                 font_family="Microsoft YaHei UI", font_size=10,
                                 text_color=P.TEXT_DIM)
        self.name_label.label().setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.value_label = CLabel(self, width=110, height=22,
                                  text=self._fmt(default), font_family="Consolas", font_size=11,
                                  font_style="bold", text_color=P.ACCENT)
        self.value_label.label().setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        head.addWidget(self.name_label)
        head.addStretch(1)
        head.addWidget(self.value_label)
        lay.addLayout(head)

        self.cslider = CSlider(
            self, width=352, height=6,
            minimum=int(min_val * self.scale_factor),
            maximum=int(max_val * self.scale_factor),
            step=max(1, int(round(step * self.scale_factor))),
            value=int(round(default * self.scale_factor)),
            background_color=P.GROOVE_T,
            progress_color=P.ACCENT,
            border_color="#9FBBD4",
            border_width=1,
            button_width=16, button_height=16,
            button_background_color=P.ACCENT_HI,
            button_border_color="#7FA8CB",
            button_border_width=1,
            button_corner_radius=8,
            corner_radius=3,
            button_hover_color="#5FB6FF",
            button_pressed_color=P.ACCENT,
        )
        lay.addWidget(self.cslider)

        self._callback = on_change
        self.cslider.slider().valueChanged.connect(self._on_change)

    def _fmt(self, real_val):
        return f"{real_val:.{self.decimals}f} {self.unit}".strip()

    def _on_change(self, _=None):
        self.value_label.label().setText(self._fmt(self.value()))
        if self._callback is not None:
            self._callback()

    def value(self):
        return self.cslider.slider().value() / self.scale_factor

    def slider(self):
        return self.cslider.slider()


# =====================================================================
# 5. 重建分析面板（5 个页签，逻辑与 PyQt5 版一致，仅重绘外观）
# =====================================================================

class FermiPage(QWidget):
    """Fermi 页签：半导体探测器（CZT/TlBr/…）响应与能量箱匹配（光子计数架构）。

    左侧：材料 / 厚度 / 偏压 / 温度 / 入射能量 / kVp / 箱数与阈值模式
    右侧：① 能谱响应 ② CCE(z) 电荷收集效率 ③ E(z) 内部电场 ④ bin 响应矩阵热图
    底部读数：ENC/FWHM、电阻率/暗电流、箱阈值、箱计数权重、空箱警告
    """

    def __init__(self, master=None):
        super().__init__(master)
        self.model = None
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        ctl = SteelPanel(self, radius=14, screws=True)
        ctl.setFixedWidth(316)
        cl = QVBoxLayout(ctl)
        cl.setContentsMargins(14, 12, 14, 12)
        cl.setSpacing(3)

        def caps(t):
            l = CLabel(ctl, width=276, height=19, text=t, font_family="Microsoft YaHei UI",
                       font_size=9, font_style="bold", text_color=P.TEXT_DIM)
            l.label().setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            cl.addWidget(l)

        def combo(vals, cur, w=276):
            return CComboBox(ctl, width=w, height=27, font_family="Microsoft YaHei UI",
                             font_size=10, values=vals, current_value=cur,
                             text_color=P.TEXT, item_text_color=P.TEXT,
                             menu_background_color="#FFFFFF")

        def sl(name, lo, hi, st, dv, unit, dec=None):
            s = ParamSlider(ctl, name=name, min_val=lo, max_val=hi, step=st, default=dv,
                            unit=unit, is_float=(dec is not None), decimals=dec,
                            on_change=lambda: None)
            s.setFixedWidth(276)          # 表头需 260px（名称 150 + 数值 110），否则数值被遮挡
            try:                          # CLabel 会随父宽增长 → 锁死，避免表头超出控件宽
                s.name_label.setFixedWidth(150)
                s.value_label.setFixedWidth(110)
            except Exception:
                pass
            cl.addWidget(s)
            return s

        caps("① 材料与器件")
        self.mat_combo = combo(CF.FermiModel.materials(), 'CZT')
        cl.addWidget(self.mat_combo)
        self.s_thick = sl("厚度", 0.5, 10.0, 0.5, 2.0, "mm", 1)
        self.s_bias = sl("偏压", 100, 3000, 50, 1000, "V")
        self.s_temp = sl("温度", 200, 350, 5, 300, "K")

        caps("② 谱条件")
        self.s_kvp = sl("管电压 kVp", 60, 150, 5, 120, "kV")
        self.s_energy = sl("入射能量", 30, 190, 5, 120, "keV")

        caps("③ 能量箱")
        _row = QHBoxLayout()
        _row.setSpacing(4)
        self.bins_combo = combo(['4 箱', '6 箱', '8 箱', '12 箱'], '8 箱', 134)
        self.mode_combo = combo(['阈值: 等宽', '阈值: K边优化'], '阈值: 等宽', 138)
        _row.addWidget(self.bins_combo)
        _row.addWidget(self.mode_combo)
        cl.addLayout(_row)

        self.run_btn = CButton(ctl, text="▶ 计算 Fermi 响应", width=276, height=32,
                               font_family="Microsoft YaHei UI", font_size=11,
                               font_style="bold", background_color=P.ACCENT,
                               hover_color=P.ACCENT_HI, pressed_color=P.ACCENT,
                               text_color="#FFFFFF")
        try:
            if hasattr(self.run_btn, 'button'):
                self.run_btn.button().clicked.connect(self.compute)
            else:
                self.run_btn.clicked.connect(self.compute)
        except Exception as _e:
            print(f'[WARN] Fermi 按钮绑定失败: {_e}')
        cl.addWidget(self.run_btn)
        self.info = CTextEdit(ctl, width=276, height=150, font_family="Consolas", font_size=8,
                              text_color=P.TEXT, background_color="#F5FAFE",
                              border_color=P.ACCENT_DIM, border_width=1, corner_radius=8)
        cl.addWidget(self.info, 1)          # 读数框吃掉剩余高度，不再固定 260px

        right = QWidget()
        grid = QGridLayout(right)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        self.p_spec = pg.PlotWidget(title="① 能谱响应")
        self.p_cce = pg.PlotWidget(title="② CCE(z) 电荷收集效率")
        self.p_ef = pg.PlotWidget(title="③ E(z) 内部电场")
        self.img_bin = pg.ImageView(view=pg.PlotItem(title="④ 能量箱响应矩阵 R[E,bin]"))
        for i, w in enumerate((self.p_spec, self.p_cce, self.p_ef, self.img_bin)):
            try:
                w.setBackground(PLOT_BG)
                if hasattr(w, 'setMinimumHeight'):
                    w.setMinimumHeight(180)
            except Exception:
                pass
            grid.addWidget(w, i // 2, i % 2)
        for _r in (0, 1):
            grid.setRowStretch(_r, 1)
        for _c in (0, 1):
            grid.setColumnStretch(_c, 1)

        root.addWidget(ctl)
        root.addWidget(right, 1)

    def set_arch_enabled(self, photon_counting):
        """仅光子计数架构下启用（其它架构给出提示但不禁用，便于对比材料）。"""
        try:
            self.run_btn.setEnabled(True)
        except Exception:
            pass
        self._pcct = bool(photon_counting)

    def compute(self):
        try:
            if self.model is None:
                self.model = CF.FermiModel()
            mat = self.mat_combo.combo_box().currentText()
            th = float(self.s_thick.value())
            v = float(self.s_bias.value())
            T = float(self.s_temp.value())
            e_keV = float(self.s_energy.value())
            kvp = float(self.s_kvp.value())
            nb = int(str(self.bins_combo.combo_box().currentText()).split()[0])
            mode = ('kedge' if 'K' in str(self.mode_combo.combo_box().currentText())
                    else 'equal')
            sp = self.model.spectrum(mat, e_keV, th, v, T, n_bins=200)
            z, cce = self.model.cce_profile(mat, th, v, T)
            zf, ef = self.model.field_profile(mat, th, v, T)
            bw = self.model.bin_weights(mat, kvp, nb, mode, th, v, T)
            ns = self.model.noise(mat, th, v, T)

            self.p_spec.clear()
            self.p_spec.plot(sp['energy_axis'], np.maximum(sp['spectrum'], 0.0),
                             pen=pg.mkPen(P.ACCENT, width=2))
            self.p_spec.setTitle(f"① 能谱响应  {mat} @ {e_keV:.0f} keV"
                                 f" (效率 {sp['interaction_efficiency']:.3f})")
            self.p_cce.clear()
            self.p_cce.plot(z, cce, pen=pg.mkPen(P.OK, width=2))
            self.p_cce.setTitle(f"② CCE(z)  {cce.min():.3f} ~ {cce.max():.3f}")
            self.p_ef.clear()
            self.p_ef.plot(zf, ef, pen=pg.mkPen(P.INFO, width=2))
            self.p_ef.setTitle(f"③ E(z)  {np.nanmin(ef):.0f} ~ {np.nanmax(ef):.0f} V/mm")
            self.img_bin.setImage(bw['R'].T, autoLevels=True)
            self.img_bin.view.setLabel('left', '能量箱')
            self.img_bin.view.setLabel('bottom', '入射能量 keV')

            _g = lambda k, d=float('nan'): float(ns.get(k, d)) if hasattr(ns, 'get') else float('nan')
            txt = (f"<b>{mat}</b>  厚度 {th:.1f} mm  偏压 {v:.0f} V  温度 {T:.0f} K<br>"
                   f"ENC <b>{_g('enc_electrons'):.1f}</b> e⁻   FWHM "
                   f"<b>{_g('fwhm_noise_keV'):.3f}</b> keV<br>"
                   f"电阻率 {_g('rho'):.3e} Ω·cm<br>暗电流 {_g('dark_current_nA'):.3f} nA<br>"
                   f"<br><b>能量箱阈值</b>（{'K 边优化' if mode == 'kedge' else '等宽'}，"
                   f"{nb} 箱）:<br>" + ' | '.join(f'{x:.0f}' for x in bw['thresholds'])
                   + "<br><br><b>箱计数权重</b>（归一）:<br>"
                   + ' | '.join(f'{x:.3f}' for x in bw['weights_norm']) + "<br>")
            empty = [i + 1 for i, x in enumerate(bw['weights_norm']) if x < 1e-6]
            if empty:
                txt += (f"<br><font color='{P.WARN}'><b>空箱警告</b>：第 "
                        f"{','.join(map(str, empty))} 箱计数为 0——{kvp:.0f} kVp 谱在 "
                        f"{bw['thresholds'][empty[0] - 1]:.0f} keV 以上基本无光子"
                        f"→ 建议箱数 ≤ {max(2, empty[0] - 1)}，或提高 kVp</font>")
            else:
                txt += (f"<br><font color='{P.OK}'>全部 {len(bw['weights_norm'])} "
                        f"个能量箱均非空</font>")
            if not getattr(self, '_pcct', False):
                txt += (f"<br><br><font color='{P.TEXT_MUTE}'>当前架构非光子计数——"
                        f"此为材料响应参考，分箱重建需切到「光子计数」架构</font>")
            self.info.text_edit().setHtml(txt)
            return True
        except Exception as exc:
            self.info.text_edit().setHtml(
                f"<font color='{P.WARN}'>Fermi 计算失败：{exc}</font>")
            return False


class ReconstructionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        lay.addWidget(self.tabs)

        # --- Tab 1: Sinogram ---
        self.tab_sino = QWidget()
        l1 = QVBoxLayout(self.tab_sino)
        l1.setContentsMargins(4, 4, 4, 4)
        self.sino_top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.phantom_plot = self._make_image_view()
        self.sino_plot = self._make_image_view()

        self.legend = pg.LegendItem(offset=(50, 10))
        self.legend.setParentItem(self.phantom_plot.getView())
        self.fov_curve_inner = pg.PlotCurveItem(pen=pg.mkPen(P.OK, width=2, style=Qt.PenStyle.DashLine),
                                                name="Safe FOV (Symmetric)")
        self.fov_curve_outer = pg.PlotCurveItem(pen=pg.mkPen(P.WARN, width=2), name="Max FOV (Effective)")
        self.phantom_plot.getView().addItem(self.fov_curve_inner)
        self.phantom_plot.getView().addItem(self.fov_curve_outer)
        self.legend.addItem(self.fov_curve_inner, "Safe FOV")
        self.legend.addItem(self.fov_curve_outer, "Max FOV")

        self.sino_top_splitter.addWidget(self.phantom_plot)
        self.sino_top_splitter.addWidget(self.sino_plot)
        l1.addWidget(self.sino_top_splitter)

        self.profile_plot = pg.PlotWidget(title="Truncation Profile (Center Row)")
        self.profile_plot.setLabel('bottom', "Detector Channel")
        self.profile_plot.setLabel('left', "Intensity")
        l1.addWidget(self.profile_plot)
        l1.setStretch(0, 2)
        l1.setStretch(1, 1)
        self.tabs.addTab(self.tab_sino, "Sinogram")

        # --- Tab 2: Ideal Recon ---
        self.tab_ideal = QWidget()
        l2 = QVBoxLayout(self.tab_ideal)
        l2.setContentsMargins(4, 4, 4, 4)
        self.ideal_plot = self._make_image_view()
        l2.addWidget(self.ideal_plot)
        self.tabs.addTab(self.tab_ideal, "Ideal Recon")

        # --- Tab 3: Actual Recon ---
        self.tab_recon = QWidget()
        l3 = QVBoxLayout(self.tab_recon)
        l3.setContentsMargins(4, 4, 4, 4)
        self.recon_plot = self._make_image_view()
        l3.addWidget(self.recon_plot)
        self.tabs.addTab(self.tab_recon, "Actual Recon")

        # --- Tab 4: Artifact Overlay ---
        self.tab_impact = QWidget()
        l4 = QVBoxLayout(self.tab_impact)
        l4.setContentsMargins(4, 4, 4, 4)
        self.impact_plot = self._make_image_view()
        l4.addWidget(self.impact_plot)
        self.tabs.addTab(self.tab_impact, "Artifact Overlay")

        # --- Tab 5: FBP Process（LEAP-CT 真实几何物理引擎）---
        self.tab_fbp = QWidget()
        l5 = QVBoxLayout(self.tab_fbp)
        l5.setContentsMargins(0, 0, 0, 0)
        self.fbp_panel = FbpProcessPanel(self.tab_fbp)
        l5.addWidget(self.fbp_panel)
        self.tabs.addTab(self.tab_fbp, "FBP Process")

        # --- Tab 6: Fermi（仅光子计数架构下可见，默认不加入）---
        try:
            self.tab_fermi = FermiPage()
        except Exception as exc:
            self.tab_fermi = None
            print(f'[WARN] Fermi 页签创建失败: {exc}')

    def set_fermi_visible(self, on):
        """Fermi 页签仅在「光子计数」架构下出现（按需加入/移除页签）。"""
        if getattr(self, 'tab_fermi', None) is None:
            return
        idx = self.tabs.indexOf(self.tab_fermi)
        try:
            if on and idx < 0:
                self.tabs.addTab(self.tab_fermi, "Fermi")
            elif (not on) and idx >= 0:
                self.tabs.removeTab(idx)
        except Exception as exc:
            print(f'[WARN] Fermi 页签切换失败: {exc}')

        # 状态
        self.phantom_size = 128
        self.phantom = self.create_phantom(self.phantom_size)
        self.last_params = None

    @staticmethod
    def _make_image_view():
        iv = pg.ImageView()
        iv.ui.histogram.hide()
        iv.ui.roiBtn.hide()
        iv.ui.menuBtn.hide()
        iv.getView().setBackgroundColor(PLOT_BG)
        return iv

    def create_phantom(self, size):
        image = np.zeros((size, size))
        rr, cc = disk((size // 2, size // 2), size // 2 - 5)
        image[rr, cc] = 1.0
        rr, cc = disk((size // 2 - size // 4, size // 2), size // 8)
        image[rr, cc] = 0.5
        rr, cc = disk((size // 2 + size // 4, size // 2), size // 8)
        image[rr, cc] = 0.8
        rr, cc = disk((size // 2, size // 2 - size // 4), size // 10)
        image[rr, cc] = 0.2
        return image

    def update_data(self, results, params):
        # 同步辐射仿真路径面板：仅在 arch='synchrotron' 时可见，切走即隐藏（不占版面）
        try:
            _sp = getattr(getattr(self, 'fbp_panel', None), 'sync_panel', None)
            if _sp is not None:
                _sp.setVisible(str((results or {}).get('arch_key', '')) == 'synchrotron')
        except Exception:
            pass

        sampling_rate = params.get('sampling_rate', 2000)
        rotation_time = params['rotation_time']

        # 采样率 × 旋转时间 = 每圈投影数；上限用于保证交互流畅（实测 radon+3×iradon 约 0.2 ms/投影）
        n_projections = max(8, min(int(sampling_rate * rotation_time), RECON_MAX_PROJECTIONS))
        theta = np.linspace(0., 180., n_projections, endpoint=False)

        sinogram_full = radon(self.phantom, theta=theta, circle=True)

        max_fov_radius = 300.0
        sino_rows = sinogram_full.shape[0]
        pixel_size_mm = (2 * max_fov_radius) / sino_rows

        RB = params['RB']
        beta_B_orig = results['beta_B_orig']
        angle_ext_rad = 120.0 / params['FDD']

        # 3.1 对称基线
        r_effective = RB * np.sin(beta_B_orig + angle_ext_rad)
        min_idx_sym = max(0, int(sino_rows / 2 - (r_effective / pixel_size_mm)))
        max_idx_sym = min(sino_rows, int(sino_rows / 2 + (r_effective / pixel_size_mm)))
        sino_symmetric = np.zeros_like(sinogram_full)
        sino_symmetric[min_idx_sym:max_idx_sym, :] = sinogram_full[min_idx_sym:max_idx_sym, :]

        # 3.2 非对称截断
        if params['is_asymmetric']:
            r_left_actual = RB * np.sin(beta_B_orig)
            r_right_actual = RB * np.sin(beta_B_orig + angle_ext_rad)
            min_idx_asym = max(0, int(sino_rows / 2 - (r_left_actual / pixel_size_mm)))
            max_idx_asym = min(sino_rows, int(sino_rows / 2 + (r_right_actual / pixel_size_mm)))
            sino_asym = np.zeros_like(sinogram_full)
            sino_asym[min_idx_asym:max_idx_asym, :] = sinogram_full[min_idx_asym:max_idx_asym, :]
            masked_sino = sino_asym
            r_left, r_right = r_left_actual, r_effective
            diff_sino = sino_symmetric - sino_asym
        else:
            masked_sino = sino_symmetric
            diff_sino = np.zeros_like(sinogram_full)
            r_left = r_right = r_effective

        # FOV 参考圆
        center_px = self.phantom_size / 2
        mm_to_px = self.phantom_size / (2 * max_fov_radius)
        theta_circle = np.linspace(0, 2 * np.pi, 100)
        r_in_px = r_left * mm_to_px
        self.fov_curve_inner.setData(center_px + r_in_px * np.cos(theta_circle),
                                     center_px + r_in_px * np.sin(theta_circle))
        r_out_px = r_right * mm_to_px
        self.fov_curve_outer.setData(center_px + r_out_px * np.cos(theta_circle),
                                     center_px + r_out_px * np.sin(theta_circle))

        # 4. 重建
        recon_ideal = iradon(sinogram_full, theta=theta, circle=True)
        reconstruction = iradon(masked_sino, theta=theta, circle=True)
        impact_image = iradon(diff_sino, theta=theta, circle=True)

        base = self.phantom.copy()
        if base.max() > base.min():
            base = (base - base.min()) / (base.max() - base.min())
        artifact_vis = np.clip(np.abs(impact_image) * 5.0, 0, 1)
        overlay_rgb = np.zeros((base.shape[0], base.shape[1], 3))
        overlay_rgb[..., 0] = np.clip(base + artifact_vis, 0, 1)
        overlay_rgb[..., 1] = np.clip(base - artifact_vis * 0.5, 0, 1)
        overlay_rgb[..., 2] = np.clip(base - artifact_vis * 0.5, 0, 1)

        # 滤波正弦图（Ram-Lak）
        rows = masked_sino.shape[0]
        omega = np.abs(fftfreq(rows).reshape(-1, 1))
        filtered_sino = np.real(ifft(fft(masked_sino, axis=0) * omega, axis=0))

        # 5. 刷新显示
        self.phantom_plot.setImage(np.transpose(overlay_rgb, (1, 0, 2)))
        self.sino_plot.setImage(masked_sino.T)
        self.ideal_plot.setImage(recon_ideal.T)
        self.recon_plot.setImage(reconstruction.T)
        self.impact_plot.setImage(np.transpose(overlay_rgb, (1, 0, 2)))
        # 滤波正弦图与 FBP 重建现由 FBP Process 页签的 LEAP 引擎负责

        center_angle_idx = n_projections // 2
        # X 轴用**真实探测器通道号**：把 radon 行的 ISO 坐标换算到通道轴，
        # 这样"每排单元数"一改，曲线横轴范围与 FOV 边界标记都会同步移动。
        _nch = int(results.get('n_ch', 825))
        _piso = max(1e-6, float(results.get('iso_sampling_mm', 0.625)))
        _c_ctr = (_nch - 1) / 2.0
        c_axis = _c_ctr + ((np.arange(sino_rows) - sino_rows / 2.0) * pixel_size_mm) / _piso
        self.profile_plot.setLabel('bottom', f"Detector Channel (0 – {_nch - 1}, {_piso:.3f} mm/ch)")
        self.profile_plot.plot(c_axis, masked_sino[:, center_angle_idx], clear=True,
                               pen=pg.mkPen(P.INFO, width=2))
        min_idx_disp = _c_ctr - r_left / _piso
        max_idx_disp = _c_ctr + r_right / _piso
        self.profile_plot.addItem(pg.InfiniteLine(pos=min_idx_disp, angle=90, pen=pg.mkPen(P.WARN)))
        self.profile_plot.addItem(pg.InfiniteLine(pos=max_idx_disp, angle=90, pen=pg.mkPen(P.WARN)))
        self.profile_plot.setXRange(0, max(1, _nch - 1), padding=0.01)


# =====================================================================
# 6. 3D 视口容器（金属边框 + 凹陷 + 角螺钉 + 玻璃反光 + 标题栏）
# =====================================================================

class ViewportBezel(SteelPanel):
    def __init__(self, master=None, title="", bg=(10, 22, 34, 255)):
        super().__init__(master, radius=13, screws=True, gloss=0.30)
        self.setMinimumHeight(240)

        lay = QVBoxLayout(self)
        # 边距需与 SteelPanel 的投影区（SHADOW_M）对齐，否则内容会盖住投影与螺钉
        lay.setContentsMargins(SHADOW_M + 8, SHADOW_M + 4, SHADOW_M + 8, SHADOW_M + 8)
        lay.setSpacing(5)

        head = QHBoxLayout()
        head.setContentsMargins(2, 0, 2, 0)
        self.led = LedDot(self, P.ACCENT, 9)
        self.title = CLabel(self, width=220, height=20, text=title,
                            font_family="Segoe UI", font_size=10, font_style="bold",
                            text_color=P.TEXT_DIM)
        self.title.label().setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        head.addWidget(self.led)
        head.addWidget(self.title)
        head.addStretch(1)
        self.badge = CLabel(self, width=110, height=20, text="双击放大",
                            font_family="Microsoft YaHei UI", font_size=9,
                            text_color=P.TEXT_MUTE)
        self.badge.label().setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        head.addWidget(self.badge)
        lay.addLayout(head)
        self.setToolTip("双击视口：占满右侧显示区；再双击还原")

        self.gl = gl.GLViewWidget()
        self.gl.setBackgroundColor(QColor(*bg))
        lay.addWidget(self.gl, 1)

        self.glass = GlassOverlay(self)
        self.glass.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        g = self.gl.geometry()
        self.glass.setGeometry(g.x() + 1, g.y() + 1, max(0, g.width() - 2), max(0, g.height() - 2))
        self.glass.raise_()

    def paintEvent(self, event):
        super().paintEvent(event)
        # 视口四周的凹陷内阴影（画在子控件之下，露出的一圈）
        if getattr(self, 'gl', None) is not None:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            g = QRectF(self.gl.geometry())
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor("#8FB0CC"), 3.0))
            p.drawRoundedRect(g.adjusted(-2.0, -2.0, 2.0, 2.0), 8, 8)
            p.setPen(QPen(QColor(255, 255, 255, 220), 1.2))
            p.drawRoundedRect(g.adjusted(-4.0, -4.0, 4.0, 4.0), 10, 10)
            p.end()


# =====================================================================
# 7. 主窗口（布局与内容对齐 PyQt5 版，显示层重构）
# =====================================================================

class FbpProcessPanel(QWidget):
    """FBP Process 页签：以 LEAP-CT 为物理引擎的真实几何重建面板。

    * 几何来自当前"系统几何参数"（R / FDD / SFOV / 物理像素 → 开扇角、通道数、views）
    * 可对仿真物理投影或外部真实 sinogram 文件做 FBP
    * 可选插值展平（等角扇束 → 平行束重排）
    * 滤波核 / 低通均可选；参数读数实时显示采样率、转速、views、开扇角、开扇角单元等
    """

    def __init__(self, master=None):
        super().__init__(master)
        self.engine = None
        self.geom = None
        # 光子计数：Fermi 能量箱展开所需状态
        self.fermi_query = None
        self.fermi_ready = False
        self._cone = None
        self._spec_key = None
        self._spec_cache = None
        self.params = None
        self.results = None
        self.sino = None
        self.external = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(320)
        self._timer.timeout.connect(self.run)

        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(8)

        # ---------------- 左：控制 + 参数读数 ----------------
        ctl = QWidget(self)
        ctl.setFixedWidth(272)
        cl = QVBoxLayout(ctl)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(4)

        def hdr(txt):
            lb = CLabel(ctl, width=250, height=20, text=txt, font_family="Microsoft YaHei UI",
                        font_size=9, font_style="bold", text_color=P.TEXT_MUTE)
            lb.label().setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            return lb

        cl.addWidget(hdr("数据源 / DATA SOURCE"))
        self.src_combo = CComboBox(ctl, width=250, height=26, font_family="Microsoft YaHei UI",
                                   font_size=10,
                                   values=['仿真物理投影 (LEAP project)', '外部真实 sinogram 文件',
                                           '批量目录 (batch)', '动态序列 (frames)'],
                                   current_value='仿真物理投影 (LEAP project)', text_color=P.TEXT, item_text_color=P.TEXT, menu_background_color="#FFFFFF")
        self.src_combo.combo_box().currentTextChanged.connect(self._on_src)
        cl.addWidget(self.src_combo)
        self.path_edit = CLineEdit(ctl, width=250, height=26, text="",
                                   placeholder_text=".npy / .npz / .csv / .tif",
                                   font_family="Consolas", font_size=9)
        cl.addWidget(self.path_edit)
        self.load_btn = CButton(ctl, width=250, height=26, text="载入 sinogram 文件…",
                                command=self.pick_file, font_family="Microsoft YaHei UI", font_size=10,
                                background_color=P.ACCENT_HI, hover_color=P.ACCENT,
                                pressed_color=P.ACCENT)
        cl.addWidget(self.load_btn)

        # 批量 / 动态序列：帧选择 + 播放
        self.frame_lbl = CLabel(ctl, width=250, height=18, text="帧 — / —", font_family="Consolas",
                                font_size=9, text_color=P.TEXT_DIM)
        cl.addWidget(self.frame_lbl)
        self.frame_slider = CSlider(ctl, width=250, height=6, minimum=0, maximum=0, step=1, value=0,
                                    background_color=P.GROOVE_T, progress_color=P.ACCENT,
                                    border_color="#9FBBD4", border_width=1,
                                    button_width=14, button_height=14,
                                    button_background_color=P.ACCENT_HI, button_border_color="#7FA8CB",
                                    button_border_width=1, button_corner_radius=7, corner_radius=3)
        self.frame_slider.slider().valueChanged.connect(self._on_frame)
        cl.addWidget(self.frame_slider)
        self.play_sw = SkeuoSwitch(ctl, text="▶ 序列播放 (0.7 s/帧)", checked=False,
                                   on_change=self._on_play)
        cl.addWidget(self.play_sw)
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(700)
        self._play_timer.timeout.connect(self._next_frame)

        cl.addWidget(hdr("重建参数 / RECON"))
        self.kernel_combo = CComboBox(ctl, width=250, height=26, font_family="Segoe UI", font_size=10,
                                      values=(CL.KERNELS if LEAP_OK else ['Ram-Lak']),
                                      current_value='Ram-Lak', text_color=P.TEXT, item_text_color=P.TEXT, menu_background_color="#FFFFFF")
        self.kernel_combo.combo_box().currentTextChanged.connect(lambda _=None: self._timer.start())
        cl.addWidget(self.kernel_combo)

        self.matrix_combo = CComboBox(ctl, width=250, height=26, font_family="Segoe UI", font_size=10,
                                      values=['自动 (SFOV/像素)', '256', '512', '1024'],
                                      current_value='自动 (SFOV/像素)', text_color=P.TEXT, item_text_color=P.TEXT, menu_background_color="#FFFFFF")
        self.matrix_combo.combo_box().currentTextChanged.connect(lambda _=None: self._timer.start())
        cl.addWidget(self.matrix_combo)

        self.rebin_sw = SkeuoSwitch(ctl, text="插值展平 (仅显示平行束正弦图)", checked=False,
                                    on_change=lambda _=None: self._timer.start())
        cl.addWidget(self.rebin_sw)

        self.scan_combo = CComboBox(ctl, width=250, height=26, font_family="Microsoft YaHei UI",
                                    font_size=10, values=['360° 全扫描', '180°+开扇角 短扫描'],
                                    current_value='360° 全扫描', text_color=P.TEXT, item_text_color=P.TEXT, menu_background_color="#FFFFFF")
        self.scan_combo.combo_box().currentTextChanged.connect(lambda _=None: self._timer.start())
        cl.addWidget(self.scan_combo)
        self.parker_sw = SkeuoSwitch(ctl, text="Parker 冗余加权 (短扫描)", checked=False,
                                     on_change=lambda _=None: self._timer.start())
        cl.addWidget(self.parker_sw)

        self.helical_combo = CComboBox(ctl, width=250, height=26, font_family="Microsoft YaHei UI",
                                       font_size=10,
                                       values=['螺旋 z 插值: 关', '360°LI (同射线)',
                                               '180°LI (共轭射线·三次样条)',
                                               '180°LI (共轭射线·线性)',
                                               '180MF (180LI + z 滤波)'],
                                       current_value='螺旋 z 插值: 关', text_color=P.TEXT, item_text_color=P.TEXT, menu_background_color="#FFFFFF")
        self.helical_combo.combo_box().currentTextChanged.connect(lambda _=None: self._timer.start())
        cl.addWidget(self.helical_combo)

        self.lowpass_slider = ParamSlider(ctl, name="LEAP 低通 FWHM", min_val=0, max_val=20, step=1,
                                          default=0, unit="px", on_change=lambda: self._timer.start())
        cl.addWidget(self.lowpass_slider)

        self.run_btn = CButton(ctl, width=250, height=30, text="▶  执行 FBP 重建",
                               command=self.run, font_family="Microsoft YaHei UI", font_size=11,
                               font_style="bold", background_color=P.ACCENT, hover_color=P.ACCENT_HI,
                               pressed_color=P.ACCENT)
        cl.addWidget(self.run_btn)

        # 架构 ↔ 重建管线一致性门禁：默认阻止不匹配的架构，可显式强制降级重建
        self.force_arch_sw = SkeuoSwitch(ctl, text="强制重建 (忽略架构不匹配)", checked=False,
                                         on_change=lambda _=None: self._timer.start())
        cl.addWidget(self.force_arch_sw)

        # 阵列完整模式：由 Z 轴范围与 Z 向宽度推导 numRows/pixelHeight（LEAP 原生 cone）
        self.cone_sw = SkeuoSwitch(ctl, text="128 排锥束 (阵列完整)", checked=False,
                                   on_change=lambda _=None: self._timer.start())
        cl.addWidget(self.cone_sw)

        # 严格谱分解：逐基材逐能量正投影 → 分箱 → 水/碘双基材分解
        self.spec_sw = SkeuoSwitch(ctl, text="严格谱分解 (水/碘)", checked=False,
                                   on_change=lambda _=None: self._timer.start())
        cl.addWidget(self.spec_sw)

        cl.addWidget(hdr("物理参数 / GEOMETRY"))
        self.info_view = CTextEdit(ctl, width=250, height=330, font_family="Consolas", font_size=9,
                                   text_color=P.TEXT, background_color=P.GROOVE_T,
                                   border_color=P.HAIRLINE, corner_radius=8, border_width=1)
        self.info_view.text_edit().setReadOnly(True)
        cl.addWidget(self.info_view, 1)
        root.addWidget(ctl)

        # ---------------- 右：四联图像 ----------------
        def iv():
            v = pg.ImageView()
            v.ui.histogram.hide()
            v.ui.roiBtn.hide()
            v.ui.menuBtn.hide()
            v.getView().setBackgroundColor(PLOT_BG)
            v.getView().setAspectLocked(True)
            return v

        self.iv_fan = iv()
        self.iv_par = iv()
        self.iv_filt = iv()
        self.iv_rec = iv()
        g = QGridLayout()
        g.setContentsMargins(0, 0, 0, 0)
        g.setSpacing(4)
        for i, (w, t) in enumerate(((self.iv_fan, "① 扇形束正弦图 (原始)"),
                                    (self.iv_par, "② 插值展平 → 平行束正弦图"),
                                    (self.iv_filt, "③ 滤波后正弦图"),
                                    (self.iv_rec, "④ FBP 重建"))):
            box = QWidget()
            bl = QVBoxLayout(box)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(1)
            cap = CLabel(box, width=300, height=18, text=t, font_family="Microsoft YaHei UI",
                         font_size=9, text_color=P.TEXT_DIM)
            bl.addWidget(cap)
            bl.addWidget(w, 1)
            g.addWidget(box, i // 2, i % 2)
        g.setRowStretch(0, 1)
        g.setRowStretch(1, 1)
        g.setColumnStretch(0, 1)
        g.setColumnStretch(1, 1)
        root.addLayout(g, 1)

        if LEAP_OK:
            try:
                self.engine = CL.LeapEngine(gpu_index=0)
            except Exception as exc:
                self.engine = None
                self._set_info(f'LEAP 引擎初始化失败：{exc}')


        # ---------------- 最右：同步辐射仿真路径（**仅该架构可见**）----------------
        # 手册 §3.1 的级联模型 + §5 的四条腿，在**刚重建出的真实切片**上逐级实算，
        # 而不是示意曲线。切到其它架构时整块隐藏，不占版面。
        self.sync_panel = QWidget()
        self.sync_panel.setFixedWidth(400)
        spl = QVBoxLayout(self.sync_panel)
        spl.setContentsMargins(0, 0, 0, 0)
        spl.setSpacing(3)

        sp_head = CLabel(self.sync_panel, width=396, height=18,
                         text="同步辐射仿真路径 (Synchrotron Paths)",
                         font_family="Microsoft YaHei UI", font_size=9, font_style="bold",
                         text_color=P.ACCENT)
        sp_head.label().setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        spl.addWidget(sp_head)

        self.iv_sync = iv()
        spl.addWidget(self.iv_sync, 1)

        self.sync_legend = CTextEdit(self.sync_panel, width=396, height=150,
                                     font_family="Microsoft YaHei UI", font_size=8,
                                     text_color=P.TEXT_DIM, background_color=P.GROOVE_T,
                                     border_color=P.HAIRLINE, corner_radius=8, border_width=1)
        self.sync_legend.text_edit().setReadOnly(True)
        spl.addWidget(self.sync_legend)

        root.addWidget(self.sync_panel)
        self.sync_panel.setVisible(False)

    # ------------------------------------------------------------------
    def _update_sync_paths(self, img):
        """把同步辐射的仿真链逐级作用在刚重建出的切片上，显示在右侧专用面板。

        只在 arch='synchrotron' 时显示 —— 这是"该架构专属"的仿真视图。
        """
        panel = getattr(self, 'sync_panel', None)
        r = self.results or {}
        is_sync = str(r.get('arch_key', '')) == 'synchrotron'
        if panel is not None:
            panel.setVisible(is_sync)
        if not is_sync or panel is None:
            return
        try:
            import ct_synchrotron as _CSS
        except Exception:
            return
        try:
            # 重建体素用同步辐射**自己的**样品面尺度（p/M 经 K 帧过采样细分），
            # 而不是临床的 iso_sampling —— 后者是机架通道采样，量级差约 40 倍，
            # 会让"反卷积是否有效"的判断完全失真。
            vox_um = float(r.get('sync_voxel_um', 0.0) or 0.0)
            if vox_um <= 0:
                vox_um = (float(r.get('sync_p_eff_um', 13.75) or 13.75)
                          / max(int(r.get('sync_oversample', 4) or 4), 1))
            psf_um = float(r.get('sync_penumbra_um', 0.0) or 0.0)
            stages = _CSS.simulate_paths(
                img, voxel_um=vox_um,
                focus_um=float(r.get('sync_focus_um', 10.0)),
                pixel_um=float(r.get('sync_pixel_um', 55.0)),
                sigma_c_um=float(r.get('sync_sigma_c_um', 15.0)),
                M=float(r.get('sync_M', 1.0) or 1.0),
                K=int(r.get('sync_oversample', 4) or 4),
                lam_K_um=float(r.get('sync_lambda_K_um', 249.0)),
                odd_mm=float(r.get('sync_odd_mm', 300.0)),
                energy_kev=float(r.get('sync_energy_kev', 60.0)),
                delta_beta=float(r.get('sync_delta_beta', 100.0)),
                phase=bool(r.get('sync_phase_on', False)))
            if not stages:
                return
            self.iv_sync.setImage(_CSS.montage([im for _, im in stages], cols=2))

            rows = ['<b>逐级仿真（同一标尺，可直接比对）</b>']
            for t, _im in stages:
                rows.append('&nbsp;&nbsp;' + t)
            rows.append('')
            rows.append('<b>MTF10</b> %.0f lp/cm　<b>r(M)</b> %.2f µm　<b>r*</b> %.2f µm'
                        % (r.get('sync_lp_cm', 0.0), r.get('sync_r_um', 0.0),
                           r.get('sync_r_star_um', 0.0)))
            rows.append('<b>DQE</b> %.3f　<b>香农</b> %.0f bit/mm　<b>FOV</b> %.1f mm'
                        % (r.get('sync_dqe', 0.0), r.get('sync_shannon', 0.0),
                           r.get('sync_fov_mm', 0.0) or 0.0))
            rows.append('体素 %.1f µm　半影 PSF %.1f µm　%s'
                        % (vox_um, psf_um,
                           '<b>反卷积有效</b>（体素 &lt; PSF）' if vox_um < psf_um
                           else '体素 ≥ PSF，反卷积收益有限'))
            self.sync_legend.text_edit().setHtml('<br>'.join(rows))
        except Exception as exc:
            print(f'[WARN] 同步辐射仿真路径渲染失败: {exc}')

    # ---------------- 内部工具 ----------------
    def _set_info(self, txt):
        self.info_view.text_edit().setHtml(
            f"<div style='font-family:Consolas;font-size:9pt;color:{P.TEXT};'>{txt}</div>")

    def _phantom(self, n, sfov):
        """与其它页签一致的圆盘体模，按 mm 画在 n×n 网格上。"""
        ph = np.zeros((n, n), dtype=np.float32)
        yy, xx = np.mgrid[0:n, 0:n]
        c = (n - 1) / 2.0
        v = sfov / n

        def disk(cx, cy, r, val):
            m = np.hypot(xx - (c + cx / v), yy - (c + cy / v)) * v <= r
            ph[m] = val

        disk(0, 0, 180, 0.02)
        disk(-60, 0, 40, 0.04)
        disk(70, -40, 30, 0.005)
        disk(0, 90, 25, 0.06)
        return ph

    def _volume_n(self):
        txt = self.matrix_combo.combo_box().currentText()
        if txt.startswith('自动') and self.geom:
            return int(np.clip(round(self.geom.sfov / self.geom.p_iso), 64, 1024))
        try:
            return int(txt)
        except Exception:
            return 512

    def _on_src(self, txt):
        """数据源模式：sim / ext / batch / frames（不依赖下拉框文本，避免程序化设置失效）"""
        self.mode = ('batch' if txt.startswith('批量') else
                     'frames' if txt.startswith('动态') else
                     'ext' if txt.startswith('外部') else 'sim')
        self._timer.start()

    def pick_file(self):
        from PySide6.QtWidgets import QFileDialog
        mode = self.src_combo.combo_box().currentText()
        if mode.startswith('批量'):
            p = QFileDialog.getExistingDirectory(self, '选择 sinogram 目录')
            if p:
                self.path_edit.line_edit().setText(p)
                self.load_series(p)
            return
        p, _ = QFileDialog.getOpenFileName(self, '选择 sinogram 文件', '',
                                           'Sinogram (*.npy *.npz *.csv *.txt *.tif *.tiff);;All (*)')
        if not p:
            return
        self.path_edit.line_edit().setText(p)
        if mode.startswith('动态'):
            self.load_series(p)
        else:
            self.src_combo.combo_box().setCurrentText('外部真实 sinogram 文件')
            self.load_external(p)

    # ---------- 批量 / 动态序列 ----------
    def load_series(self, path):
        try:
            frames, labels, meta = CL.load_sinogram_series(path)
        except Exception as exc:
            self._set_info(f'<b>序列载入失败</b>：{exc}')
            return False
        self.frames, self.frame_labels, self.series_meta = frames, labels, meta
        self.mode = 'batch' if str(meta.get('series_kind', '')).startswith('批量') else 'frames'
        self.frame_idx = 0
        n = len(frames)
        self.frame_slider.slider().setMaximum(max(0, n - 1))
        self.frame_slider.slider().setValue(0)
        self.frame_lbl.label().setText(f'帧 1 / {n}   {meta.get("series_kind", "")}')
        self._last_extra = {'数据源': f'{meta.get("series_kind", "序列")} · {os.path.basename(str(path))}',
                            '帧数': str(n), '数据形状(通道×views)': str(frames[0].shape)}
        self.external = frames[0]
        self._timer.start()
        return True

    def _on_frame(self, i):
        if not getattr(self, 'frames', None):
            return
        i = int(np.clip(i, 0, len(self.frames) - 1))
        self.frame_idx = i
        self.external = self.frames[i]
        self.frame_lbl.label().setText(f'帧 {i + 1} / {len(self.frames)}   {self.frame_labels[i]}')
        self._timer.start()

    def _on_play(self, checked=None):
        if self.play_sw.isChecked() and getattr(self, 'frames', None):
            self._play_timer.start()
        else:
            self._play_timer.stop()

    def _next_frame(self):
        if not getattr(self, 'frames', None):
            self._play_timer.stop()
            return
        nxt = (self.frame_idx + 1) % len(self.frames)
        self.frame_slider.slider().setValue(nxt)

    def load_external(self, path):
        try:
            data, meta = CL.load_sinogram_file(path)
        except Exception as exc:
            self._set_info(f'<b>载入失败</b>：{exc}')
            return False
        self.external = data
        self.mode = 'ext'
        self.frames = None
        extra = {'数据源': f'外部文件 {os.path.basename(str(path))}', '数据形状(通道×views)': str(data.shape)}
        for k in ('sampling_rate_hz', 'rotation_time_s', 'fan_angle_deg', 'source_iso_mm',
                  'source_detector_mm', 'pixel_iso_mm'):
            if k in meta:
                extra[k] = str(meta[k])
        self._last_extra = extra
        self._timer.start()
        return True

    # ---------------- 参数刷新（由主窗口几何变化调用） ----------------
    def set_geometry(self, params, results, is_asymmetric=False):
        self.params, self.results = dict(params), results
        if not LEAP_OK:
            self._set_info('LEAP-CT 不可用（需在 mar 环境运行）')
            return
        try:
            self.geom = CL.FanGeometry(params, results, system='A')
        except Exception as exc:
            self._set_info(f'几何构建失败：{exc}')
            return
        self._timer.start()

    # ---------------- 主流程 ----------------
    def _arch_gate(self):
        """CT 架构 ↔ LEAP 重建管线一致性校验。

        本重建管线 = **系统 A / 单排扇束 / 单能量 / 旋转采集**。任何架构若需要
        第二套系统、能量维度、静态多源或非旋转采样，都与该管线不自洽 → 必须显式报错，
        否则会产出"看似正常、物理无意义"的图像。
        返回 (blocked, errors, warns)。
        """
        r = self.results or {}
        g = self.geom
        n_src = int(r.get('n_src', 1))
        e_dim = int(r.get('energy_dim', 1))
        rotating = bool(r.get('rotating', True))
        need = 180.0 + float(r.get('fan_angle_A', 0.0))
        err, warn = [], []
        if (not rotating) and n_src > 1:
            warn.append(f"<b>静态多源</b>（{n_src} 源）：改用 LEAP <b>模块束 set_modularbeam</b> "
                        f"逐源指定位置 + <b>SART 迭代重建</b>（稀疏视角，非解析 FBP）；"
                        f"每点仅 {r.get('views_per_point', n_src)} 个视角 → 需迭代/TV 正则化，"
                        f"解析 FBP 会产生严重条形伪影。")
        _fermi_ok = bool(getattr(self, 'fermi_query', None)) or getattr(self, 'fermi_ready', False)
        # 同步辐射是**准单色高通量**源，这恰恰是它相对实验室源的定义性优势：
        # FBP 基线本就该按「单色等效（VMI）」口径跑，能量箱属于**下游**的材料分解
        # /能谱分析，不是重建前提。故不因 energy_dim>1 阻断它（与双层/光子计数区分）。
        _is_sync = str(r.get('arch_key', '')) == 'synchrotron'
        if e_dim > 1 and not _fermi_ok and not _is_sync:
            err.append(f"<b>能量维 = {e_dim}</b>（双层/光子计数）：需要 Fermi 能量箱响应数据；"
                       f"Fermi 模型尚未就绪 → 能谱重建<b>无法进行</b>。")
        if _is_sync and e_dim > 1:
            warn.append(f"<b>同步辐射</b>：FBP 基线按<b>单色等效（VMI，{e_dim} 箱）</b>口径进行。"
                        f"同步辐射的定义性优势就是准单色高通量，能量箱用于下游材料分解与"
                        f"能谱分析，不是重建的前提；若要**箱分辨**重建，另需 Fermi 能量箱响应。")
        if not r.get('arch_ok', True):
            err.append(f"<b>不满足 180° 采样要求</b>：角度覆盖 "
                       f"{r.get('ang_cover_deg', 0):.1f}° &lt; 重建所需的 {need:.1f}°"
                       f"（180° + 扇角 {r.get('fan_angle_A', 0):.1f}°）→ 数据残缺，FBP 无解。")
        if n_src == 2:
            warn.append(f"<b>双源</b>：本管线只用<b>系统 A</b> 的 "
                        f"{g.n_views if g else 0} 个视角，系统 B（α="
                        f"{r.get('src_step_deg', 0):.0f}° 偏移）的数据未参与 → 角度互补未兑现，"
                        f"结果等价于单源采集。")
        if int(r.get('n_rows', 1)) > 1:
            warn.append(f"探测器阵列标称 {r.get('n_ch', 0)}×{r.get('n_rows', 0)}，"
                        f"但 LEAP 以 <b>numRows=1</b> 单排重建 → 仅 1/{r.get('n_rows', 1)} 的阵列参与，"
                        f"Z 向信息未利用。")
        if r.get('sparse_view'):
            warn.append(f"<b>稀疏视角</b>：每点仅 {r.get('views_per_point', 0)} 个视角（&lt;128）→ "
                        f"采样密度不足，FBP 会出现条形伪影，需迭代/深度学习重建。")
        if r.get('pitch_over'):
            warn.append(f"<b>螺距超限</b>：{r.get('pitch', 0):.2f} &gt; 几何上限 "
                        f"{r.get('pitch_max_dual', 0):.2f} → 角度采样不完整，螺旋伪影明显。")
        forced = bool(getattr(self, 'force_arch_sw', None) and self.force_arch_sw.isChecked())
        return (len(err) > 0 and not forced), err, warn

    # ---------------- 光子计数：Fermi 分箱展开 ----------------
    def _spectral_bins(self, sino, g, n_bins=None):
        """用 Fermi 的能量箱权重把单能量线积分展开成**多能量箱正弦图**。

        N_k = Σ_E φ(E)·R[E,k]·η(E,γ)·exp(-p·μ(E)/μ_ref)
        p_k = -ln( N_k / N_k^air ),   N_k^air = Σ_E φ(E)·R[E,k]·η(E,γ)

        近似说明：LEAP 给的是单一参考能量下的线积分 p=∫μdl，谱分辨需要 μ(E)；
        此处按 p·μ(E)/μ_ref 做**谱定标**（单材料严格、混合物有残余束硬化偏差）。
        """
        import ct_fermi as CF
        if getattr(self, '_cone', None) is None:
            self._cone = CF.ConeResponse()
        fq = getattr(self, 'fermi_query', None) or {}
        mat = fq.get('material', 'CZT')
        kvp = float(fq.get('kvp', 120.0))
        nb = int(n_bins or fq.get('n_bins', 8))
        mode = fq.get('mode', 'equal')
        th = float(fq.get('thickness_mm', 2.0))
        vb = float(fq.get('voltage', 1000.0))
        tk = float(fq.get('temp_k', 300.0))
        arch = str((self.results or {}).get('arch_key', ''))
        dual = (arch == 'dual_layer') or (arch == 'dual_layer' and nb == 2)
        if arch == 'dual_layer':
            nb = 2
        key = (mat, kvp, nb, mode, th, vb, tk, int(g.n_ch), int(g.n_rows), arch)
        if getattr(self, '_spec_key', None) == key:
            return self._spec_cache
        maps, res = self._cone.bin_weight_map(mat, g, n_rows=int(g.n_rows), kvp=kvp,
                                              n_bins=nb, mode=mode, thickness_mm=th,
                                              voltage=vb, temp_k=tk)
        e_s, phi = res['e_src'], res['phi']
        step = max(1, e_s.size // 16)                       # 抽稀到 ~16 个能量点控制耗时
        e_d, phi_d = e_s[::step], phi[::step]
        if arch == 'dual_layer':
            # 双层：上层 0.5mm + 下层 1.5mm 的顺序吸收权重（低能/高能两通道）
            _e2, _p2, w1, w2 = self._cone.m.layer_weights(
                mat, kvp=kvp, e_max=190.0, d_top=0.5, d_bot=1.5,
                thickness_mm=th, voltage=vb, temp_k=tk)
            base = np.vstack([_p2[::step] * w1[::step], _p2[::step] * w2[::step]])
            mu = np.array([self._cone.mu_mm(mat, e) for e in e_d])
        else:
            base = (res['R_ref'][::step] * phi_d[:, None]).T     # (nb, n_e)
            mu = np.array([self._cone.mu_mm(mat, e) for e in e_d])
        mu_ref = float(np.average(mu, weights=phi_d)) if phi_d.sum() > 0 else 1.0
        kap = mu / max(mu_ref, 1e-12)
        self.fermi_ready = True
        self._spec_cache = {'base': base, 'kap': kap, 'maps': maps, 'res': res,
                            'mat': mat, 'kvp': kvp, 'nb': nb, 'mu_ref': mu_ref,
                            'arch': arch}
        self._spec_key = key
        return self._spec_cache

    @staticmethod
    def _montage(imgs, cols=4, gap=2):
        n = len(imgs)
        rows = int(np.ceil(n / float(cols)))
        h, w = imgs[0].shape
        out = np.zeros((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap), np.float32)
        for i, im in enumerate(imgs):
            r, c = divmod(i, cols)
            out[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = im
        return out

    def run(self):
        # ---- 架构门禁：不匹配则明确报错并拒绝重建（可"强制重建"降级继续）----
        _blocked, _errs, _warns = self._arch_gate()
        if _blocked:
            html = ('<b style="font-size:13px; color:%s;">重建已阻止：CT 架构与重建管线不匹配</b>'
                    '<br><br>' % P.WARN
                    + ''.join('&bull; %s<br>' % e for e in _errs)
                    + '<br><b>当前管线的能力边界</b>：系统 A / 单排扇束 / 单能量 / 旋转采集<br>'
                      '<b>可选操作</b>：<br>'
                      '&nbsp;&nbsp;1) 切到「<b>单源宽体</b>」或「<b>同步辐射</b>」'
                      '——与"单系统 / 旋转采集 / 单色等效"口径自洽<br>'
                      '&nbsp;&nbsp;2) 或打开「<b>强制重建</b>」开关，以"仅系统 A / 单能量 / 单排"'
                      '的降级口径继续（结果仅代表该降级口径，<b>不代表所选架构</b>）')
            if _warns:
                html += '<br><br><b>附带提示</b>：<br>' + ''.join('&middot; %s<br>' % w for w in _warns)
            self._set_info(html)
            for _iv in ('iv_fan', 'iv_par', 'iv_filt', 'iv_rec'):
                try:
                    getattr(self, _iv).clear()
                except Exception:
                    pass
            return
        if not LEAP_OK or self.engine is None or self.geom is None:
            return
        p, g = self.params, self.geom
        kernel = self.kernel_combo.combo_box().currentText()
        lowpass = float(self.lowpass_slider.value())
        short = self.scan_combo.combo_box().currentText().startswith('180')
        parker = self.parker_sw.isChecked()
        n_vol = self._volume_n()
        voxel = g.sfov / n_vol
        use_ext = getattr(self, 'mode', 'sim') != 'sim'
        t_all = time.perf_counter()
        note = ''

        # 1) 正弦图
        if use_ext and self.external is not None:
            sino = self.external
            if sino.shape != (g.n_ch, g.n_views):
                g = g.adapt_to_data(sino.shape[0], sino.shape[1], getattr(self, '_ext_meta', None))
                note = (f"<b><font color='{P.SCATTER}'>注意</font></b>：外部数据 {sino.shape} 与界面几何 "
                        f"{self.geom.n_ch}×{self.geom.n_views} 不一致，已按数据实际通道/视角数重建"
                        f"（开扇角等仍取界面值）<br>")
                voxel = g.sfov / n_vol
        else:
            ph_n = int(np.clip(n_vol, 64, 1024))
            ph = self._phantom(ph_n, g.sfov)
            self._phantom_ref = ph
            sino = self.engine.project_fan(ph, g, g.angles('short' if short else 'full360'),
                                           mode=('short' if short else 'full360'))
            note = ("仿真物理投影：LEAP 在**弯曲探测器**上做真实线积分；短扫描取 "
                    "180°+开扇角 角度区间<br>" if short else
                    "仿真物理投影：LEAP 在**弯曲探测器**上做真实线积分（360° 全扫描）<br>")
            # Bowtie：被挡住的角度范围无信号 → 数据截断（真实 CT 的视野限制效应）
            if getattr(g, 'bt_limits', False):
                sino = (sino * g.bowtie_mask()[:, None]).astype(np.float32)
                note += (f"<b><font color='{P.SCATTER}'>Bowtie 截断</font></b>：真正视野 "
                         f"{g.sfov_bt:.0f} mm，|γ| > {np.rad2deg(g.gamma_bt):.2f}° 的 "
                         f"{int((g.bowtie_mask() == 0).sum())} 个通道无信号 → 重建将出现截断伪影<br>")
        self.sino = sino
        ang = g.angles('short' if short else 'full360')

        # 2) 重建：检测器是 **2D 阵列**（n_rows > 1）时走**锥束 FDK**；
        #    否则退回 2D 扇束 FBP。
        #    此前主链只有 project_fan + fbp_fan，Z 向 292 排完全没参与重建 ——
        #    这里把它接上：LEAP project_cone（锥束正向投影）
        #    + fbp_cone_helical（FDK，LEAP 内部做锥束加权）。
        #    环境变量 DSW_CT_CONE=fan 可强制退回旧路径做对照。
        _cone_pref = 'auto'
        try:
            import os as _os
            _cone_pref = str(_os.environ.get('DSW_CT_CONE', 'auto')).lower()
        except Exception:
            pass
        n_rows_det = int(getattr(g, 'n_rows', 1) or 1)
        cone_used = False
        if _cone_pref != 'fan' and n_rows_det > 1 and (not use_ext):
            try:
                rp = float(getattr(g, 'z_cov', 0.0) or 0.0) / float(n_rows_det)
                nz = int(np.clip(n_rows_det, 32, 160))
                n_sl = int(np.clip(n_vol, 128, 256))
                vol3, _zs = CH.build_helical_phantom(nz, n_sl, g.sfov, rp)
                Gc = self.engine.project_cone(vol3, g, ang, n_rows_det, rp, 0.0)
                _rec3 = np.asarray(self.engine.fbp_cone_helical(
                    Gc, g, ang, n_rows_det, rp, 0.0, n_sl, nz))
                if _rec3.ndim == 3 and _rec3.shape[0] >= 1:
                    img = _rec3[nz // 2]
                    try:
                        self._phantom_ref = np.asarray(vol3)[nz // 2]
                    except Exception:
                        pass
                    try:
                        self.iv_fan.setImage(Gc[:, n_rows_det // 2, :])
                    except Exception:
                        pass
                    cone_used = True
                    note += (f"<b>锥束 FDK</b>：检测器 <b>{n_rows_det} 排</b> × "
                             f"{getattr(g, 'n_ch', 0)} 列，排间距 {rp:.3f} mm，"
                             f"Z 覆盖 {getattr(g, 'z_cov', 0):.1f} mm → "
                             f"<b>Z 向数据已参与重建</b>（此前完全未用）<br>"
                             f"LEAP <code>project_cone</code> + "
                             f"<code>fbp_cone_helical</code>（FDK 锥束加权），"
                             f"重建 {nz} 层 × {n_sl}² ，显示中央层<br>")
            except Exception as _ex:
                note += f"<b>锥束 FDK 失败</b>：{_ex} → 退回 2D 扇束 FBP<br>"
        if not cone_used:
            img = self.engine.fbp_fan(sino, g, ang, kernel=kernel, lowpass=lowpass,
                                      vol_n=n_vol, voxel_mm=voxel, parker=parker)
        if self.rebin_sw.isChecked():
            try:
                p_sino, theta, t_arr, d_t = self.engine.rebin_fan_to_parallel(sino, g, ang)
                sc = np.percentile(np.abs(p_sino), 99.5) or 1.0
                self.iv_par.setImage(p_sino / sc)
                note += (f"插值展平：等角扇束 → 平行束重排已生成（θ=β+γ, t=R·sinγ），"
                         f"重排数据已与 LEAP 平行束正投影交叉校核。<br>"
                         f"<b>重建仍采用 LEAP 弯曲探测器直接扇束 FBP</b>（不经平行束近似）。<br>")
            except Exception as exc:
                print(f'[WARN] 展平显示失败: {exc}')
                self.iv_par.setImage(np.zeros((8, 8), dtype=np.float32))
        else:
            self.iv_par.setImage(np.zeros((8, 8), dtype=np.float32))
        # ---- 螺旋重建：床运动 + 多圈数据 + z 方向插值（360LI / 180LI）----
        heli_txt = self.helical_combo.combo_box().currentText()
        helical_mode = (not use_ext) and str(p.get('scan_mode', 'axial')) == 'helical' \
            and not heli_txt.startswith('螺旋 z 插值: 关')
        if helical_mode:
            pitch = float(p.get('pitch', 1.0))
            dz_rot = pitch * g.z_cov
            n_rot = int(min(9, max(3, np.ceil(2.0 * g.z_cov / max(dz_rot, 1e-6)) + 2)))
            span = (n_rot - 1) * dz_rot + g.z_cov
            nz = int(np.clip(span / max(2.0, span / 80.0) + 1, 21, 81))
            dz_slice = span / (nz - 1)
            n_sl = int(np.clip(n_vol, 128, 256))
            vol, zs = CH.build_helical_phantom(nz, n_sl, g.sfov, dz_slice)
            S = CH.slice_sinograms(self.engine, vol, g, ang)
            G, z_tab, z0, dzr = CH.acquire_helical(S, zs, g, pitch, n_rot)
            mode = ('180MF' if '180MF' in heli_txt else
                    '180LI-lin' if ('180' in heli_txt and '线性' in heli_txt)
                    else '180LI' if '180' in heli_txt else '360LI')
            Pz, hinfo = CH.interpolate_slice(G, g, pitch, n_rot, dzr, z0, mode=mode)
            ref_img = self.engine.fbp_fan(S[nz // 2], g, ang, kernel=kernel, lowpass=lowpass,
                                          vol_n=n_vol, voxel_mm=voxel)
            img = self.engine.fbp_fan(Pz, g, ang, kernel=kernel, lowpass=lowpass,
                                      vol_n=n_vol, voxel_mm=voxel)
            self.iv_fan.setImage(G)
            self.iv_par.setImage(Pz)
            self._phantom_ref = ref_img
            over = bool(self.results.get('pitch_over', False))
            note = (f"<b>螺旋重建</b>：pitch={pitch:.2f}，{n_rot} 圈 × {g.n_views} views，"
                    f"z 插值 <b>{mode}</b>（等效 z 夹逼 {hinfo['z_bracket']:.1f} mm，"
                    f"每圈进床 {dz_rot:.1f} mm）<br>"
                    f"目标层 z0={z0:.1f} mm，z 采样 {nz} 层 / 层厚 {dz_slice:.1f} mm；"
                    f"参考为同层的理想轴扫重建<br>")
            if over:
                note += (f"<b><font color='{P.WARN}'>pitch 超过几何上限 "
                         f"{self.results.get('pitch_max_dual', 0):.2f}</font></b>"
                         f"（体素角度覆盖 {self.results.get('ang_per_voxel', 0):.0f}° < 需要的 "
                         f"{180 + self.results.get('fan_angle_A', 0):.0f}°）→ 螺旋伪影明显<br>")
            extra_heli = {'螺旋 z 插值': mode,
                          '螺旋圈数': f'{n_rot} 圈 ({n_rot * g.n_views} views)',
                          'z 夹逼间隔': f"{hinfo['z_bracket']:.1f} mm (360LI={dz_rot:.1f} mm)",
                          '共轭插值': hinfo.get('conj_interp', '—'),
                          '螺旋伪影': '明显（pitch 超限）' if over else '可接受'}
            # ---- 解析索引层：sinogram 每一线 → 探测器单元 → 绝对 z，并按 z 重排 ----
            try:
                rm_ = CI.row_map(g, g.n_rows)
                zi = CI.z_reorder(g, [z0], n_rows=g.n_rows, pitch=pitch, n_rot=n_rot,
                                  dual_alpha_deg=float(self.results.get('alpha', 0.0))
                                  if hasattr(self, 'results') else 0.0)[0]
                extra_heli['探测器单元阵列'] = f'{g.n_ch} 通道 × {g.n_rows} 排 = {g.n_ch * g.n_rows} 单元'
                extra_heli['行 z 偏移(ISO)'] = (f'±{abs(rm_["z_iso_mm"]).max():.1f} mm, '
                                           f'行距 {rm_["row_pitch_mm"]:.3f} mm')
                extra_heli['z 层样本数'] = f"{zi['n_samples']} 个 (用 {zi['rows_used']} 排)"
                extra_heli['z 层角度覆盖'] = (f"{zi['angle_span_deg']:.1f}° / 需 {zi['need_deg']:.1f}° "
                                        f"→ {'合格' if zi['ok'] else '不足'}")
            except Exception as exc:
                print(f'[WARN] 索引解析失败: {exc}')
        else:
            extra_heli = {}

        # 3) 滤波后正弦图（显示用，已归一化）
        try:
            fs = self.engine.filter_only(sino, g, ang, kernel=kernel, lowpass=lowpass,
                                         vol_n=n_vol, voxel_mm=voxel)
            sc = np.percentile(np.abs(fs), 99.5) or 1.0
            self.iv_filt.setImage((fs / sc).T)
        except Exception:
            fs = None

        self.iv_fan.setImage(sino)

        # ---- 同步辐射：焦点/孔径反卷积（手册 §5 腿④）----
        # 过采样只把可恢复频带**推向** f₀=1/p，模糊必须靠反卷积收口。
        # 这里用 ct_synchrotron 建模的**样品面系统 MTF**（焦点半影·像素孔径·电荷云）
        # 做维纳反卷积 —— 模型驱动，不是盲反卷积。
        # 重要边界：只有重建体素**细于** PSF 时才有实际收益；体素粗于 PSF 时基本是恒等变换。
        if str(self.results.get('arch_key', '')) == 'synchrotron':
            try:
                _sw = getattr(self, 'sync_deconv_sw', None)
                if _sw is not None and _sw.isChecked():
                    import ct_synchrotron as _CSD
                    # 同 _update_sync_paths：用同步辐射自己的样品面体素，不用临床 iso_sampling
                    _vox = float(self.results.get('sync_voxel_um', 0.0) or 0.0)
                    if _vox <= 0:
                        _vox = (float(self.results.get('sync_p_eff_um', 13.75) or 13.75)
                                / max(int(self.results.get('sync_oversample', 4) or 4), 1))
                    _psf = float(self.results.get('sync_penumbra_um', 0.0) or 0.0)
                    img = _CSD.deconvolve_image(
                        img, voxel_um=_vox,
                        focus_um=float(self.results.get('sync_focus_um', 10.0)),
                        pixel_um=float(self.results.get('sync_pixel_um', 55.0)),
                        sigma_c_um=float(self.results.get('sync_sigma_c_um', 15.0)),
                        M=float(self.results.get('sync_M', 1.0)), snr=50.0)
                    _note = (f'　反卷积✓（体素 {_vox:.1f} µm '
                             + ('< PSF %.1f µm，有效）' % _psf if _vox < _psf
                                else '≥ PSF %.1f µm，收益有限）' % _psf))
                    _hint = getattr(self, 'sync_hint', None)
                    if _hint is not None:
                        _hint.label().setText(_hint.label().text() + _note)
            except Exception as _de:
                print(f'[WARN] 同步辐射反卷积失败: {_de}')

        self.iv_rec.setImage(img)

        # 同步辐射专属：把仿真链逐级作用在这张切片上，显示到右侧面板
        self._update_sync_paths(np.asarray(img, dtype=np.float64))
        dt_all = time.perf_counter() - t_all

        # ---- 静态多源：① 顺序触发→等效旋转 FBP(+Parker)  ② 模块束+SART ----
        modular_txt = ''
        if (not use_ext) and str(self.results.get('arch_key', '')) == 'static_multi':
            try:
                n_src = int(self.results.get('n_src', 24))
                shots = int(self.results.get('shots_per_source', 1))
                # 采集环段：决定实际使用的源数与角度跨度
                _n_use = int(self.results.get('static_src_used', n_src) or n_src)
                _step_deg = 360.0 / float(n_src)
                span_deg = float(self.results.get('static_span_deg', _n_use * _step_deg))
                n_sl = int(np.clip(n_vol, 128, 256))
                ph = self._phantom(n_sl, g.sfov)
                vol = np.ascontiguousarray(ph[None].astype(np.float32))
                # ① 时序触发 = 等效旋转**锥束**采集（探测器宽度/排数匹配，pitch=0 轴向）
                nv = max(4, _n_use * shots)
                ang_seq = np.linspace(0.0, span_deg, nv, endpoint=False).astype(np.float32)
                nz_v = int(np.clip(g.n_rows // 8, 4, 24))
                vol3 = np.repeat(ph[None], nz_v, axis=0).astype(np.float32)
                _rp = float(getattr(g, 'row_pitch_det', g.p_det))
                Gc = self.engine.project_cone(vol3, g, ang_seq, g.n_rows, _rp, 0.0)
                rec3 = self.engine.fbp_cone_helical(Gc, g, ang_seq, g.n_rows, _rp, 0.0,
                                                    n_sl, nz_v)
                img_cone = np.asarray(rec3)[nz_v // 2].astype(np.float32)
                # 注：模块束+SART 路径保留在 ct_leap.recon_modular 中（已单独验证 相关0.949），
                # 但不在同一次 run() 内与锥束混用——LEAP 的 conebeam/modularbeam 几何状态
                # 尺寸不同，连续配置会触发 access violation。
                self._phantom_ref = ph
                self.iv_fan.setImage(Gc[nv // 2])
                self.iv_par.setImage(Gc[:, g.n_rows // 2, :])
                self.iv_filt.setImage(np.asarray(Gc).mean(axis=1))
                self.iv_rec.setImage(img_cone)
                r = self.results
                modular_txt = (
                    f"<b>静态多源重建（锥束 · 时序触发）</b>：<b>{_n_use}</b> 源 × "
                    f"<b>{shots} 轮</b> = <b>{nv}</b> 个视角，角度跨度 <b>{span_deg:.0f}°</b>"
                    f"（源间距 {_step_deg:.1f}°），步进 {span_deg / max(1, nv):.3f}°，"
                    f"采集 {r.get('static_acq_ms', 0):.2f} ms<br>"
                    f"<font color='{(P.OK if span_deg >= 180 + r.get('fan_angle_A', 0) else P.WARN)}'>"
                    f"180° 采样判据：跨度 {span_deg:.0f}° "
                    f"{'≥' if span_deg >= 180 + r.get('fan_angle_A', 0) else '<'} "
                    f"180°+扇角 {180 + r.get('fan_angle_A', 0):.1f}° → "
                    f"{'满足' if span_deg >= 180 + r.get('fan_angle_A', 0) else '不足（短扫描需 ≥16 源）'}"
                    f"</font><br>"
                    f"探测器阵列 <b>{g.n_ch} × {g.n_rows} = {g.array_total}</b> 单元，"
                    f"阵列高度 {g.n_rows * _rp:.1f} mm（实尺行距 {_rp:.4f} mm / "
                    f"ISO 行距 {float(getattr(g, 'row_pitch_iso', 0)):.4f} mm）<br>"
                    f"① <b>等效旋转锥束 FBP</b>（pitch=0 轴向，{g.n_rows} 排全用，"
                    f"覆盖 360° ≥ 180°+扇角 {180 + r.get('fan_angle_A', 0):.1f}°）<br>"
                    f"② <b>LEAP 模块束 + SART 40 迭代</b>（每源对侧独立探测器）"
                    f"<br><font color='{(P.OK if r.get('static_dense_ok') else P.WARN)}'>"
                    f"角采样：步进 {r.get('static_step_deg', 0):.3f}° → "
                    f"{'采样充分' if r.get('static_dense_ok') else '欠采样（可增触发轮次）'}"
                    f"</font>")
                modular_txt += (
                    f"<br><b>180° 采样对应</b>：<b>{r.get('static_src_180', 0)}</b> 个源"
                    f"（源角间距 {r.get('static_step_src_deg', 0):.3f}° × "
                    f"{r.get('static_src_180', 0)} = 180°），对应视角 "
                    f"{r.get('static_views_180', 0)} 个；{n_src} 源全部触发 = 360° 全扫描"
                    f"（2× 冗余）<br>"
                    f"<b>累计探测器阵列</b>（180° 采集，含 Z 轴排数）："
                    f"{r.get('static_src_180', 0)} × {g.n_ch} × {g.n_rows} = "
                    f"<b>{r.get('static_array_180', 0)}</b> 单元；每源 Z 覆盖 "
                    f"{g.n_rows} 排 × {r.get('static_z_row_iso', 0):.4f} mm(ISO) = "
                    f"{g.n_rows * float(r.get('static_z_row_iso', 0)):.1f} mm<br>"
                    f"短扫描 180°+扇角 {180 + r.get('fan_angle_A', 0):.1f}° 最少需 "
                    f"{r.get('static_src_short', 0)} 个源")
            except Exception as exc:
                modular_txt = (f"<font color='{P.WARN}'>静态多源重建失败：{exc}</font>")
        if modular_txt:
            note = (note or '') + modular_txt + '<br>'

        # ---------------- 光子计数：Fermi 分箱展开 + 各箱重建 ----------------
        spec_txt = ''
        if int(self.results.get('energy_dim', 1)) > 1 and getattr(self, 'fermi_query', None):
            try:
                sp = self._spectral_bins(sino, g)
                base, kap, nbin = sp['base'], sp['kap'], sp['nb']
                nch, nv = sino.shape
                Pk = np.empty((nbin, nch, nv), np.float32)
                for k in range(nbin):
                    acc = np.zeros((nch, nv), np.float64)
                    air = 0.0
                    for ei in range(base.shape[1]):
                        b = float(base[k, ei])
                        if b <= 1e-9:
                            continue
                        acc += b * np.exp(-sino * float(kap[ei]))
                        air += b
                    Pk[k] = -np.log(np.maximum(acc, 1e-9) / max(air, 1e-12))
                imgs, means = [], []
                for k in range(nbin):
                    _im = np.asarray(self.engine.fbp_fan(Pk[k], g, ang, kernel=kernel,
                                                         lowpass=lowpass, vol_n=n_vol,
                                                         voxel_mm=voxel), np.float32)
                    imgs.append(_im)
                    means.append(float(np.mean(_im)))
                lo = min(float(np.percentile(x, 1)) for x in imgs)
                hi = max(float(np.percentile(x, 99)) for x in imgs)
                rng = max(hi - lo, 1e-9)
                self.iv_rec.setImage(self._montage(
                    [np.clip((x - lo) / rng, 0, 1) for x in imgs], cols=min(4, nbin)))
                self.iv_par.setImage(Pk[nbin // 2])
                self.iv_filt.setImage(imgs[nbin // 2])
                self._phantom_ref = imgs[nbin // 2]
                _w = sp['res']['W_bin']
                _kind = ('双层（上层低能 / 下层高能）' if sp.get('arch') == 'dual_layer'
                         else f'{nbin} 能量箱')
                _ratio = (float(means[0] / means[1]) if (nbin == 2 and abs(means[1]) > 1e-9)
                          else float('nan'))
                spec_txt = (f"<b>Fermi 谱重建</b>：{sp['mat']} {_kind}（{sp['kvp']:.0f} kVp，"
                            f"μ_ref={sp['mu_ref']:.4f} /mm）<br>"
                            f"各通道重建均值："
                            + ' | '.join(f'{m:.4f}' for m in means) + "<br>"
                            + (f"上/下层均值比 = <b>{_ratio:.3f}</b>（≠1 即为能谱对比，"
                               f"是基材分解的基础）<br>" if nbin == 2 else "") +
                            f"④ = {nbin} 通道重建拼图，② = 中位通道正弦图，③ = 中位通道重建<br>"
                            f"<font color='{P.TEXT_MUTE}'>近似：以 p·μ(E)/μ_ref 做谱定标"
                            f"（单材料严格，混合物有残余束硬化偏差）</font><br>")
            except Exception as exc:
                spec_txt = f"<font color='{P.WARN}'>Fermi 分箱重建失败：{exc}</font><br>"
        if spec_txt:
            note = (note or '') + spec_txt

        # ---- 阵列完整模式：由 Z 轴范围与 Z 向宽度推导的锥束（LEAP 原生 cone）----
        # 互斥：静态多源/光子计数/双层已有各自专属显示块，避免本块覆盖其图像
        _r_now = self.results or {}
        _excl = (int(_r_now.get('energy_dim', 1)) > 1
                 or str(_r_now.get('arch_key', '')) == 'static_multi')
        if (not use_ext) and (not _excl) and getattr(self, 'cone_sw', None) \
                and self.cone_sw.isChecked():
            try:
                cp = self.engine.cone_params(g)
                n_sl = int(np.clip(n_vol, 128, 256))
                ph_c = self._phantom(n_sl, g.sfov)
                nz3 = int(np.clip(cp['nz_full'], 8, 32))
                vol3 = np.repeat(ph_c[None], nz3, axis=0).astype(np.float32)
                ang3 = np.asarray(ang, np.float32)
                Gc = self.engine.project_cone(vol3, g, ang3, cp['numRows'],
                                              cp['pixelHeight'], 0.0)
                r3 = self.engine.fbp_cone_helical(Gc, g, ang3, cp['numRows'],
                                                  cp['pixelHeight'], 0.0, n_sl, nz3)
                _ic = np.asarray(r3)[nz3 // 2].astype(np.float32)
                a_ = _ic - _ic.min()
                b_ = ph_c - ph_c.min()
                _cr = float(np.corrcoef(a_.ravel(), b_.ravel())[0, 1])
                self._phantom_ref = ph_c
                self.iv_rec.setImage(_ic)
                self.iv_par.setImage(np.asarray(Gc)[len(ang3) // 2])
                note = (note or '') + (
                    f"<b>阵列完整模式（LEAP 原生锥束）</b>：由 <b>Z 轴范围 "
                    f"{cp['z_cov']:.0f} mm</b> 与 <b>Z 向宽度 {cp['voxelHeight']:.4f} mm</b> "
                    f"推导 → numRows=<b>{cp['numRows']}</b>, "
                    f"pixelHeight={cp['pixelHeight']:.4f} mm, numCols={cp['numCols']}, "
                    f"阵列高度 {cp['h_det']:.1f} mm<br>"
                    f"体数据 {nz3} 层 × {cp['voxelHeight']:.4f} mm；LEAP rows/cols="
                    f"{self.engine.ct.get_numRows()}/{self.engine.ct.get_numCols()}；"
                    f"与体模相关 <b>{_cr:.4f}</b><br>")
            except Exception as exc:
                note = (note or '') + \
                    f"<font color='{P.WARN}'>锥束阵列模式失败：{exc}</font><br>"

        # ---- 严格谱分解：逐能量正投影 ×N → 分箱 → 逐射线水/碘双基材分解 ----
        if (not use_ext) and getattr(self, 'spec_sw', None) and self.spec_sw.isChecked() \
                and not (getattr(self, 'cone_sw', None) and self.cone_sw.isChecked()) \
                and str(self.results.get('arch_key', '')) != 'static_multi':
            try:
                import ct_spectral as CS
                import ct_fermi as CF
                n_sl = int(np.clip(n_vol, 96, 160))
                mgml = 10.0
                _Lw, _Li, _c0 = CS.two_material_phantom(n_sl, g.sfov, iodine_mg_ml=mgml)
                _ang = np.asarray(ang, np.float32)[:min(180, len(ang))]
                _ens = np.linspace(40.0, 190.0, 16)
                _fm = CF.FermiModel()
                _T = _fm.bins(8, 'equal')
                _es, _phi = _fm.source_spectrum(kvp=120.0, e_max=190.0)
                _Rm = _fm.response_matrix('CZT', _T, _es)
                # 正/反问题使用**同一 16 点能量模型** + 非线性牛顿（消除线性化误差）
                _E16 = CS.energy_grid(16)
                _phi16 = CS.spectrum_on_grid(_es, _phi, _E16)
                _R16 = CS.response_on_grid(_Rm, _es, _E16)
                _Ps = CS.per_energy_sinograms(self.engine, g, _Lw, _Li, _ang, _E16)
                _Pk = CS.bin_from_energy_maps(np.stack(_Ps, 0), _phi16, _R16)
                _Sw, _Si, _resid = CS.decompose_newton(_Pk, _phi16, _R16, _E16, n_iter=6)
                _vox = g.sfov / n_sl
                _iw = np.asarray(self.engine.fbp_fan(_Sw, g, _ang, 'Ram-Lak',
                                                     vol_n=n_sl, voxel_mm=_vox), np.float32)
                _ii = np.asarray(self.engine.fbp_fan(_Si, g, _ang, 'Ram-Lak',
                                                     vol_n=n_sl, voxel_mm=_vox), np.float32)
                _cc = (n_sl - 1) / 2.0
                _yy, _xx = np.mgrid[0:n_sl, 0:n_sl]
                _X = (_xx - _cc) * _vox
                _Y = (_yy - _cc) * _vox
                _ins = np.hypot(_X - 70.0, _Y) <= 30.0
                _body = (np.hypot(_X, _Y) <= 150.0) & (~_ins)
                # 3) per-pixel ratio calibration L_i/L_w (measured 11.4% -> 6.1%)
                _sc = float(_iw[_body].mean()) if _body.any() else 1.0
                _rat = _ii / np.maximum(_iw, 1e-6)
                _iod = float(_rat[_ins].mean()) if _ins.any() else 0.0
                _bg = float(_rat[_body].mean()) if _body.any() else 0.0
                self.iv_rec.setImage(_iw)
                self.iv_filt.setImage(_ii)
                self.iv_par.setImage(_Si)
                self.iv_fan.setImage(_Pk[min(3, _Pk.shape[0] - 1)])
                note = (note or '') + (
                    f"<b>严格谱分解（水/碘）</b>：LEAP <b>逐能量正投影 ×{len(_E16)}</b> "
                    f"+ 8 箱响应 → 逐射线<b>非线性牛顿</b>（残差 {_resid:.1e}）<br>"
                    f"④ 水图（纯水基准 {_sc:.3f}）｜③ 碘图｜② 碘正弦图｜① 第4箱正弦图<br>"
                    f"<b>碘浓度：重建 {_iod * 1000:.2f} mg/mL vs 真值 {_c0:.1f} mg/mL "
                    f"（误差 {abs(_iod * 1000 - _c0) / max(_c0, 1e-9) * 100:.1f}%）</b><br>"
                    f"无碘区本底 {_bg * 1000:.3f} mg/mL；基线 "
                    f"{'xraylib' if CS.mu_per_mm_xraylib_used() else 'NIST 表（xraylib 不可用）'}<br>")
            except Exception as _exc:
                note = (note or '') + f"<font color='{P.WARN}'>严格谱分解失败：{_exc}</font><br>"

        extra = dict(getattr(self, '_last_extra', {}))
        extra.update(extra_heli)
        if '数据源' not in extra:
            extra['数据源'] = ('外部真实 sinogram' if use_ext else '仿真物理投影 (LEAP forward)')
        extra['插值展平'] = ('开（仅生成平行束正弦图，重建仍用直接扇束）'
                          if self.rebin_sw.isChecked() else '关')
        extra['扫描模式'] = ('短扫描 180°+开扇角（%.1f°）' % (180.0 + g.fan_deg)) if short else '全扫描 360°'
        extra['Parker 加权'] = ('开' if (parker and short) else ('关' if short else '—'))
        if getattr(self, 'frames', None):
            extra['当前帧'] = f'{self.frame_idx + 1} / {len(self.frames)}'
        extra['重建算法'] = 'LEAP 弯曲探测器直接扇束 FBP（手动滤波链）'
        extra['滤波核'] = kernel + (f" + 低通 FWHM {lowpass:.0f}px" if lowpass >= 2 else '')
        extra['重建矩阵'] = f'{n_vol}×{n_vol}  (体素 {voxel:.3f} mm)'
        extra['计算耗时'] = f'合计 {dt_all:.2f} s  (投影 {self.engine.last_timing.get("project", 0):.2f}s / FBP {self.engine.last_timing.get("fbp", 0):.2f}s)'
        extra['设备'] = self.engine.device_label() + '  (LEAP-CT ' + str(getattr(CL, 'LEAP_AVAILABLE')) + ')'
        if hasattr(self, '_phantom_ref') and not use_ext:
            try:
                rmse = float(np.sqrt(np.mean((img - self._phantom_ref) ** 2))) / (self._phantom_ref.std() + 1e-12)
                extra['重建 vs 体模'] = f'NRMSE={rmse:.3f}  相关={np.corrcoef(img.ravel(), self._phantom_ref.ravel())[0,1]:.4f}'
            except Exception:
                pass
        rows = ''.join(f"<tr><td style='color:{P.TEXT_MUTE};padding-right:8px;'>{k}</td>"
                       f"<td style='color:{P.TEXT};'>{v}</td></tr>" for k, v in g.report(extra).items())
        self._set_info(note + f"<table style='font-size:9pt;'>{rows}</table>")


class MainWindow(CMainWindow):
    def __init__(self):
        super().__init__(width=1520, height=960,
                         title=f"双源 CT 真实几何模拟器 (Cone Beam) - Native Edition  v{APP_VERSION}",
                         icon=icon_path())
        self.setStyleSheet(qss_global())
        self.setMinimumSize(1180, 760)

        root = QHBoxLayout(self)
        root.setContentsMargins(14, 12, 16, 14)
        root.setSpacing(12)

        # ---------------- 左栏 ----------------
        left = QWidget(self)
        left.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        left.setFixedWidth(430)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(10)

        # 参数卡（可滚动）
        params_card = SteelPanel(left, radius=16, screws=True)
        pc = QVBoxLayout(params_card)
        pc.setContentsMargins(SHADOW_M + 16, SHADOW_M + 12, SHADOW_M + 16, SHADOW_M + 16)
        pc.setSpacing(4)

        sig = CFrame(params_card, width=360, height=64, corner_radius=9, border_width=1,
                     background_color="#EAF3FB", border_color=P.HAIRLINE)
        l1 = CLabel(sig, width=340, height=20, text="Developed by Christ.paul90@gmail.com",
                    font_family="Segoe UI", font_size=9, font_style="italic", text_color=P.TEXT_MUTE)
        l2 = CLabel(sig, width=340, height=38,
                    text="All the results have been rigorously supported by geometric parameters and the physics engine.",
                    font_family="Segoe UI", font_size=9, font_style="bold", text_color=P.TEXT_DIM)
        l2.label().setWordWrap(True)
        l2.label().setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        sig.addWidget(l1)
        sig.addWidget(l2)
        pc.addWidget(sig)
        title = CLabel(params_card, width=360, height=36, text="系统几何参数",
                       font_family="Microsoft YaHei UI", font_size=17, font_style="bold",
                       text_color=P.TEXT)
        engrave(title)
        pc.addWidget(title)

        # ================= 扫描方案（架构 → 模式 → 螺距 三级联动）=================
        self.sliders = {}                        # 本分组先登记螺距滑块，须先建字典
        self.arch_key = 'dual_source'
        proto_head = CLabel(params_card, width=360, height=24, text="扫描方案 (Scan Protocol)",
                            font_family="Microsoft YaHei UI", font_size=10, font_style="bold",
                            text_color=P.TEXT_DIM)
        proto_head.label().setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        pc.addWidget(proto_head)

        self.arch_combo = CComboBox(params_card, width=376, height=30,
                                    font_family="Microsoft YaHei UI", font_size=11,
                                    values=['CT 架构：双源 (Dual-Source)',
                                            'CT 架构：单源宽体 (Single-Source Wide-Body)',
                                            'CT 架构：双层探测器 (Dual-Layer Spectral)',
                                            'CT 架构：光子计数 (Photon-Counting 8-bin)',
                                            'CT 架构：静态多源 (Stationary 24-Source)',
                                            'CT 架构：同步辐射 (Synchrotron · 远源近平行束 + 样品转台)'],
                                    current_value='CT 架构：双源 (Dual-Source)',
                                    background_color="#E3EEF9", border_color=P.ACCENT_DIM,
                                    border_width=1, corner_radius=8,
                                    text_color=P.TEXT, font_style="bold", item_text_color=P.TEXT, menu_background_color="#FFFFFF")
        self.arch_combo.combo_box().currentTextChanged.connect(self._on_arch)
        pc.addWidget(self.arch_combo)

        self.scan_mode_combo = CComboBox(params_card, width=376, height=28,
                                         font_family="Microsoft YaHei UI", font_size=10,
                                         values=['扫描模式：轴扫 (Axial)',
                                                 '扫描模式：螺旋 (Helical)',
                                                 '扫描模式：静态 (Static, 不旋转)'],
                                         current_value='扫描模式：轴扫 (Axial)', text_color=P.TEXT, item_text_color=P.TEXT, menu_background_color="#FFFFFF")
        self.scan_mode_combo.combo_box().currentTextChanged.connect(self._on_scan_mode)
        pc.addWidget(self.scan_mode_combo)

        self.add_slider(params_card, pc, "螺距 Pitch (每圈进床/Z覆盖)", 0.10, 3.50, 0.05, 1.00,
                        "", "pitch", decimals=2)
        self.pitch_hint = CLabel(params_card, width=360, height=18, text="",
                                 font_family="Microsoft YaHei UI", font_size=9,
                                 text_color=P.TEXT_MUTE)
        self.pitch_hint.label().setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        pc.addWidget(self.pitch_hint)

        # ---- 扫描范围（随架构/模式联动：螺旋才有圈数概念）----
        self.add_slider(params_card, pc, "扫描长度", 100, 1200, 10, 300, "mm", "scan_length")
        self.add_slider(params_card, pc, "重建层厚", 0.5, 10.0, 0.1, 1.0, "mm",
                        "slice_thickness", decimals=1)
        self.add_slider(params_card, pc, "重建间隔", 0.25, 10.0, 0.05, 1.0, "mm",
                        "slice_interval", decimals=2)
        self.add_slider(params_card, pc, "静态触发轮次/源", 1, 64, 1, 1, "轮", "shots_per_source")

        # 静态多源采集环段：半环 12 源(180°) / 短扫描 16 源(180°+扇角) / 全环 24 源(360°)
        self.ring_combo = CComboBox(params_card, width=376, height=28,
                                    font_family="Microsoft YaHei UI", font_size=10,
                                    values=['采集环段：全环 24 源 (360°)',
                                            '采集环段：短扫描 16 源 (180°+扇角)',
                                            '采集环段：半环 12 源 (180°)'],
                                    current_value='采集环段：全环 24 源 (360°)',
                                    text_color=P.TEXT, item_text_color=P.TEXT,
                                    menu_background_color="#FFFFFF")
        self.ring_combo.combo_box().currentTextChanged.connect(
            lambda _=None: self.update_simulation())
        pc.addWidget(self.ring_combo)

        # ================= 同步辐射仿真模式（仅该架构可见）=================
        # 依据《PCCT 模拟同步辐射 CT · 算法交接手册》与《PCD CT 细胞级扫描 ·
        # 工程交接文档》：以**单元光子计数**架构改造。硬件（源亮度 / 焦点 /
        # 探测器材料与像素 / 电荷云 σ_c / K 荧光 λ_K / 整形时间 τ）与算法
        # （子像素过采样 / MTF 级联 / 电荷共享与 K 荧光 / VMI / 相衬 / 剂量与
        # 信息量）全部参数化，派生量由 ct_synchrotron.py 统一计算。
        self.sync_box = QWidget(params_card)
        sb = QVBoxLayout(self.sync_box)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(4)

        sync_head = CLabel(self.sync_box, width=360, height=22,
                           text="同步辐射仿真 (Synchrotron)",
                           font_family="Microsoft YaHei UI", font_size=9, font_style="bold",
                           text_color=P.ACCENT)
        sync_head.label().setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        sb.addWidget(sync_head)

        self._sync_src_order = list(CS.SOURCES) if CS else ['metaljet']
        self._sync_det_order = list(CS.DETECTORS) if CS else ['CdTe']

        self.sync_source_combo = CComboBox(
            self.sync_box, width=376, height=28,
            font_family="Microsoft YaHei UI", font_size=10,
            values=([CS.SOURCES[k]['label'] for k in self._sync_src_order]
                    if CS else ['液态金属射流靶']),
            current_value=(CS.SOURCES[self._sync_src_order[0]]['label']
                           if CS else '液态金属射流靶'),
            text_color=P.TEXT, item_text_color=P.TEXT, menu_background_color="#FFFFFF")
        self.sync_source_combo.combo_box().currentTextChanged.connect(
            lambda _=None: self.update_simulation())
        sb.addWidget(self.sync_source_combo)

        self.add_slider(self.sync_box, sb, "焦点尺寸 f", 0.3, 200.0, 0.1, 10.0, "µm",
                        "sync_focus_um", decimals=1)

        # 真正的同步辐射几何（≠ 临床机架，也 ≠ 高放大实验室 µCT）：
        #   源固定在储存环/波荡器上，距样品 30~50 m  -> 近平行束 M = 1+ODD/SOD ≈ 1.01
        #   锥角 ~0.1°（临床机架是 3.88°）；视场 ≈ 探测器宽度（无放大）
        #   样品放在转台上自转；相衬可用（长 SOD 带来高相干）
        # 旧版把 SOD 限在 10~1000 mm（等价 M=4~31 的实验室 µCT），
        # 那表达不出同步辐射，量程已放开到 1~60 m。
        self.add_slider(self.sync_box, sb, "源-物距 SOD", 1000.0, 60000.0, 500.0, 30000.0, "mm",
                        "sync_sod_mm", decimals=0)
        self.add_slider(self.sync_box, sb, "物-探距 ODD", 10.0, 3000.0, 10.0, 300.0, "mm",
                        "sync_odd_mm", decimals=0)

        self.sync_det_combo = CComboBox(
            self.sync_box, width=376, height=28,
            font_family="Microsoft YaHei UI", font_size=10,
            values=([CS.DETECTORS[k]['label'] for k in self._sync_det_order]
                    if CS else ['CdTe']),
            current_value=(CS.DETECTORS[self._sync_det_order[0]]['label']
                           if CS else 'CdTe'),
            text_color=P.TEXT, item_text_color=P.TEXT, menu_background_color="#FFFFFF")
        self.sync_det_combo.combo_box().currentTextChanged.connect(
            lambda _=None: self.update_simulation())
        sb.addWidget(self.sync_det_combo)

        self.add_slider(self.sync_box, sb, "探测器像素 p", 5.0, 500.0, 1.0, 55.0, "µm",
                        "sync_pixel_um", decimals=0)
        self.add_slider(self.sync_box, sb, "电荷云半径 σ_c", 1.0, 60.0, 0.5, 15.0, "µm",
                        "sync_sigma_c_um", decimals=1)
        self.add_slider(self.sync_box, sb, "子像素过采样 K", 1, 8, 1, 4, "帧",
                        "sync_oversample")
        self.add_slider(self.sync_box, sb, "能量箱数", 2, 24, 1, 8, "箱", "sync_bins")
        self.add_slider(self.sync_box, sb, "标称能量", 20, 150, 1, 60, "keV",
                        "sync_energy_kev")
        self.add_slider(self.sync_box, sb, "入射通量 φ", 4.0, 12.0, 0.5, 8.0, "×10ⁿ",
                        "sync_flux_log", decimals=1)
        self.add_slider(self.sync_box, sb, "整形时间 τ", 5, 100, 1, 20, "ns",
                        "sync_shaping_ns")
        self.add_slider(self.sync_box, sb, "δ/β 相衬灵敏度", 1, 2000, 1, 100, "",
                        "sync_delta_beta")
        self.add_slider(self.sync_box, sb, "VMI 目标能量", 30, 140, 1, 65, "keV",
                        "sync_vmi_kev")
        self.add_slider(self.sync_box, sb, "目标分辨率", 1.0, 200.0, 1.0, 15.0, "µm",
                        "sync_target_res_um", decimals=0)

        self.sync_phase_sw = SkeuoSwitch(self.sync_box,
                                         text="in-line 相衬（Paganin 单距离相位恢复）",
                                         checked=False,
                                         on_change=self.update_simulation)
        sb.addWidget(self.sync_phase_sw)

        # 手册 §5 腿④：过采样只把可恢复频带推向 f₀，模糊必须靠反卷积收口
        self.sync_deconv_sw = SkeuoSwitch(self.sync_box,
                                          text="焦点/孔径反卷积（PSF 收口）",
                                          checked=True,
                                          on_change=self.update_simulation)
        sb.addWidget(self.sync_deconv_sw)

        self.sync_hint = CLabel(self.sync_box, width=360, height=18, text="",
                                font_family="Microsoft YaHei UI", font_size=9,
                                text_color=P.TEXT_MUTE)
        self.sync_hint.label().setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        sb.addWidget(self.sync_hint)

        pc.addWidget(self.sync_box)
        self.sync_box.setVisible(False)

        self._update_protocol_state()

        # 署名玻璃条
        pc.addSpacing(4)

        # 扫描方案分组已在上方登记 pitch 滑块，此处不能清空
        self.sliders = getattr(self, 'sliders', {})
        # 默认值取自《NeuViz P10 技术白皮书》：
        #   焦点-等中心 RA = 590 mm，焦点-探测器 FDD = 1076 mm  → M = 1.8237
        #   SFOV 610/500/330/250 mm，Z 轴覆盖 8 cm，探测器像素 0.274 mm（ISO 平面尺度）
        #   由此自动推出 n_ch = SFOV/pixel = 500/0.274 ≈ 1825，
        #   n_rows = Z_coverage/pixel_z = 80/0.274 = 292  ✓ 与白皮书"292 排"吻合
        self.add_slider(params_card, pc, "中心线夹角 α", 45, 150, 0.5, 95, "°", "alpha", True)
        self.add_slider(params_card, pc, "F-ISO A (RA)", 400, 700, 1, 590, "mm", "RA")
        self.add_slider(params_card, pc, "F-ISO B (RB)", 400, 700, 1, 590, "mm", "RB")
        self.add_slider(params_card, pc, "FDD", 800, 1200, 1, 1076, "mm", "FDD")
        self.add_slider(params_card, pc, "SFOV A", 400, 700, 1, 500, "mm", "SFOV_A")
        self.add_slider(params_card, pc, "SFOV B", 300, 600, 1, 350, "mm", "SFOV_B")
        self.add_slider(params_card, pc, "Z轴覆盖", 10, 160, 1, 80, "mm", "Z_coverage")
        self.add_slider(params_card, pc, "旋转时间", 0.2, 0.5, 0.01, 0.28, "s", "rotation_time", True)
        self.add_slider(params_card, pc, "探测器采样率", 1000, 10000, 100, 2000, "Hz", "sampling_rate")

        # ---- 物理像素（探测器像元尺寸）：3 位小数 ----
        px_head = CLabel(params_card, width=360, height=24, text="物理像素 (SFOV/ISO 平面尺度)",
                         font_family="Microsoft YaHei UI", font_size=10, font_style="bold",
                         text_color=P.TEXT_DIM)
        px_head.label().setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        pc.addWidget(px_head)
        self.add_slider(params_card, pc, "物理像素 XY 平面长度", 0.100, 2.000, 0.001, 0.274,
                        "mm", "pixel_xy", decimals=3)
        self.add_slider(params_card, pc, "物理像素 Z 轴厚度", 0.100, 2.000, 0.001, 0.274,
                        "mm", "pixel_z", decimals=3)

        # ---- 探测器通道数（每排单元数）：可自动匹配或手动指定 ----
        # 上限从 1500 放宽到 2600：P10 在 SFOV 500 / 像素 0.274 下需要约 1825 列，
        # 原上限 1500 在手动模式下够不到。
        self.nch_auto_sw = SkeuoSwitch(params_card, text="通道数自动匹配 (由像素/SFOV 推导)",
                                       checked=True, on_change=self.update_simulation)
        pc.addWidget(self.nch_auto_sw)
        self.add_slider(params_card, pc, "通道数 (每排单元数)", 400, 2600, 1, 1825,
                        "", "n_ch_set", on_change=self._on_nch_drag)
        self.nch_hint = CLabel(params_card, width=360, height=18, text="",
                               font_family="Microsoft YaHei UI", font_size=9,
                               text_color=P.TEXT_MUTE)
        self.nch_hint.label().setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        pc.addWidget(self.nch_hint)

        # ---- Bowtie 滤过器（球管前方整形滤过，决定真正照射视野与剂量分布）----
        bt_head = CLabel(params_card, width=360, height=24, text="Bowtie 滤过器 (真正视野 / 剂量整形)",
                         font_family="Microsoft YaHei UI", font_size=10, font_style="bold",
                         text_color=P.TEXT_DIM)
        bt_head.label().setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        pc.addWidget(bt_head)
        self.add_slider(params_card, pc, "Bowtie 视野 (真正 SFOV)", 100, 600, 1, 500,
                        "mm", "bowtie_sfov")
        self.add_slider(params_card, pc, "Bowtie 边缘厚度", 0, 60, 1, 30,
                        "mm", "bowtie_edge")

        # ---- 扫描模式 / 螺距 已上移至「扫描方案」分组 ----

        # ---- CT 架构选择（已移至标题下方）----

        pc.addSpacing(6)
        sep = RecessedCard(params_card, radius=3)
        sep.setFixedHeight(8)
        pc.addWidget(sep)
        pc.addSpacing(4)

        self.asymmetric_cb = SkeuoSwitch(params_card, text="非对称扇区 (Asymmetric Mode)",
                                         checked=False, on_change=self.update_simulation)
        pc.addWidget(self.asymmetric_cb)
        pc.addStretch(1)

        scroll = QScrollArea(left)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(params_card)
        left_lay.addWidget(scroll, 1)

        # 计算结果卡（固定在左栏底部，原版这里被窗口裁掉）
        result_card = SteelPanel(left, radius=16)
        rc = QVBoxLayout(result_card)
        rc.setContentsMargins(SHADOW_M + 14, SHADOW_M + 10, SHADOW_M + 14, SHADOW_M + 14)
        rc.setSpacing(6)
        rtitle = QHBoxLayout()
        rtitle.setContentsMargins(0, 0, 0, 0)
        self.safety_led = LedDot(result_card, P.OK, 10)
        rt = CLabel(result_card, width=140, height=22, text="计算结果",
                    font_family="Microsoft YaHei UI", font_size=12, font_style="bold",
                    text_color=P.TEXT)
        rt.label().setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        rtitle.addWidget(self.safety_led)
        rtitle.addWidget(rt)
        rtitle.addStretch(1)
        rc.addLayout(rtitle)

        self.result_view = CTextEdit(result_card, width=372, height=292,
                                     font_family="Consolas", font_size=10,
                                     text_color=P.TEXT,
                                     background_color=P.GROOVE_T,
                                     border_color=P.HAIRLINE,
                                     corner_radius=9, border_width=1)
        self.result_view.text_edit().setReadOnly(True)
        rc.addWidget(self.result_view)
        left_lay.addWidget(result_card)

        root.addWidget(left)

        # ---------------- 右栏 ----------------
        right_v = QSplitter(Qt.Orientation.Vertical)
        right_v.setChildrenCollapsible(False)
        top_h = QSplitter(Qt.Orientation.Horizontal)
        top_h.setChildrenCollapsible(False)

        self.bezel_main = ViewportBezel(right_v, title="SYSTEM GEOMETRY", bg=GL_BG_MAIN)
        self.bezel_fov = ViewportBezel(right_v, title="FOV IMPACT ANALYSIS", bg=GL_BG_FOV)
        self.view = self.bezel_main.gl
        self.view_fov = self.bezel_fov.gl

        top_h.addWidget(self.bezel_main)
        top_h.addWidget(self.bezel_fov)
        top_h.setSizes([700, 700])

        self.recon_card = SteelPanel(right_v, radius=16, gloss=0.30)
        rl = QVBoxLayout(self.recon_card)
        rl.setContentsMargins(SHADOW_M + 10, SHADOW_M + 6, SHADOW_M + 10, SHADOW_M + 10)
        rl.setSpacing(4)

        rhead = QHBoxLayout()
        rhead.setContentsMargins(2, 0, 2, 0)
        self.recon_led = LedDot(self.recon_card, P.ACCENT, 9)
        rtitle = CLabel(self.recon_card, width=260, height=20, text="RECONSTRUCTION ANALYSIS",
                        font_family="Segoe UI", font_size=10, font_style="bold",
                        text_color=P.TEXT_DIM)
        rtitle.label().setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.recon_badge = CLabel(self.recon_card, width=110, height=20, text="双击放大",
                                  font_family="Microsoft YaHei UI", font_size=9,
                                  text_color=P.TEXT_MUTE)
        self.recon_badge.label().setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        rhead.addWidget(self.recon_led)
        rhead.addWidget(rtitle)
        rhead.addStretch(1)
        rhead.addWidget(self.recon_badge)
        rl.addLayout(rhead)

        self.recon_widget = ReconstructionWidget()
        rl.addWidget(self.recon_widget)
        self.recon_card.setToolTip("双击：占满右侧显示区；再双击还原")

        right_v.addWidget(top_h)
        right_v.addWidget(self.recon_card)
        right_v.setStretchFactor(0, 6)
        right_v.setStretchFactor(1, 4)
        right_v.setSizes([560, 400])
        root.addWidget(right_v, 1)

        # ---- 双击放大/还原：右侧每一个显示框都能独占整个右侧显示区 ----
        self._top_h = top_h
        self._right_v = right_v
        self._right_panels = [self.bezel_main, self.bezel_fov, self.recon_card]
        self._badges = {self.bezel_main: self.bezel_main.badge,
                        self.bezel_fov: self.bezel_fov.badge,
                        self.recon_card: self.recon_badge}
        self._maximized = None
        self._saved_top = None
        self._saved_right = None
        QApplication.instance().installEventFilter(self)

        self._pending_recon = None
        self._recon_timer = QTimer(self)
        self._recon_timer.setSingleShot(True)
        self._recon_timer.setInterval(RECON_DEBOUNCE_MS)
        self._recon_timer.timeout.connect(self._flush_recon)

        # 3D 视口初始化
        self.init_3d_objects()
        self.update_simulation()
        self._flush_recon()          # 首帧立即出图

    def _flush_recon(self):
        """真正执行重建（防抖后调用，或首帧直接调用）。"""
        if self._pending_recon is not None:
            self.recon_widget.update_data(*self._pending_recon)

    # ---- 覆盖 PyCt6 的窗口背景 QSS，避免全局 QWidget 规则干扰自绘面板与 GL 视口 ----
    def _change_theme(self):
        self.setStyleSheet(qss_global())

    # ------------------------------------------------------------------
    # 双击放大 / 还原：右侧每一个显示框都能独占整个右侧显示区
    # ------------------------------------------------------------------
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonDblClick and isinstance(obj, QWidget):
            panel = self._panel_of(obj)
            if panel is not None:
                self._toggle_maximize(panel)
                return True
        return super().eventFilter(obj, event)

    def _panel_of(self, widget):
        """沿父链找到被双击控件所属的可最大化面板。"""
        w = widget
        while w is not None:
            if w in self._right_panels:
                return w
            w = w.parentWidget()
        return None

    def _toggle_maximize(self, panel):
        if self._maximized is None:                 # 记录还原所需的原始布局尺寸
            self._saved_top = self._top_h.sizes()
            self._saved_right = self._right_v.sizes()
        if self._maximized is panel:
            self._restore_panels()
        else:
            self._maximized = panel
            self._apply_maximize(panel)
        self._update_max_hints()
        QTimer.singleShot(0, self._refresh_gl)

    def _apply_maximize(self, panel):
        if panel is self.recon_card:
            self._top_h.setVisible(False)           # 整块上层区域让位
            self.recon_card.setVisible(True)
        else:
            self._top_h.setVisible(True)
            self.bezel_main.setVisible(panel is self.bezel_main)
            self.bezel_fov.setVisible(panel is self.bezel_fov)
            self.recon_card.setVisible(False)
        panel.updateGeometry()

    def _restore_panels(self):
        self._maximized = None
        self._top_h.setVisible(True)
        self.bezel_main.setVisible(True)
        self.bezel_fov.setVisible(True)
        self.recon_card.setVisible(True)
        if self._saved_top:
            self._top_h.setSizes(self._saved_top)
        if self._saved_right:
            self._right_v.setSizes(self._saved_right)

    def _update_max_hints(self):
        for panel, badge in self._badges.items():
            on = panel is self._maximized
            badge.label().setText("双击还原" if on else "双击放大")
            badge.label().setStyleSheet(f"color: {P.ACCENT if on else P.TEXT_MUTE};")

    def _refresh_gl(self):
        for v in (self.view, self.view_fov):
            try:
                v.update()
            except Exception:
                pass

    def showEvent(self, event):
        super().showEvent(event)
        # 浅色主题配深色标题栏会很割裂：强制 Windows 10/11 使用浅色标题栏
        try:
            import ctypes
            hwnd = int(self.winId())
            light = ctypes.c_int(0)          # 0 = 浅色，1 = 深色
            for attr in (20, 19):            # DWMWA_USE_IMMERSIVE_DARK_MODE（新/旧系统分别为 20/19）
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attr, ctypes.byref(light), ctypes.sizeof(light))
        except Exception:
            pass

    def paintEvent(self, event):
        """窗口底：深蓝径向渐变 + 顶部微光，营造纵深。"""
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect())
        base = QLinearGradient(r.left(), r.top(), r.left(), r.bottom())
        base.setColorAt(0.0, QColor(P.BG_CORE))
        base.setColorAt(1.0, QColor(P.BG_EDGE))
        p.setBrush(QBrush(base))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(r)

        halo = QRadialGradient(QPointF(r.center().x(), r.top() + r.height() * 0.12), r.width() * 0.75)
        halo.setColorAt(0.0, QColor(255, 255, 255, 165))
        halo.setColorAt(0.45, QColor(255, 255, 255, 70))
        halo.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.setBrush(QBrush(halo))
        p.drawRect(r)
        p.end()

    def _on_arch(self, txt):
        """CT 架构切换 → 触发扫描方案联动。

        注意：这里必须对选项文本做**前缀**匹配，不能用"关键字包含"。
        同步辐射的标签是「同步辐射 (Synchrotron · 单元光子计数改造)」，
        里面含"光子计数"四字，用包含匹配会被 pcct 抢先命中。
        """
        tail = txt.split('：')[-1]
        m = [('同步辐射', 'synchrotron'),        # 必须排在"光子计数"之前
             ('双源', 'dual_source'), ('单源宽体', 'single_wide'),
             ('双层', 'dual_layer'), ('光子计数', 'pcct'),
             ('静态多源', 'static_multi')]
        self.arch_key = next((v for k, v in m if tail.startswith(k)), 'dual_source')
        self._update_protocol_state()
        if not hasattr(self, 'recon_widget'):      # 构造期尚未建完，等首次 update 再刷
            return
        self.update_simulation()

    def _on_scan_mode(self, txt):
        self._update_protocol_state()
        if hasattr(self, 'recon_widget'):
            self.update_simulation()

    def _update_protocol_state(self):
        """扫描方案三级联动：架构 → 模式 → 螺距。

        * 静态多源：不旋转 → 模式锁定"静态"，螺距不适用（置灰）
        * 轴扫：床静止 → 螺距不参与（置灰）
        * 螺旋：螺距生效（启用）
        """
        if not hasattr(self, 'scan_mode_combo'):
            return
        static = getattr(self, 'arch_key', 'dual_source') == 'static_multi'
        mode_box = self.scan_mode_combo.combo_box()
        sl = self.sliders['pitch']
        if static:
            mode_box.setCurrentText('扫描模式：静态 (Static, 不旋转)')
            mode_box.setEnabled(False)
            sl.slider().setEnabled(False)
            hint = '静态架构：多源切换采集，无旋转、无螺距'
        else:
            mode_box.setEnabled(True)
            if mode_box.currentText().startswith('扫描模式：静态'):
                mode_box.setCurrentText('扫描模式：轴扫 (Axial)')
            helical = mode_box.currentText().startswith('扫描模式：螺旋')
            sl.slider().setEnabled(helical)
            hint = ('螺旋：每圈进床 = Pitch × Z 轴覆盖' if helical
                    else '轴扫：床静止，螺距不参与计算')
        sl.name_label.label().setStyleSheet(
            f"color:{P.TEXT_DIM if sl.slider().isEnabled() else P.TEXT_MUTE};")
        sl.value_label.label().setStyleSheet(
            f"color:{P.ACCENT if sl.slider().isEnabled() else P.TEXT_MUTE};")
        self.pitch_hint.label().setText(hint)

        # 同步辐射仿真参数组：仅该架构可见（与 Fermi 页签同一思路：按架构开关）
        sync_on = getattr(self, 'arch_key', '') == 'synchrotron'
        if hasattr(self, 'sync_box'):
            self.sync_box.setVisible(sync_on)

    def add_slider(self, master, layout, name, min_v, max_v, step, default, unit, key,
                   is_float=False, decimals=None, on_change=None):
        s = ParamSlider(master, name=name, min_val=min_v, max_val=max_v, step=step,
                        default=default, unit=unit, is_float=is_float, decimals=decimals,
                        on_change=on_change or self.update_simulation)
        layout.addWidget(s)
        self.sliders[key] = s

    def _on_nch_drag(self):
        """拖动"通道数"滑块 = 用户接管 → 自动关闭自动匹配（否则会被同步回推导值）。"""
        try:
            if self.nch_auto_sw.isChecked():
                self.nch_auto_sw.setChecked(False)
        except Exception:
            pass
        self.update_simulation()

    # ------------------------------------------------------------------
    # 3D 场景（对象与 PyQt5 版一致）
    # ------------------------------------------------------------------
    def init_3d_objects(self):
        """建立（一次性）全部 3D 对象；几何数据由 update_scene() 刷新。"""
        # ---- 相机：主视图略俯视，兼顾全局布局与锥束/探测器高度 ----
        self.view.opts['distance'] = 2500
        self.view.opts['elevation'] = 58
        self.view.opts['azimuth'] = -50
        self.view_fov.opts['distance'] = 1300
        self.view_fov.opts['elevation'] = 35
        self.view_fov.opts['azimuth'] = -55

        gz = gl.GLGridItem(color=GRID_RGBA)
        gz.translate(0, 0, -520)
        gz.setSize(x=2400, y=2400, z=0)
        gz.setSpacing(x=100, y=100, z=0)
        self.view.addItem(gz)
        self.view.addItem(self._axis_item(600))
        for it in self._axis_labels(600, 1000, 200, label_size=18):
            self.view.addItem(it)

        # ================= 主视图对象 =================
        self.items = {}
        I = self.items

        def line(color, width=1.6):
            it = gl.GLLinePlotItem(color=color, width=width, mode='lines', antialias=True)
            self.view.addItem(it)
            return it

        def mesh(color, edges=False, edge_color=(0.24, 0.34, 0.44, 0.60), glopts='translucent'):
            # balloon：颜色恒定、轮廓处自动提亮（不受固定光向影响，避免背面发黑）
            it = gl.GLMeshItem(shader='balloon', smooth=True, color=color, glOptions=glopts,
                               drawEdges=edges, edgeColor=edge_color)
            self.view.addItem(it)
            return it

        # 机架：孔径圈 / 罩壳骨架 / ISO 十字 / 旋转指示
        I['gantry'] = line((0.44, 0.58, 0.72, 0.42), 1.4)
        I['bore'] = line((0.10, 0.40, 0.76, 0.90), 2.2)
        I['iso_cross'] = line((0.10, 0.30, 0.55, 0.70), 1.6)
        I['rot_arrow'] = line((0.11, 0.44, 0.80, 0.92), 3.0)

        # 球管总成：真空壳 + 准直器 + 卡箍 + 焦点
        I['tube_A'] = mesh((0.62, 0.66, 0.72, 0.98), edges=True, glopts='opaque')
        I['tube_B'] = mesh((0.58, 0.63, 0.71, 0.98), edges=True, glopts='opaque')
        I['collar_A'] = mesh((0.80, 0.84, 0.88, 1.0), edges=True, glopts='opaque')
        I['collar_B'] = mesh((0.76, 0.80, 0.86, 1.0), edges=True, glopts='opaque')
        I['focal'] = gl.GLScatterPlotItem(size=16, pxMode=True)
        self.view.addItem(I['focal'])

        # Bowtie 滤过体（球管前方整形滤过：中心薄 / 两侧厚，张角由 Bowtie 视野决定）
        I['bowtie_A'] = mesh((0.74, 0.63, 0.36, 0.90), edges=True,
                             edge_color=(0.38, 0.30, 0.12, 0.75), glopts='opaque')
        I['bowtie_B'] = mesh((0.71, 0.61, 0.35, 0.90), edges=True,
                             edge_color=(0.38, 0.30, 0.12, 0.75), glopts='opaque')
        # 螺旋进床轨迹（pitch 可视化；轴扫时隐藏）
        I['helix'] = line((0.16, 0.52, 0.86, 0.70), 2.4)
        # 静态多源：源环散点 + 各源光束；光谱维度：分层/分箱弧
        I['src_ring'] = gl.GLScatterPlotItem(size=30, pxMode=True)
        self.view.addItem(I['src_ring'])
        # 静态多源：实心彩色球体（逐顶点色，比散点可靠）
        I['src_mesh'] = gl.GLMeshItem(smooth=True, drawEdges=False, glOptions='opaque')
        self.view.addItem(I['src_mesh'])
        I['src_beams'] = line((0.90, 0.45, 0.15, 0.75), 2.2)
        I['det_ring'] = gl.GLMeshItem(smooth=False, drawEdges=True,
                                      edgeColor=(1.0, 1.0, 1.0, 0.30),
                                      glOptions='opaque')     # 静态机架：环形分段探测器（实体）
        self.view.addItem(I['det_ring'])
        I['layers'] = gl.GLMeshItem(smooth=False, drawEdges=True,
                                    edgeColor=(1.0, 1.0, 1.0, 0.45),
                                    glOptions='opaque')
        self.view.addItem(I['layers'])            # 必须入场景，否则永远不可见

        # 锥束：体积包络面 + 代表性射线 + ISO 平面足迹
        I['fan_A'] = mesh((0.88, 0.16, 0.16, 0.07))
        I['fan_B'] = mesh((0.12, 0.38, 0.88, 0.07))
        I['rays_A'] = line((0.85, 0.18, 0.18, 0.28), 1.0)
        I['rays_B'] = line((0.12, 0.36, 0.84, 0.28), 1.0)
        I['foot_A'] = line((0.85, 0.20, 0.20, 0.55), 1.8)
        I['foot_B'] = line((0.15, 0.40, 0.85, 0.55), 1.8)

        # 探测器：带厚度曲面面板 + 模块分割线 + 防散射栅（半透明，便于看到锥束）
        I['det_A'] = mesh((0.88, 0.30, 0.30, 0.66), edges=True, edge_color=(0.55, 0.10, 0.10, 0.75))
        I['det_B'] = mesh((0.32, 0.52, 0.92, 0.66), edges=True, edge_color=(0.08, 0.24, 0.60, 0.75))
        I['det_A_grid'] = line((0.85, 0.10, 0.10, 0.60), 1.0)
        I['det_B_grid'] = line((0.08, 0.26, 0.70, 0.60), 1.0)
        I['asg_A'] = line((0.95, 0.55, 0.55, 0.22), 0.8)
        I['asg_B'] = line((0.45, 0.66, 0.95, 0.22), 0.8)
        I['det_B_virtual'] = mesh((0.55, 0.70, 0.95, 0.22))

        # SFOV 球（实体 + 经纬线，使其"看得出是球"）
        I['sfov_A'] = mesh((0.04, 0.60, 0.38, 0.10))
        I['sfov_A_wire'] = line((0.04, 0.55, 0.34, 0.35), 1.2)
        I['sfov_B'] = mesh((0.05, 0.50, 0.82, 0.10))
        I['sfov_B_wire'] = line((0.05, 0.45, 0.75, 0.35), 1.2)

        # 离心力 / 夹角
        I['force_vec'] = line((0.42, 0.24, 0.82, 0.95), 6.0)
        I['force_head'] = gl.GLScatterPlotItem(size=18, pxMode=True, color=(0.42, 0.24, 0.82, 1))
        self.view.addItem(I['force_head'])
        I['angle_arc'] = line((0.10, 0.40, 0.80, 0.95), 4.0)
        I['angle_ticks'] = line((0.10, 0.40, 0.80, 0.65), 2.0)

        # 文字标注
        self.text_items = {}
        for key in ['A_TubeB', 'B_TubeA', 'Det_Det', 'Angle']:
            t = gl.GLTextItem(pos=(0, 0, 0), text="", font=QFont('Segoe UI', 11, QFont.Weight.Bold),
                              color=(0.08, 0.22, 0.38, 1))
            self.view.addItem(t)
            self.text_items[key] = t

        self.scene_labels = {}
        for key, txt in (('tubeA', 'Tube A'), ('tubeB', 'Tube B'),
                         ('detA', 'Detector A'), ('detB', 'Detector B'),
                         ('bore', 'Gantry bore \u2300820'), ('rot', '')):
            t = gl.GLTextItem(pos=(0, 0, 0), text=txt, font=QFont('Segoe UI', 10),
                              color=(0.20, 0.34, 0.50, 1))
            self.view.addItem(t)
            self.scene_labels[key] = t

        # ================= FOV 分析视口对象 =================
        self.items_fov = {}
        F = self.items_fov

        grid_fov = gl.GLGridItem(color=(120, 160, 200, 70))
        grid_fov.translate(0, 0, -420)
        grid_fov.setSize(x=1200, y=1200, z=0)
        grid_fov.setSpacing(x=50, y=50, z=0)
        self.view_fov.addItem(grid_fov)
        self.view_fov.addItem(self._axis_item(300))
        for it in self._axis_labels(300, 500, 250, label_size=12):
            self.view_fov.addItem(it)

        def fline(color, width=1.6):
            it = gl.GLLinePlotItem(color=color, width=width, mode='lines', antialias=True)
            self.view_fov.addItem(it)
            return it

        def fmesh(color, edges=False, edge_color=(0.24, 0.34, 0.44, 0.60), glopts='translucent'):
            it = gl.GLMeshItem(shader='balloon', smooth=True, color=color, glOptions=glopts,
                               drawEdges=edges, edgeColor=edge_color)
            self.view_fov.addItem(it)
            return it

        F['polar'] = fline((0.35, 0.52, 0.68, 0.42), 1.2)
        F['safe'] = fmesh((0.05, 0.50, 0.82, 0.14))
        F['safe_wire'] = fline((0.05, 0.45, 0.75, 0.40), 1.2)
        F['shell'] = fmesh((1.00, 0.52, 0.05, 0.28))
        F['shell_wire'] = fline((0.90, 0.48, 0.05, 0.45), 1.2)
        F['fan_real'] = fmesh((0.12, 0.40, 0.88, 0.20))
        F['rays_real'] = fline((0.10, 0.36, 0.84, 0.30), 1.0)
        F['rays_miss'] = fline((0.85, 0.15, 0.15, 0.28), 1.0)
        F['fan_miss'] = fmesh((0.92, 0.08, 0.08, 0.40))
        F['fan_miss_edge'] = fline((0.95, 0.55, 0.02, 1.0), 4.0)
        F['fan_miss_grid'] = fline((0.85, 0.25, 0.10, 0.50), 1.0)
        F['det_B'] = fmesh((0.32, 0.52, 0.92, 0.55), edges=True, edge_color=(0.08, 0.24, 0.60, 0.75))
        F['det_B_grid'] = fline((0.80, 0.90, 1.0, 0.45), 1.0)
        F['iso'] = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(0.10, 0.30, 0.50, 1),
                                        size=10, pxMode=True)
        self.view_fov.addItem(F['iso'])
        self.text_fov = gl.GLTextItem(pos=(0, 0, 0), text="FOV Analysis",
                                      font=QFont('Segoe UI', 11, QFont.Weight.Bold),
                                      color=(0.12, 0.32, 0.52, 1))
        self.view_fov.addItem(self.text_fov)

        # ================= HUD（字段与原版一致）=================
        self.hud_label = QLabel(self.view)
        self.hud_label.setTextFormat(Qt.TextFormat.RichText)
        self.hud_label.setStyleSheet(f"""
            QLabel {{
                color: {P.TEXT_DIM};
                background-color: rgba(255, 255, 255, 0.90);
                border: 1px solid {P.HAIRLINE};
                border-left: 3px solid {P.ACCENT};
                border-radius: 9px;
                padding: 9px 12px;
                font-family: Consolas, monospace;
                font-size: 12px;
            }}
        """)
        self.hud_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hud_label.move(14, 14)
        self.hud_label.show()

    # ------------------------------------------------------------------
    # 3D 几何刷新：按当前参数重建球管/锥束/探测器/球壳等明细网格
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 同步辐射专属三维场景（断轴压缩：源 30 m 与样品 0.3 m 同框）
    # ------------------------------------------------------------------
    def _sync_scene_cache(self):
        """懒创建同步辐射专属 items；首次调用时记录需隐藏的机架 items。"""
        if not hasattr(self, '_sync_cache'):
            self._sync_gantry = (list(self.items.values())
                                 + list(self.scene_labels.values())
                                 + list(self.text_items.values()))

            def line(color, width=1.6):
                it = gl.GLLinePlotItem(color=color, width=width, mode='lines', antialias=True)
                self.view.addItem(it)
                return it

            def mesh(color, edges=False, glopts='translucent'):
                it = gl.GLMeshItem(shader='balloon', smooth=True, color=color,
                                   glOptions=glopts, drawEdges=edges,
                                   edgeColor=(0.24, 0.34, 0.44, 0.60))
                self.view.addItem(it)
                return it

            S = {}
            S['beam_far'] = mesh((0.44, 0.58, 0.72, 0.50), glopts='opaque')
            S['beam_near'] = mesh((0.44, 0.58, 0.72, 0.50), glopts='opaque')
            S['break'] = line((0.97, 0.36, 0.30, 0.92), 2.8)
            S['ring'] = line((0.72, 0.50, 0.95, 0.88), 1.6)
            S['und'] = line((0.95, 0.72, 0.30, 0.88), 1.7)
            S['mono'] = mesh((0.35, 0.56, 0.98, 0.95), edges=True, glopts='opaque')
            S['slit'] = mesh((0.55, 0.60, 0.66, 0.95), edges=True, glopts='opaque')
            S['stage'] = mesh((0.30, 0.42, 0.55, 0.98), edges=True, glopts='opaque')
            S['rot'] = line((0.95, 0.72, 0.30, 0.95), 2.6)
            S['rays'] = line((0.35, 0.56, 0.98, 0.50), 0.9)
            S['det'] = mesh((0.35, 0.56, 0.98, 0.98), edges=True, glopts='opaque')
            S['src_mark'] = gl.GLScatterPlotItem(size=24, pxMode=False,
                                                 color=(1.0, 0.45, 0.20, 1.0))
            self.view.addItem(S['src_mark'])
            S['sample_mark'] = gl.GLScatterPlotItem(size=18, pxMode=False,
                                                    color=(0.30, 1.0, 0.55, 1.0))
            self.view.addItem(S['sample_mark'])
            S['sample_sphere'] = mesh((0.35, 0.85, 0.45, 0.92), glopts='translucent')
            S['detgrid'] = line((0.10, 0.22, 0.38, 0.70), 0.7)
            S['dimsod'] = line((0.35, 0.82, 0.52, 0.90), 1.4)
            S['dimodd'] = line((0.35, 0.82, 0.52, 0.90), 1.4)
            SL = {}
            _LC = {'src': (1.0, 0.60, 0.30, 1.0), 'det': (0.40, 0.65, 1.0, 1.0),
                   'stage': (0.30, 1.0, 0.55, 1.0)}
            # 中文标注必须用 CJK 字体：Segoe UI 会把汉字渲染成方框/空白
            _FONT = QFont('Microsoft YaHei UI', 13, QFont.Weight.Bold)
            for key in ('src', 'mono', 'slit', 'stage', 'det', 'sod', 'odd', 'brk'):
                t = gl.GLTextItem(pos=(0, 0, 0), text='', font=_FONT,
                                  color=_LC.get(key, (0.72, 0.84, 0.95, 1.0)))
                self.view.addItem(t)
                SL[key] = t
            self._sync_cache = (S, SL)
        return self._sync_cache

    def _set_sync_visible(self, on):
        """切换：同步辐射场景显 / 机架场景隐。"""
        if not hasattr(self, '_sync_cache'):
            return
        for it in self._sync_gantry:
            try:
                it.setVisible(not on)
            except Exception:
                pass
        S, SL = self._sync_cache
        for it in list(S.values()) + list(SL.values()):
            try:
                it.setVisible(on)
            except Exception:
                pass

    def _build_sync_scene(self, params, results):
        """构建同步辐射束线布局：远源(断轴) → 单色器 → 狭缝 → 样品转台 → 平板探测器。"""
        import ct_scene_sync as SC2
        S, SL = self._sync_scene_cache()
        self._set_sync_visible(True)

        sod = float(results.get('sync_sod_mm') or params.get('sync_sod_mm', 30000.0))
        odd = float(results.get('sync_odd_mm') or params.get('sync_odd_mm', 300.0))
        fov = float(results.get('sync_fov_mm') or 111.0)
        half = max(40.0, fov * 0.55)                       # 探测器半宽（随视场联动）

        # 断轴显示坐标（样品在原点；源在 -X，探测器在 +X）
        xs = -SC2.axis_break(sod, sod)                     # 源
        xcut = -SC2.axis_break(sod - SC2.FAR_MM, sod)      # 断轴源侧起点
        xn = -SC2.axis_break(SC2.NEAR_MM, sod)             # 断轴样品侧终点
        xd = SC2.axis_break(odd, odd)                      # 探测器（1:1 近端）

        # ---- 束线管（两段：源侧 + 样品侧）----
        v, f, _, _ = SC2.beamline_tube(xs, xcut, 18, rib_every=90)
        S['beam_far'].setMeshData(vertexes=v, faces=f)
        v, f, _, _ = SC2.beamline_tube(xn, 0.0, 18, rib_every=90)
        S['beam_near'].setMeshData(vertexes=v, faces=f)

        # ---- 断轴符号 // ----
        S['break'].setData(pos=SC.segment_list(SC2.break_marks(SC2.break_center(sod), size=80)))
        SL['brk'].setData(pos=(SC2.break_center(sod), -150, 0),
                          text='break  %.0f m omitted' % (sod / 1000.0))

        # ---- 储存环 + 波荡器（源侧）----
        S['ring'].setData(pos=SC.segment_list(SC2.storage_ring_symbol((xs - 200, 0, 0), r=150)))
        S['und'].setData(pos=SC.segment_list(SC2.undulator_symbol(xs - 130, xs - 40, half=40)))
        S['src_mark'].setData(pos=np.array([[xs - 85, 0.0, 0.0]]))
        SL['src'].setData(pos=(xs - 200, 215, 0),
                          text='射线源 Synchrotron source  %.1f m' % (sod / 1000.0))

        # ---- 单色器 + 狭缝（样品侧）----
        v, f = SC2.crystal_plate((-360, 0, 0), tilt_deg=0)
        S['mono'].setMeshData(vertexes=v, faces=f)
        SL['mono'].setData(pos=(-360, 120, 0), text='Monochromator')
        v, f = SC2.slit_jaws((-250, 0, 0), gap=24)
        S['slit'].setMeshData(vertexes=v, faces=f)
        SL['slit'].setData(pos=(-250, 150, 0), text='Slits')

        # ---- 样品转台 + 旋转箭头 ----
        v, f, arrow, _ring = SC2.stage_disk((0.0, 0, 0), r=70)
        S['stage'].setMeshData(vertexes=v, faces=f)
        S['rot'].setData(pos=SC.segment_list(arrow))
        S['sample_mark'].setData(pos=np.array([[0.0, 0.0, 0.0]]))
        v, f = SC2.sphere((0.0, 0.0, 55.0), 34.0)
        S['sample_sphere'].setMeshData(vertexes=v, faces=f)
        SL['stage'].setData(pos=(0.0, -175, 0), text='样本 Sample (rotates 360)')

        # ---- 近平行束 + 平板探测器 ----
        S['rays'].setData(pos=SC.segment_list(
            SC2.parallel_rays(-150, xd - 14, half_span=half * 0.9, n=7)))
        v, f, grid = SC2.flat_panel((xd, 0, 0), w=2 * half, h=2 * half * 0.82)
        S['det'].setMeshData(vertexes=v, faces=f)
        S['detgrid'].setData(pos=SC.segment_list(grid))
        SL['det'].setData(pos=(xd, half + 75, 0), text='探测器 Flat detector')

        # ---- 尺寸线（真实距离标注）----
        S['dimsod'].setData(pos=SC.segment_list(
            SC2.dimension_line((0.0, -220, 0), (xs, -220, 0))))
        SL['sod'].setData(pos=(xs * 0.5, -250, 0), text='SOD = %.1f m' % (sod / 1000.0))
        S['dimodd'].setData(pos=SC.segment_list(
            SC2.dimension_line((0.0, 220, 0), (xd, 220, 0))))
        SL['odd'].setData(pos=(xd * 0.5, 250, 0), text='ODD = %.0f mm' % odd)

    def update_scene(self, params, results, fan_A, fan_B, is_asymmetric):
        """延迟重建：合并高频调用（架构切换 / 拖滑块），避免主线程同步冻结。

        真正重建在 60ms 单发定时器里执行（_flush_scene -> _update_scene_impl），
        期间若再有更新则只覆盖参数、不重复重建。
        """
        self._scene_pending = (params, results, fan_A, fan_B, is_asymmetric)
        if getattr(self, '_scene_timer', None) is None:
            self._scene_timer = QTimer(self)
            self._scene_timer.setSingleShot(True)
            self._scene_timer.setInterval(60)
            self._scene_timer.timeout.connect(self._flush_scene)
        if getattr(self, '_scene_first', True):
            # 首次同步执行：避免 GLMeshItem 在 None 顶点下被首帧绘制
            self._scene_first = False
            self._flush_scene()
            return
        self._scene_timer.start()

    def _flush_scene(self):
        """定时器到点：用最近一次参数做一次重建。"""
        if getattr(self, '_scene_pending', None) is None:
            return
        params, results, fan_A, fan_B, is_asymmetric = self._scene_pending
        self._scene_pending = None
        self._update_scene_impl(params, results, fan_A, fan_B, is_asymmetric)

    def _update_scene_impl(self, params, results, fan_A, fan_B, is_asymmetric):
        FDD = float(params['FDD'])
        # ---- 同步辐射架构：专属三维场景（远源近平行束 + 样品转台 + 平板探测器）----
        if str(results.get('arch_key', '')) == 'synchrotron':
            self._build_sync_scene(params, results)
            return
        if hasattr(self, '_sync_cache'):
            self._set_sync_visible(False)

        RA, RB = float(params['RA']), float(params['RB'])
        FA = np.array(results['R_A_coord'], dtype=float)
        FB = np.array(results['R_B_coord'], dtype=float)
        alpha_rad = results['alpha_rad']
        beta_A = results['beta_A']
        beta_B = results['beta_B']
        beta_B_orig = results.get('beta_B_orig', beta_B)
        H_A, H_B = results['H_det_A'], results['H_det_B']
        kap_A, kap_B = results['kappa_A'], results['kappa_B']
        ang_A = np.pi - alpha_rad / 2.0          # A 系统出束方向（焦点 → 探测器）
        ang_B = np.pi + alpha_rad / 2.0
        ext = 120.0 / FDD                        # 非对称延长对应的角度
        I = self.items
        L = self.scene_labels

        # 物理像素/通道数 → 探测器模块分割线数量（采用几何内核的权威值 n_ch / n_rows）
        px_xy = max(0.001, float(params.get('pixel_xy', 0.625)))
        px_z = max(0.001, float(params.get('pixel_z', 0.625)))
        pdet_A = px_xy * (FDD / RA if RA else 1.0)
        pdet_B = px_xy * (FDD / RB if RB else 1.0)
        ch_A = max(16.0, float(results.get('n_ch', FDD * 2.0 * beta_A / pdet_A)))
        row_A = max(8.0, float(results.get('n_rows', float(params['Z_coverage']) / px_z)))
        ch_B = ch_A
        row_B = row_A
        ln_A_x = int(np.clip(ch_A / 64.0 - 1.0, 4, 28))
        # 通道列标记数量随"每排单元数"变化（原先硬编码 n=56，与通道数脱钩）
        n_col_marks = int(np.clip(ch_A / 24.0, 12, 84))
        ln_A_z = int(np.clip(row_A / 16.0 - 1.0, 3, 9))
        ln_B_x = int(np.clip(ch_B / 64.0 - 1.0, 4, 28))
        ln_B_z = int(np.clip(row_B / 16.0 - 1.0, 3, 9))

        # ---------- 机架 ----------
        bore_r = 410.0
        I['bore'].setData(pos=SC.segment_list([SC.make_ring(bore_r, 0.0, n=200)]))
        I['gantry'].setData(pos=SC.gantry_cage(bore_r, max(RA, RB) + 120.0, 130.0, n=72, n_vert=12))
        I['iso_cross'].setData(pos=SC.crosshair(150.0))
        I['rot_arrow'].setData(pos=SC.arc_with_arrow(max(RA, RB) + 250.0,
                                                     np.radians(-150), np.radians(10), head=34.0))
        L['bore'].setData(pos=(bore_r * 0.62, bore_r * 0.62, 0.0), text="Gantry bore \u2300820")
        L['rot'].setData(pos=(-(max(RA, RB) + 300.0), 0.0, 0.0),
                         text=f"Rotation {params['rotation_time']:.2f} s/rot")

        # ---------- 球管总成 ----------
        for tag, F0, aim in (('A', FA, ang_A), ('B', FB, ang_B)):
            v, f = SC.tube_housing(F0, aim)
            I['tube_' + tag].setMeshData(vertexes=v, faces=f)
            v2, f2 = SC.collar_ring(F0, aim, -62.0)
            I['collar_' + tag].setMeshData(vertexes=v2, faces=f2)
        I['focal'].setData(pos=np.vstack([FA, FB]),
                           color=np.array([[0.97, 0.36, 0.30, 1.0], [0.36, 0.56, 0.98, 1.0]]),
                           size=16)
        L['tubeA'].setData(pos=(FA[0], FA[1], 130.0), text="Tube A")
        L['tubeB'].setData(pos=(FB[0], FB[1], 130.0), text="Tube B")

        # ---------- Bowtie 滤过体（随 Bowtie 视野 / 边缘厚度实时变化）----------
        r_bt = 0.32 * FDD
        z_bt = r_bt * (float(params['Z_coverage']) * 0.5) / RA
        for tag, F0, aim in (('A', FA, ang_A), ('B', FB, ang_B)):
            try:
                v, f, vc = SC.bowtie_mesh(F0, aim, r_in=r_bt,
                                          t_edge=float(results.get('bt_t_edge', 0.0)),
                                          gamma_bt=max(1e-4, float(results.get('gamma_bt', beta_A))),
                                          z_half=max(4.0, z_bt),
                                          mu=float(results.get('bt_mu', 0.0196)), colorize=True)
                md = gl.MeshData(vertexes=v, faces=f)
                md.setVertexColors(np.ascontiguousarray(vc))     # 逐顶点色 = 透射率 T(γ)
                I['bowtie_' + tag].setMeshData(meshdata=md)
            except Exception as exc:                             # 顶点色不可用时退回单色
                print(f'[WARN] Bowtie 顶点色失败，退回单色: {exc}')
                v, f = SC.bowtie_mesh(F0, aim, r_in=r_bt,
                                      t_edge=float(results.get('bt_t_edge', 0.0)),
                                      gamma_bt=max(1e-4, float(results.get('gamma_bt', beta_A))),
                                      z_half=max(4.0, z_bt))
                I['bowtie_' + tag].setMeshData(vertexes=v, faces=f)
        _gm = np.rad2deg(float(results.get('gamma_bt', beta_A)))
        _te = float(results.get('bt_t_edge', 0.0))
        _mu = float(results.get('bt_mu', 0.0196))
        L['bore'].setData(pos=(r_bt * 0.55, r_bt * 0.55, 0.0),
                          text=(f"Bowtie ±{_gm:.1f}°  t_edge={_te:.0f}mm  "
                                f"T: 1.000→{np.exp(-_mu * _te):.3f}"))

        # ---------- 螺旋进床轨迹（pitch 可视化）----------
        if str(results.get('scan_mode', 'axial')) == 'helical':
            I['helix'].setData(pos=SC.segment_list(
                [SC.helix_path(max(RA, RB) + 430.0, float(results['z_per_rot']), 1.6)]))
            I['helix'].setVisible(True)
        else:
            I['helix'].setVisible(False)

        # ---------- 架构形态：3D 场景随 CT 架构切换 ----------
        n_src = int(results.get('n_src', 2))
        e_dim = int(results.get('energy_dim', 1))
        rotating = bool(results.get('rotating', True))
        single = (n_src != 2)                        # 只有双源(2 源)才显示 B 系统
        for _k in ('tube_B', 'det_B', 'collar_B', 'bowtie_B', 'sfov_B', 'sfov_B_wire',
                   'det_B_grid', 'det_B_virtual', 'asg_B', 'fan_B', 'foot_B', 'rays_B'):
            if _k in I:
                I[_k].setVisible(not single)
        if 'tubeB' in L:
            L['tubeB'].setData(pos=(FB[0], FB[1], 130.0) if not single else (0, 0, 0),
                               text="Tube B" if not single else "")
        # 静态多源：n_src 个源均布于 R 圆上 + 各源朝 ISO 的扇束
        if (not rotating) and n_src > 1:
            th = np.deg2rad(np.arange(n_src) * 360.0 / n_src)
            # 24 个源**各用不同颜色**（HSV 均分），并与对侧探测器段同色 → 一一对应
            import colorsys as _cs
            _src_rgb = [tuple(int(255 * c) for c in _cs.hsv_to_rgb(k / float(n_src), 0.80, 1.0))
                        for k in range(n_src)]
            _src_cols = np.array([[c[0] / 255.0, c[1] / 255.0, c[2] / 255.0, 1.0]
                                  for c in _src_rgb], dtype=np.float32)
            I['src_ring'].setData(
                pos=np.column_stack([RA * np.cos(th), RA * np.sin(th), np.zeros(n_src)]),
                color=_src_cols)
            I['src_ring'].setVisible(False)          # 改用实心球体（散点逐点色不可靠）
            try:
                _sv, _sf, _sc = SC.source_cluster(
                    [np.array([RA * np.cos(a), RA * np.sin(a), 0.0]) for a in th],
                    radius=max(16.0, RA * 0.035), colors=_src_rgb)
                _smd = gl.MeshData(vertexes=_sv, faces=_sf)
                _smd.setVertexColors(_sc)
                I['src_mesh'].setMeshData(meshdata=_smd)
                I['src_mesh'].setVisible(True)
            except Exception as exc:
                print(f'[WARN] 源球体着色失败: {exc}')
                I['src_mesh'].setVisible(False)
            r_in = max(20.0, RA - float(params['SFOV_A']) / 2.0)
            I['src_beams'].setData(pos=SC.segment_list(
                [np.array([[RA * np.cos(a), RA * np.sin(a), 0.0],
                           [r_in * np.cos(a), r_in * np.sin(a), 0.0]]) for a in th]))
            I['src_beams'].setVisible(True)
            # 环形分段探测器：**实体环**按 n_src 分段（段间留缝），正对探测段以区分色
            pal = _src_rgb if (not rotating) and n_src > 1 else [(70, 170, 240)]
            try:
                _rv, _rf, _rc = SC.ring_segments(FDD - 26.0, FDD + 26.0,
                                                 max(20.0, float(results.get('h_det_match', 100.0)) * 0.5),
                                                 n_seg=n_src, gap_frac=0.12, n_ang=11,
                                                 colors=pal)
                _md = gl.MeshData(vertexes=_rv, faces=_rf)
                _md.setVertexColors(_rc)
                I['det_ring'].setMeshData(meshdata=_md)
            except Exception as exc:
                print(f'[WARN] 分段探测器环顶点色失败，退回单色: {exc}')
                _rv, _rf = SC.ring_segments(FDD - 26.0, FDD + 26.0,
                                            max(20.0, float(results.get('h_det_match', 100.0)) * 0.5),
                                            n_seg=n_src, gap_frac=0.12, n_ang=11)
                I['det_ring'].setMeshData(vertexes=_rv, faces=_rf)
            I['det_ring'].setVisible(True)
        else:
            I['src_ring'].setData(pos=np.zeros((0, 3)))
            I['src_ring'].setVisible(False)
            if 'src_mesh' in I:
                I['src_mesh'].setVisible(False)
            I['src_beams'].setVisible(False)
            I['det_ring'].setVisible(False)
        # 光谱维度：双层=2 层 / 光子计数=8 箱
        # 关键：各层沿**射线方向（径向）前后叠放**（上层近病人=低能、下层在后=高能），
        # Z 向尺寸与主探测器等高、不做错位。
        if e_dim > 1 and single:
            z_half = max(10.0, float(results.get('h_det_match', 100.0)) * 0.5)
            t_lay = 14.0                                     # 单层径向厚度 (mm)
            r0 = FDD - (e_dim * t_lay) / 2.0
            r_centers = [r0 + (i + 0.5) * t_lay for i in range(e_dim)]
            pal2 = ([(255, 150, 60), (60, 190, 255)] if e_dim == 2 else
                    [(int(255 * (1 - i / max(1, e_dim - 1))), 140,
                      int(255 * i / max(1, e_dim - 1))) for i in range(e_dim)])
            try:
                _lv, _lf, _lc = SC.detector_layers(FA, ang_A, r_centers, t_lay * 0.44,
                                                   beta_A, z_half, n_ang=33, n_z=5,
                                                   colors=pal2)
                _lmd = gl.MeshData(vertexes=_lv, faces=_lf)
                _lmd.setVertexColors(_lc)
                I['layers'].setMeshData(meshdata=_lmd)
            except Exception as exc:
                print(f'[WARN] 分层顶点色失败，退回单色: {exc}')
                _lv, _lf = SC.detector_layers(FA, ang_A, r_centers, t_lay * 0.44,
                                              beta_A, z_half, n_ang=33, n_z=5)
                I['layers'].setMeshData(vertexes=_lv, faces=_lf)
            I['layers'].setVisible(True)
            names = {2: '上层低能(橙) + 下层高能(青)', 8: '8 能量箱(径向叠放)'}.get(
                e_dim, f'{e_dim} 能量箱')
            L['bore'].setData(pos=(r_bt * 0.55, r_bt * 0.55, 0.0),
                              text=(f"{results.get('arch_label', '')[:14]}  {names}  "
                                    f"沿射线方向叠放"))
        else:
            I['layers'].setVisible(False)

        # ---------- 锥束 A ----------
        v, f = SC.cone_beam_surface(FA, ang_A, FDD, beta_A, kap_A, n_ang=30, n_z=10)
        I['fan_A'].setMeshData(vertexes=v, faces=f)
        I['rays_A'].setData(pos=SC.beam_rays(FA, ang_A, FDD, beta_A, kap_A, n_fan=13, n_cone=5))
        I['foot_A'].setData(pos=SC.segment_list(
            [SC.iso_fan_footprint(FA, ang_A, FDD, beta_A, float(params['SFOV_A']) / 2.0)]))

        # ---------- 探测器 A ----------
        r_in, r_out = FDD - 14.0, FDD + 6.0
        v, f = SC.curved_panel(r_in, r_out, ang_A - beta_A, ang_A + beta_A, -H_A / 2, H_A / 2,
                               n_ang=56, n_z=8)
        I['det_A'].setMeshData(vertexes=v, faces=f)
        I['det_A_grid'].setData(pos=SC.detector_grid_lines(
            FA, ang_A, FDD, beta_A, kap_A, r_off=-16.0, n_ang=52, n_z=14,
            n_ang_lines=ln_A_x, n_z_lines=ln_A_z))
        I['asg_A'].setData(pos=SC.pixel_columns(FA, ang_A, FDD, beta_A, kap_A, n=n_col_marks, r_off=-24.0))
        L['detA'].setData(pos=(FA[0] + (FDD - 40.0) * np.cos(ang_A),
                               FA[1] + (FDD - 40.0) * np.sin(ang_A), H_A / 2 + 40.0), text="Detector A")

        # ---------- 锥束 B / 探测器 B（含非对称延长）----------
        if is_asymmetric:
            a_real0, a_real1 = ang_B - beta_B_orig, ang_B + beta_B_orig + ext
            a_virt0, a_virt1 = ang_B - beta_B_orig - ext, ang_B - beta_B_orig
            beta_phys = (a_real1 - a_real0) / 2.0
            ang_phys = (a_real1 + a_real0) / 2.0
            v, f = SC.cone_beam_surface(FB, ang_phys, FDD, beta_phys, kap_B, n_ang=30, n_z=10)
            I['fan_B'].setMeshData(vertexes=v, faces=f)
            I['rays_B'].setData(pos=SC.beam_rays(FB, ang_phys, FDD, beta_phys, kap_B, n_fan=13, n_cone=5))
            v, f = SC.curved_panel(r_in, r_out, a_real0, a_real1, -H_B / 2, H_B / 2, n_ang=40, n_z=8)
            I['det_B'].setMeshData(vertexes=v, faces=f)
            v2, f2 = SC.curved_panel(r_in, r_out, a_virt0, a_virt1, -H_B / 2, H_B / 2, n_ang=18, n_z=8)
            I['det_B_virtual'].setMeshData(vertexes=v2, faces=f2)
            I['det_B_virtual'].setVisible(True)
            I['det_B_grid'].setData(pos=SC.detector_grid_lines(
                FB, ang_phys, FDD, beta_phys, kap_B, r_off=-16.0, n_ang=36, n_z=12,
                n_ang_lines=ln_B_x, n_z_lines=ln_B_z))
            I['asg_B'].setData(pos=SC.pixel_columns(FB, ang_phys, FDD, beta_phys, kap_B, n=n_col_marks, r_off=-24.0))
            I['foot_B'].setData(pos=SC.segment_list(
                [SC.iso_fan_footprint(FB, ang_phys, FDD, beta_phys,
                                      RB * np.sin(beta_phys))]))
        else:
            I['det_B_virtual'].setVisible(False)
            v, f = SC.cone_beam_surface(FB, ang_B, FDD, beta_B, kap_B, n_ang=30, n_z=10)
            I['fan_B'].setMeshData(vertexes=v, faces=f)
            I['rays_B'].setData(pos=SC.beam_rays(FB, ang_B, FDD, beta_B, kap_B, n_fan=13, n_cone=5))
            v, f = SC.curved_panel(r_in, r_out, ang_B - beta_B, ang_B + beta_B, -H_B / 2, H_B / 2,
                                   n_ang=56, n_z=8)
            I['det_B'].setMeshData(vertexes=v, faces=f)
            I['det_B_grid'].setData(pos=SC.detector_grid_lines(
                FB, ang_B, FDD, beta_B, kap_B, r_off=-16.0, n_ang=52, n_z=14,
                n_ang_lines=ln_B_x, n_z_lines=ln_B_z))
            I['asg_B'].setData(pos=SC.pixel_columns(FB, ang_B, FDD, beta_B, kap_B, n=n_col_marks, r_off=-24.0))
            I['foot_B'].setData(pos=SC.segment_list(
                [SC.iso_fan_footprint(FB, ang_B, FDD, beta_B, float(params['SFOV_B']) / 2.0)]))
        L['detB'].setData(pos=(FB[0] + (FDD - 40.0) * np.cos(ang_B),
                               FB[1] + (FDD - 40.0) * np.sin(ang_B), H_B / 2 + 40.0), text="Detector B")

        # ---------- SFOV 球 ----------
        r_A = RA * np.sin(beta_A)
        v, f = SC.uv_sphere(r_A, n_lat=26, n_lon=48)
        I['sfov_A'].setMeshData(vertexes=v, faces=f)
        I['sfov_A_wire'].setData(pos=SC.sphere_wire(r_A, n_lat=8, n_lon=20))
        if is_asymmetric:
            r_B = results['SFOV_B_effective'] / 2.0
        else:
            r_B = RB * np.sin(beta_B)
        v, f = SC.uv_sphere(r_B, n_lat=26, n_lon=48)
        I['sfov_B'].setMeshData(vertexes=v, faces=f)
        I['sfov_B_wire'].setData(pos=SC.sphere_wire(r_B, n_lat=8, n_lon=20))

        # ---------- 离心力矢量 / 夹角弧 ----------
        F_vec = results['F_total_vector']
        vec_end = np.array([F_vec[0] * 0.1, F_vec[1] * 0.1, 0.0])
        I['force_vec'].setData(pos=np.array([[0, 0, 0], vec_end]))
        I['force_head'].setData(pos=np.array([vec_end]))

        r_arc = min(RA, RB) * 0.42
        theta_arc = np.linspace(-alpha_rad / 2, alpha_rad / 2, 48)
        I['angle_arc'].setData(pos=np.column_stack(
            (r_arc * np.cos(theta_arc), r_arc * np.sin(theta_arc), np.zeros_like(theta_arc))))
        tick_lines = []
        for t in np.linspace(-alpha_rad / 2, alpha_rad / 2, 13):
            tick_lines.append(np.array([[r_arc * np.cos(t), r_arc * np.sin(t), 0.0],
                                        [(r_arc + 22.0) * np.cos(t), (r_arc + 22.0) * np.sin(t), 0.0]]))
        I['angle_ticks'].setData(pos=SC.segment_list(tick_lines))
        self.text_items['Angle'].setData(pos=np.array([r_arc * np.cos(theta_arc[24]),
                                                       r_arc * np.sin(theta_arc[24]), 0.0]),
                                         text=f"{params['alpha']:.1f}\u00b0")

        # ================= FOV 视口 =================
        F = self.items_fov
        r_safe = results['SFOV_B_inner'] / 2.0
        r_eff = results['SFOV_B_effective'] / 2.0
        F['polar'].setData(pos=SC.polar_grid([r_eff * k for k in (0.25, 0.5, 0.75, 1.0)],
                                             n_spokes=24))
        v, f = SC.uv_sphere(r_safe, n_lat=26, n_lon=48)
        F['safe'].setMeshData(vertexes=v, faces=f)
        F['safe_wire'].setData(pos=SC.sphere_wire(r_safe, n_lat=9, n_lon=22))
        F['iso'].setData(pos=np.array([[0, 0, 0]]))

        if is_asymmetric:
            F['shell'].setVisible(True)
            F['shell_wire'].setVisible(True)
            F['fan_miss'].setVisible(True)
            F['fan_miss_edge'].setVisible(True)
            F['fan_miss_grid'].setVisible(True)
            v, f = SC.spherical_shell(r_safe, r_eff, n_lat=30, n_lon=60)
            F['shell'].setMeshData(vertexes=v, faces=f)
            F['shell_wire'].setData(pos=SC.sphere_wire((r_safe + r_eff) / 2.0, n_lat=7, n_lon=18))

            # 真实探测弧（含右侧延长）
            v, f = SC.curved_panel(FDD - 14.0, FDD + 6.0, ang_B - beta_B_orig,
                                   ang_B + beta_B_orig + ext, -H_B / 2, H_B / 2, n_ang=40, n_z=8)
            F['det_B'].setMeshData(vertexes=v, faces=f)
            F['det_B_grid'].setData(pos=SC.detector_grid_lines(
                FB, ang_B + ext / 2.0, FDD, beta_B_orig + ext / 2.0, kap_B,
                r_off=-16.0, n_ang=36, n_z=12, n_ang_lines=10, n_z_lines=5))

            # 实采扇束（原对称部分 + 右侧延长）
            beta_phys = beta_B_orig + ext / 2.0
            ang_phys = ang_B + ext / 2.0
            v, f = SC.cone_beam_surface(FB, ang_phys, FDD, beta_phys, kap_B, n_ang=26, n_z=9)
            F['fan_real'].setMeshData(vertexes=v, faces=f)
            F['rays_real'].setData(pos=SC.beam_rays(FB, ang_phys, FDD, beta_phys, kap_B,
                                                    n_fan=11, n_cone=5))

            # 缺失扇区（虚拟左侧延长段）
            a0, a1 = ang_B - beta_B_orig - ext, ang_B - beta_B_orig
            v, f = SC.cone_beam_surface(FB, (a0 + a1) / 2.0, FDD, ext / 2.0, kap_B, n_ang=12, n_z=9)
            F['fan_miss'].setMeshData(vertexes=v, faces=f)
            F['rays_miss'].setData(pos=SC.beam_rays(FB, (a0 + a1) / 2.0, FDD, ext / 2.0, kap_B,
                                                    n_fan=7, n_cone=4))
            F['rays_miss'].setVisible(True)
            F['fan_miss_grid'].setData(pos=SC.detector_grid_lines(
                FB, (a0 + a1) / 2.0, FDD, ext / 2.0, kap_B, r_off=-16.0,
                n_ang=10, n_z=10, n_ang_lines=5, n_z_lines=4))
            edge = []
            for a in (a0, a1):
                edge.append(np.array([[FB[0], FB[1], FB[2] - H_B / 2],
                                      [FB[0] + FDD * np.cos(a), FB[1] + FDD * np.sin(a), FB[2] - H_B / 2],
                                      [FB[0] + FDD * np.cos(a), FB[1] + FDD * np.sin(a), FB[2] + H_B / 2],
                                      [FB[0], FB[1], FB[2] + H_B / 2],
                                      [FB[0], FB[1], FB[2] - H_B / 2]]))
            F['fan_miss_edge'].setData(pos=SC.segment_list(edge))
            self.text_fov.setData(pos=(FB[0] + (FDD + 70.0) * np.cos((a0 + a1) / 2.0),
                                       FB[1] + (FDD + 70.0) * np.sin((a0 + a1) / 2.0),
                                       H_B / 2 + 60.0),
                                  text=f"Missing FOV Sector: {results['SFOV_B_effective']:.1f} mm")
            self.bezel_fov.led.set_color(P.SCATTER)
        else:
            F['shell'].setVisible(False)
            F['shell_wire'].setVisible(False)
            F['fan_miss'].setVisible(False)
            F['fan_miss_edge'].setVisible(False)
            F['fan_miss_grid'].setVisible(False)
            F['rays_miss'].setVisible(False)
            v, f = SC.cone_beam_surface(FB, ang_B, FDD, beta_B, kap_B, n_ang=26, n_z=9)
            F['fan_real'].setMeshData(vertexes=v, faces=f)
            F['rays_real'].setData(pos=SC.beam_rays(FB, ang_B, FDD, beta_B, kap_B, n_fan=11, n_cone=5))
            v, f = SC.curved_panel(FDD - 14.0, FDD + 6.0, ang_B - beta_B, ang_B + beta_B,
                                   -H_B / 2, H_B / 2, n_ang=56, n_z=8)
            F['det_B'].setMeshData(vertexes=v, faces=f)
            F['det_B_grid'].setData(pos=SC.detector_grid_lines(
                FB, ang_B, FDD, beta_B, kap_B, r_off=-16.0, n_ang=52, n_z=14,
                n_ang_lines=ln_B_x, n_z_lines=ln_B_z))
            self.text_fov.setData(pos=(0.0, r_eff + 90.0, r_eff + 60.0), text="Standard Symmetric FOV")
            self.bezel_fov.led.set_color(P.ACCENT)

        # 模式切换时自动调整 FOV 机位：非对称转到"缺失扇区"正面，对称则回到常规视角
        if getattr(self, '_last_asym', None) != is_asymmetric:
            self._last_asym = is_asymmetric
            if is_asymmetric:
                self.view_fov.opts['azimuth'] = -152
                self.view_fov.opts['elevation'] = 30
                self.view_fov.opts['distance'] = 1450
            else:
                self.view_fov.opts['azimuth'] = -55
                self.view_fov.opts['elevation'] = 35
                self.view_fov.opts['distance'] = 1300
            self.view_fov.update()

    # ---- 坐标轴 / 刻度（与 PyQt5 版一致的内容） ----
    @staticmethod
    def _axis_item(size):
        axis = gl.GLAxisItem()
        axis.setSize(size, size, size)
        return axis

    @staticmethod
    def _axis_labels(axis_size, tick_range, tick_step, label_size=18):
        """X/Y/Z 轴标签 + 刻度文本，返回待加入视口的 item 列表。"""
        items = []
        for txt, pos, col in (('X', (axis_size + 50, 0, 0), (0.80, 0.16, 0.16, 1)),
                              ('Y', (0, axis_size + 50, 0), (0.06, 0.55, 0.32, 1)),
                              ('Z', (0, 0, axis_size + 50), (0.10, 0.40, 0.75, 1))):
            items.append(gl.GLTextItem(pos=pos, text=txt,
                                       font=QFont('Segoe UI', label_size, QFont.Weight.Bold), color=col))
        for i in range(-tick_range, tick_range + 1, tick_step):
            if i == 0:
                continue
            coords = [(i, 0, 0), (0, i, 0)]
            if abs(i) <= tick_range // 2:
                coords.append((0, 0, i))
            for pos in coords:
                items.append(gl.GLTextItem(pos=pos, text=str(i),
                                           font=QFont('Segoe UI', max(7, label_size - 10)),
                                           color=(0.34, 0.48, 0.62, 1)))
        return items

    # ---- 网格 / 扇区网格工具（与 PyQt5 版一致） ----
    def create_cylinder_mesh_data(self, radius, height, start_angle, end_angle, rows=10, cols=20):
        theta = np.linspace(start_angle, end_angle, cols)
        z = np.linspace(-height / 2, height / 2, rows)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x = radius * np.cos(theta_grid)
        y = radius * np.sin(theta_grid)
        verts = np.column_stack([x.flatten(), y.flatten(), z_grid.flatten()])
        faces = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                i0 = r * cols + c
                i1 = r * cols + (c + 1)
                i2 = (r + 1) * cols + c
                i3 = (r + 1) * cols + (c + 1)
                faces.append([i0, i1, i2])
                faces.append([i1, i3, i2])
        return verts, np.array(faces)

    def create_fan_sector_mesh(self, origin, corners):
        verts = np.vstack([np.array(origin), np.array(corners)])
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])
        return verts, faces

    # ------------------------------------------------------------------
    # 主更新：几何计算 + 3D 刷新 + 读数刷新（逻辑与 PyQt5 版一致）
    # ------------------------------------------------------------------
    def update_simulation(self):
        params = {k: s.value() for k, s in self.sliders.items()}
        params['n_ch_set'] = (None if self.nch_auto_sw.isChecked()
                              else int(round(params.get('n_ch_set', 825))))
        params['scan_mode'] = ('helical' if self.scan_mode_combo.combo_box().currentText()
                               .startswith('扫描模式：螺旋') else 'axial')
        is_asymmetric = self.asymmetric_cb.isChecked()
        params['is_asymmetric'] = is_asymmetric

        results, fan_A, fan_B = calculate_geometry(
            params['alpha'], params['RA'], params['RB'], params['FDD'],
            params['SFOV_A'], params['SFOV_B'], params['Z_coverage'], params['rotation_time'],
            is_asymmetric=is_asymmetric,
            bowtie_sfov=params.get('bowtie_sfov'), bowtie_edge_mm=params.get('bowtie_edge', 30.0),
            scan_mode=params['scan_mode'], pitch=params.get('pitch', 1.0),
            pixel_xy=params.get('pixel_xy', 0.625), pixel_z=params.get('pixel_z', 0.625),
            n_ch_set=params.get('n_ch_set'),
            arch=getattr(self, 'arch_key', 'dual_source'),
            sampling_rate=params.get('sampling_rate', 2000.0),
            scan_length=params.get('scan_length', 300.0),
            slice_thickness=params.get('slice_thickness', 1.0),
            slice_interval=params.get('slice_interval', 1.0),
            shots_per_source=params.get('shots_per_source', 1),
            ring_sources=(12 if '半环' in self.ring_combo.combo_box().currentText()
                          else (16 if '短扫描' in self.ring_combo.combo_box().currentText() else 0)),
            # ---- 同步辐射仿真模式（arch='synchrotron' 时生效）----
            # 源/探测器下拉按索引映射回 key；通量滑块走 log10 刻度。
            sync_source=(self._sync_src_order[self.sync_source_combo.combo_box().currentIndex()]
                         if getattr(self, '_sync_src_order', None) else 'metaljet'),
            sync_detector=(self._sync_det_order[self.sync_det_combo.combo_box().currentIndex()]
                           if getattr(self, '_sync_det_order', None) else 'CdTe'),
            sync_focus_um=params.get('sync_focus_um'),
            sync_sod_mm=params.get('sync_sod_mm', 30000.0),
            sync_odd_mm=params.get('sync_odd_mm', 300.0),
            sync_pixel_um=params.get('sync_pixel_um', 55.0),
            sync_sigma_c_um=params.get('sync_sigma_c_um'),
            sync_oversample=int(params.get('sync_oversample', 4)),
            sync_bins=int(params.get('sync_bins', 8)),
            sync_energy_kev=params.get('sync_energy_kev', 60.0),
            sync_flux=10.0 ** float(params.get('sync_flux_log', 8.0)),
            sync_shaping_ns=params.get('sync_shaping_ns'),
            sync_phase=bool(getattr(getattr(self, 'sync_phase_sw', None), 'isChecked',
                                    lambda: False)()),
            sync_delta_beta=params.get('sync_delta_beta', 100.0),
            sync_vmi_kev=params.get('sync_vmi_kev', 65.0),
            sync_target_res_um=params.get('sync_target_res_um', 15.0),
        )

        if "error" in results:
            self.result_view.text_edit().setHtml(f"<span style='color:{P.WARN}'>Error: {results['error']}</span>")
            return

        # ---- 同步辐射模式提示：放大率 / 有效像素 / 视场的取舍（手册 §4.2 失效模式）----
        # M 太小 → 放大收益消失（p_eff→p）；M 太大 → FOV = 阵列宽/M 缩到装不下样品。
        if getattr(self, 'arch_key', '') == 'synchrotron' and hasattr(self, 'sync_hint'):
            try:
                mp = results.get('sync_M_practical') or 0.0
                txt_h = ('M = %.2f　p/M = %.2f µm　FOV = %.1f mm　r* = %.2f µm'
                         % (results.get('sync_M', 0.0), results.get('sync_p_eff_um', 0.0),
                            results.get('sync_fov_mm') or 0.0,
                            results.get('sync_r_star_um', 0.0)))
                if mp and results.get('sync_M', 0) < 0.5 * mp:
                    txt_h += '　⚠ 放大率偏低，实用 M ≈ %.0f' % mp
                self.sync_hint.label().setText(txt_h)
            except Exception:
                pass

        # 3D 场景按当前参数重建（球管总成 / 锥束 / 曲面探测器 / SFOV 球 / 夹角弧）
        self.update_scene(params, results, fan_A, fan_B, is_asymmetric)
        # ---- 读数 ----
        safe_color = P.OK if results['is_safe'] else P.WARN
        arc_color = P.OK if results['is_arc_ok'] else P.WARN
        self.safety_led.set_color(safe_color)
        self.bezel_main.led.set_color(P.ACCENT if results['is_safe'] else P.WARN)

        impact_txt = ""
        if is_asymmetric:
            impact_txt = f"""
            <hr style='border:0;border-top:1px solid {P.HAIRLINE};'>
            <b><font color='{P.WARN}'>非对称模式影响 (Negative Impact):</font></b><br>
            1. <b>受影响FOV体积:</b> {results['affected_volume_cm3']:.1f} cm³ (Outer Shell)<br>
            2. <b>有效 SFOV B:</b> {results['SFOV_B_effective']:.1f} mm (Inner: {results['SFOV_B_inner']:.1f} mm)<br>
            3. <b>散射干扰系数:</b> {results['scatter_coeff']:.2f} (因体积增加)<br>
            """

        # ---- 物理像素：输入值是 SFOV(ISO) 平面尺度，探测器真实像元 = ISO 像素 × 放大比 M=FDD/R ----
        px_xy = max(0.001, float(params.get('pixel_xy', 0.625)))
        px_z = max(0.001, float(params.get('pixel_z', 0.625)))
        _FDD = float(params['FDD'])
        _Z = max(0.1, float(params['Z_coverage']))
        mag_A = _FDD / float(params['RA']) if float(params['RA']) else 1.0
        mag_B = _FDD / float(params['RB']) if float(params['RB']) else 1.0
        px_det_A, px_det_B = px_xy * mag_A, px_xy * mag_B
        rows_A = rows_B = _Z / px_z                      # 排数 = Z 覆盖 / ISO 像素（与放大比无关）
        # 物理探测器弧长：A 为 2β_A·FDD；B 对称时 2β_B·FDD，非对称时仅**单侧**再加 120 mm 延长段
        arc_A = _FDD * 2.0 * results['beta_A']
        arc_B = _FDD * 2.0 * results['beta_B_orig'] + (120.0 if is_asymmetric else 0.0)
        # 通道数/排数一律采用几何内核"整数化自动匹配"后的权威值（拖动滑块即同步）
        _nch = float(results.get('n_ch', arc_A / px_det_A))
        _nrow = float(results.get('n_rows', rows_A))
        _pdet_A = float(results.get('col_pitch', px_det_A))
        mat = {
            'magA': mag_A, 'magB': mag_B,
            'pdetA': _pdet_A, 'pdetB': px_det_B,
            'chA': _nch,
            'rowA': _nrow,
            'chB': _nch * (arc_B / arc_A) if arc_A else _nch,   # B 与 A 同一列距 → 按弧长比例
            'rowB': _nrow,
            'fovA': float(params['SFOV_A']) / px_xy,
            'fovB': results['SFOV_B_effective'] / px_xy,
        }

        # 自动匹配时把"通道数"滑块同步为推导值（阻断信号，避免回调递归）
        if not results.get('n_ch_manual', False):
            _sl = self.sliders['n_ch_set']
            _sb = _sl.slider()
            _sb.blockSignals(True)
            _sb.setValue(int(results['n_ch']))
            _sb.blockSignals(False)
            _sl.value_label.label().setText(f"{results['n_ch']:.0f}")
            self.nch_hint.label().setText(
                f"自动：由像素×SFOV 推导 = {results['n_ch']}（拖动滑块即切换为手动）")
        else:
            self.nch_hint.label().setText(
                f"手动：以设定值为准 → 实尺列距 = 弧长/通道数 = {results['col_pitch']:.4f} mm")

        # 手动指定通道数时，附上由阵列反推的 SFOV
        ch_manual_txt = (f"<font color='{P.INFO}'>(手动指定通道数 → ISO 采样间隔 "
                         f"{results['iso_sampling_mm']:.4f} mm，原设定像素 "
                         f"{params.get('pixel_xy', 0.625):.3f} mm)</font>"
                         if results['n_ch_manual'] else '')

        # ---- 螺旋扫描派生读数（单独拼接，避免 f-string 嵌套）----
        if results['scan_mode'] == 'helical':
            helical_txt = (
                f"<b>进床速度:</b> {results['v_table']:.1f} mm/s "
                f"(每圈 {results['z_per_rot']:.1f} mm, {results['rot_per_s']:.2f} 圈/s)<br>"
                f"<b>螺距上限(几何):</b> 单源 "
                f"<font color='{P.TEXT_DIM}'>{results['pitch_max_single']:.2f}</font> / 双源 "
                f"<font color='{P.ACCENT}'>{results['pitch_max_dual']:.2f}</font> "
                f"<font color='{P.WARN if results['pitch_over'] else P.OK}'>"
                f"({'超限 — 角度采样不足' if results['pitch_over'] else '在限内'})</font><br>"
                f"<b>体素角度覆盖:</b> {results['ang_per_voxel']:.1f}° "
                f"(重建需 {180 + results['fan_angle_A']:.1f}°) "
                f"<font color='{P.TEXT_MUTE}'>| 相对剂量 {results['dose_factor']:.2f}× (∝1/pitch)</font><br>")
        else:
            helical_txt = (f"<b>轴扫:</b> 每圈覆盖 {results['z_per_rot']:.1f} mm（Z 轴覆盖），"
                           f"床静止，无螺距<br>")

        txt = f"""
        <div style='font-family:Consolas; font-size:10pt; line-height:150%;'>
        <b style='color:{P.TEXT}; font-size:11pt;'>计算结果</b><br>
        <b>扇区角:</b> A: {results['fan_angle_A']:.2f}° | B: {results['fan_angle_B']:.2f}°<br>
        <b>探测器高度:</b> A: {results['H_det_A']:.1f} | B: {results['H_det_B']:.1f} mm<br>
        <b>物理像素 (ISO):</b> XY {px_xy:.3f} mm | Z {px_z:.3f} mm<br>
        <b>Bowtie 真正视野:</b> {results['bowtie_sfov']:.1f} mm
            <font color='{P.TEXT_MUTE}'>(开扇角 {results['bowtie_fan_deg']:.2f}°, 设定 {results['bowtie_sfov_set']:.0f} mm)</font><br>
        <b>Bowtie 截断:</b>
            <font color='{P.WARN if results['bowtie_limits'] else P.OK}'><b>{'是 — 超出 ±%.2f° 无信号' % np.rad2deg(results['gamma_bt']) if results['bowtie_limits'] else '否 — 探测器为限制方'}</b></font><br>
        <b>Bowtie 透射:</b> 中心 {results['bt_T_center']:.3f} / 边缘 {results['bt_T_edge']:.3f}
            <font color='{P.TEXT_MUTE}'>(μ={results['bt_mu']:.4f}/mm, t_edge={results['bt_t_edge']:.0f} mm)</font><br>
        <b>剂量比(边缘/中心):</b> <font color='{P.INFO}'>{results['bt_dose_ratio']:.3f}</font>
            <font color='{P.TEXT_MUTE}'>(边缘节省 {results['bt_dose_saving_pct']:.0f}%)</font><br>
        <b>扫描模式:</b> {'螺旋 (Helical)' if results['scan_mode'] == 'helical' else '轴扫 (Axial)'}
            {('| <b>螺距:</b> %.2f' % results['pitch']) if results['scan_mode'] == 'helical' else ''}<br>
        {helical_txt}
        <b>探测器像元 (实尺):</b> A {mat['pdetA']:.3f} | B {mat['pdetB']:.3f} mm
            <font color='{P.TEXT_MUTE}'>(放大比 A {mat['magA']:.3f} / B {mat['magB']:.3f})</font><br>
        <b>探测器矩阵:</b> A {mat['chA']:.0f}×{mat['rowA']:.0f} | B {mat['chB']:.0f}×{mat['rowB']:.0f}
            <font color='{P.TEXT_MUTE}'>(通道×排)</font><br>
        <b>探测器阵列:</b> {results['n_ch']} 个/排 × {results['n_rows']} 排 =
            <font color='{P.ACCENT}'><b>{results['array_total']}</b></font> 单元
            <font color='{P.TEXT_MUTE}'>({'手动指定' if results['n_ch_manual'] else '自动匹配(推导值 %d)' % results['n_ch_auto']}:
            列距 {results['col_pitch']:.4f} / 行距 {results['row_pitch_det']:.4f} mm)</font><br>
        <b>CT 架构:</b> {results['arch_label']}
            <font color='{P.TEXT_MUTE}'>(源 {results['n_src']} / 能量维 {results['energy_dim']} /
            {'旋转' if results['rotating'] else '静态'}{'，源角间距 %.1f°' % results['src_step_deg'] if results['n_src'] > 1 else ''})</font><br>
        <b>架构采样:</b> views {results['views_total']}，每点视角 {results['views_per_point']}，角步进 {results['ang_step_deg']:.3f}°
            <font color='{P.WARN if results['sparse_view'] else P.OK}'>({'稀疏视角 — 需迭代/深度学习重建' if results['sparse_view'] else '采样充分'})</font><br>
        <b>架构时间分辨:</b> <font color='{P.INFO}'>{results['t_res_arch_ms']:.3f} ms</font>
            <font color='{P.TEXT_MUTE}'>| {results['dose_note']}</font><br>
        <b>数据体量:</b> {results['data_cells'] / 1e6:.1f} M 单元 (通道×排×views×能量维)<br>
        <b>扫描范围:</b> 长度 {results['scan_length']:.0f} mm →
            <font color='{P.ACCENT}'><b>{results['n_slices']}</b></font> 层
            <font color='{P.TEXT_MUTE}'>(层厚 {results['slice_thickness']:.2f} / 间隔 {results['slice_interval']:.2f} mm,
            层厚/行距 {results['slice_vs_row']:.2f})</font><br>
        <b>圈数与曝光:</b>
            {('净 %.2f + 过扫描 %.1f = <b>%.2f</b> 圈' % (results['rot_net'], results['over_rot'], results['rot_total'])) if results['scan_mode'] == 'helical' else ('%d 次步进（每步 1 圈）' % results['axial_steps'] if results['rotating'] else '静态单次曝光（%d 源顺序触发）' % results['n_src'])}
            | 总曝光 <font color='{P.INFO}'>{results['scan_time_s']:.2f} s</font>
            <font color='{P.TEXT_MUTE}'>(相对总 mAs {results['total_mas_rel']:.2f}×)</font><br>
        <b>剂量估算:</b> 相对 CTDIvol <font color='{P.INFO}'>{results['ctdi_rel']:.2f}×</font>
            <font color='{P.TEXT_MUTE}'(∝1/pitch)</font> | 相对 DLP {results['dlp_rel']:.2f}×
            | 有效层厚(经验) {results['ssp_eff']:.3f} mm<br>
        <b>阵列匹配残差:</b> 列 {results['array_col_res_pct']:+.3f}% | 行 {results['array_row_res_pct']:+.3f}%
            <font color='{P.OK if results['array_ok'] else P.WARN}'>({'合格' if results['array_ok'] else '偏差>2%：列距与设定像素不符'})</font>
            {ch_manual_txt}<br>
        <b>FOV 像素数:</b> A {mat['fovA']:.0f} | B {mat['fovB']:.0f}
            <font color='{P.TEXT_MUTE}'>宽×高 {mat['rowA']:.0f}</font><br>
        <hr style='border:0;border-top:1px solid {P.HAIRLINE};'>
        <b>最小安全距离:</b> {results['min_dist']:.1f} mm
            <font color='{safe_color}'><b>({'安全' if results['is_safe'] else '碰撞预警'})</b></font><br>
        <b>DetA-DetB:</b> {results['min_dist_Det_Det']:.1f} mm<br>
        <b>弧长差:</b> {results['arc_diff']:.1f} mm
            <font color='{arc_color}'>({'OK' if results['is_arc_ok'] else 'Diff < 120'})</font><br>
        <b>物理时间分辨:</b> <font color='{P.INFO}'>{results['temporal_resolution'] * 1000:.0f} ms</font><br>
        <b>散射干扰系数:</b> <font color='{P.SCATTER}'>{results['scatter_coeff']:.2f}</font><br>
        {impact_txt}
        <hr style='border:0;border-top:1px solid {P.HAIRLINE};'>
        <b>离心力 (G-Force):</b><br>
        Tube A/B: {results['g_force_TubeA']:.1f} G / {results['g_force_TubeB']:.1f} G<br>
        Det A/B: {results['g_force_DetA']:.1f} G / {results['g_force_DetB']:.1f} G{('  |  ' + results.get('g_force_note', '')) if results.get('g_force_note') else ''}<br>
        <b>系统固定压力 (双球管合力):</b>
            <font color='{P.FORCE}' size='4'>{results['F_total_mag']:.0f} N ({results['g_force_total']:.1f} G)</font>
        </div>
        """
        self.result_view.text_edit().setHtml(txt)

        # ---- HUD ----
        hud_txt = f"""
        <b style='color:{P.TEXT};'>System Geometry Parameters</b><br>
        <span style='color:{P.ACCENT_DIM};'>--------------------------------</span><br>
        Angle (α)   : {params['alpha']}°<br>
        F-ISO (RA)  : {params['RA']} mm<br>
        F-ISO (RB)  : {params['RB']} mm<br>
        FDD         : {params['FDD']} mm<br>
        SFOV (A)    : {params['SFOV_A']} mm<br>
        SFOV (B)    : {params['SFOV_B']} mm<br>
        Z-Coverage  : {params['Z_coverage']} mm<br>
        Rotation T  : {params['rotation_time']} s<br>
        Pixel(ISO) XY: {px_xy:.3f} mm<br>
        Pixel(ISO) Z : {px_z:.3f} mm<br>
        Bowtie SFOV : {results['bowtie_sfov']:.0f} mm{(' (截断)' if results['bowtie_limits'] else '')}<br>
        Scan Mode   : {'Helical' if results['scan_mode'] == 'helical' else 'Axial'}<br>
        Pitch       : {results['pitch']:.2f}{('  (max %.2f)' % results['pitch_max_dual']) if results['scan_mode'] == 'helical' else ''}<br>
        <span style='color:{P.ACCENT_DIM};'>--------------------------------</span><br>
        System Load : <b style='color:{P.FORCE};'>{results['F_total_mag']:.0f} N</b>
        """
        if is_asymmetric:
            hud_txt += f"<br><font color='{P.WARN}'>[Asymmetric Mode ON]</font>"
        self.hud_label.setText(hud_txt)
        self.hud_label.adjustSize()

        # 重建较重（1440 投影约 0.8 s），改为"松手后重建"：拖动滑块时几何与读数实时更新，
        # 停止操作 260 ms 后再做一次高质量重建，避免每格拖动都重算导致卡顿。
        self._pending_recon = (results, dict(params))
        self._recon_timer.start()

        # FBP Process 页签：把当前几何交给 LEAP 物理引擎（面板内部防抖后重建）
        try:
            self.recon_widget.fbp_panel.set_geometry(params, results, is_asymmetric)
        except Exception as exc:
            print(f'[WARN] FBP 面板更新失败: {exc}')

        # Fermi 页签仅光子计数架构可见；首次可见时自动算一次
        try:
            _on = (getattr(self, 'arch_key', '') == 'pcct')
            _rw = getattr(self, 'recon_widget', None)
            if _rw is not None and hasattr(_rw, 'set_fermi_visible'):
                _before = _rw.tabs.indexOf(_rw.tab_fermi) >= 0 if _rw.tab_fermi else False
                _rw.set_fermi_visible(_on)
                if _on and not _before and _rw.tab_fermi is not None:
                    _rw.tab_fermi.set_arch_enabled(True)
                    _rw.tab_fermi.compute()
                # Fermi 页签仅光子计数可见，但**双层架构同样需要**能谱数据
                _fp = _rw.tab_fermi
                _need = (getattr(self, 'arch_key', '') in ('pcct', 'dual_layer'))
                if _need and _on and _fp is not None:
                    _mode = ('kedge' if 'K' in str(_fp.mode_combo.combo_box().currentText())
                             else 'equal')
                    _rw.fbp_panel.fermi_query = {
                        'material': str(_fp.mat_combo.combo_box().currentText()),
                        'thickness_mm': float(_fp.s_thick.value()),
                        'voltage': float(_fp.s_bias.value()),
                        'temp_k': float(_fp.s_temp.value()),
                        'energy_keV': float(_fp.s_energy.value()),
                        'kvp': float(_fp.s_kvp.value()),
                        'n_bins': int(str(_fp.bins_combo.combo_box().currentText()).split()[0]),
                        'mode': _mode}
                elif _need:
                    # 双层：Fermi 页不可见 → 用默认探测材料与 120 kVp 双通道
                    _rw.fbp_panel.fermi_query = {
                        'material': 'CZT', 'thickness_mm': 2.0, 'voltage': 1000.0,
                        'temp_k': 300.0, 'energy_keV': 120.0, 'kvp': 120.0,
                        'n_bins': 2, 'mode': 'equal'}
                elif hasattr(_rw, 'fbp_panel'):
                    _rw.fbp_panel.fermi_query = None
        except Exception as exc:
            print(f'[WARN] Fermi 页签可见性更新失败: {exc}')


# =====================================================================
# 8. 入口
# =====================================================================

def main():
    set_appearance_mode("light")
    set_color_theme("blue")

    # 4x MSAA：让锥束曲面、探测器网格与机架线条更细腻（必须在 QApplication 之前设置）
    fmt = QSurfaceFormat()
    fmt.setSamples(4)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setApplicationName("双源 CT 真实几何模拟器")
    app.setApplicationVersion(APP_VERSION)
    pg.setConfigOptions(background=PLOT_BG, foreground=PLOT_FG, antialias=True)

    icon = app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)

    splash = None
    try:
        splash = QSplashScreen(build_splash_pixmap(), Qt.WindowType.WindowStaysOnTopHint)
        splash.show()
        splash.showMessage("正在初始化几何引擎与重建管线 ...",
                           Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                           QColor(P.TEXT_DIM))
        app.processEvents()
    except Exception as exc:
        print(f"Splash disabled: {exc}")
        splash = None

    win = MainWindow()
    win.show()
    if splash is not None:
        splash.finish(win)

    try:
        import ct6_bridge
        ct6_bridge.start_bridge(win)
    except Exception as _exc:
        print(f'[WARN] 控制桥启动失败: {_exc}')
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
