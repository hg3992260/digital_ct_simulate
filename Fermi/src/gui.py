import sys
import os
import re
import numpy as np

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QListWidget, QDoubleSpinBox, 
                            QPushButton, QGroupBox, QFormLayout, QMessageBox, 
                            QInputDialog, QTabWidget, QGridLayout, QAbstractItemView, QScrollArea, QFileDialog,
                            QDialog, QCheckBox, QDialogButtonBox, QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView, QCompleter)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon

# Matplotlib integration
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
import xraylib # Import xraylib for atomic number lookups

def parse_atomic_composition(text: str) -> dict:
    value = (text or "").strip()
    if not value:
        raise ValueError("Composition is required.")

    items = {}

    if (":" in value) or ("=" in value):
        parts = re.split(r"[,\n;]+", value)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                sym, num = part.split(":", 1)
            else:
                sym, num = part.split("=", 1)
            sym = sym.strip()
            num = num.strip()
            if not re.fullmatch(r"[A-Z][a-z]?", sym):
                raise ValueError(f"Invalid element symbol: {sym}")
            try:
                coeff = float(num)
            except ValueError:
                raise ValueError(f"Invalid composition number: {num}")
            if coeff <= 0:
                raise ValueError("Composition numbers must be > 0.")
            items[sym] = items.get(sym, 0.0) + coeff
    else:
        value = re.sub(r"\s+", "", value)
        tokens = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)?", value)
        if not tokens:
            raise ValueError("Invalid composition format.")
        for sym, num in tokens:
            coeff = float(num) if num else 1.0
            if coeff <= 0:
                raise ValueError("Composition numbers must be > 0.")
            items[sym] = items.get(sym, 0.0) + coeff

    if not items:
        raise ValueError("Composition is required.")

    for sym in list(items.keys()):
        try:
            xraylib.SymbolToAtomicNumber(sym)
        except Exception:
            raise ValueError(f"Invalid element symbol: {sym}")

    total = float(sum(items.values()))
    if total <= 0:
        raise ValueError("Composition numbers must sum to > 0.")

    return {sym: val / total for sym, val in items.items()}

_XRAYLIB_ELEMENT_SYMBOLS = None

def get_xraylib_element_symbols() -> list:
    global _XRAYLIB_ELEMENT_SYMBOLS
    if _XRAYLIB_ELEMENT_SYMBOLS is not None:
        return _XRAYLIB_ELEMENT_SYMBOLS
    symbols = []
    for z in range(1, 119):
        try:
            sym = xraylib.AtomicNumberToSymbol(z)
        except Exception:
            continue
        if sym:
            symbols.append(sym)
    _XRAYLIB_ELEMENT_SYMBOLS = sorted(set(symbols), key=lambda s: (len(s), s))
    return _XRAYLIB_ELEMENT_SYMBOLS

def composition_text_to_formula(text: str) -> str:
    value = (text or "").strip()
    if not value:
        raise ValueError("Composition is required.")

    if (":" in value) or ("=" in value):
        parts = re.split(r"[,\n;]+", value)
        order = []
        coeffs = {}
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                sym, num = part.split(":", 1)
            else:
                sym, num = part.split("=", 1)
            sym = sym.strip()
            num = num.strip()
            if not re.fullmatch(r"[A-Z][a-z]?", sym):
                raise ValueError(f"Invalid element symbol: {sym}")
            try:
                coeff = float(num)
            except ValueError:
                raise ValueError(f"Invalid composition number: {num}")
            if coeff <= 0:
                raise ValueError("Composition numbers must be > 0.")
            if sym not in coeffs:
                order.append(sym)
                coeffs[sym] = 0.0
            coeffs[sym] += coeff

        if not order:
            raise ValueError("Composition is required.")

        for sym in order:
            try:
                xraylib.SymbolToAtomicNumber(sym)
            except Exception:
                raise ValueError(f"Invalid element symbol: {sym}")

        out = []
        for sym in order:
            coeff = coeffs[sym]
            if abs(coeff - 1.0) < 1e-12:
                out.append(sym)
            else:
                out.append(f"{sym}{coeff:g}")
        return "".join(out)

    value = re.sub(r"\s+", "", value)
    if not re.fullmatch(r"(?:[A-Z][a-z]?(?:[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)?)+", value):
        raise ValueError("Invalid composition format.")

    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)?", value)
    if not tokens:
        raise ValueError("Invalid composition format.")

    for sym, _ in tokens:
        try:
            xraylib.SymbolToAtomicNumber(sym)
        except Exception:
            raise ValueError(f"Invalid element symbol: {sym}")

    return value

# --- Scientific Plotting Style Configuration ---
# Set global parameters for publication-quality plots
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.2,        # Thicker axes
    'lines.linewidth': 2.0,       # Thicker lines
    'lines.markersize': 6,
    'legend.fontsize': 8,
    'legend.frameon': False,      # Clean legend
    'xtick.direction': 'in',      # Ticks pointing in
    'ytick.direction': 'in',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.autolayout': True     # Tight layout
})

# Add project root to path
# Handle frozen path (PyInstaller)
if getattr(sys, 'frozen', False):
    # If frozen, sys.executable is the app, but resources are in _MEIPASS
    base_path = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
    # When frozen, 'src' might not be a top-level package anymore if we just bundled gui.py
    # We need to ensure we can import our modules.
    # If bundled with --onefile, usually everything is extracted to _MEIPASS
    sys.path.append(base_path)
    
    # Also add the directory containing gui.py itself (in onedir mode)
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
else:
    # Running from source
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(base_path)

APP_ICON_PATH = os.path.normpath(os.path.join(base_path, "resources", "Atom.jpeg"))
if not os.path.exists(APP_ICON_PATH):
    APP_ICON_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resources", "Atom.jpeg"))
if not os.path.exists(APP_ICON_PATH):
    APP_ICON_PATH = ""

# Try imports with fallback for frozen environment structure
try:
    # Try importing from src first (for dev environment)
    from src.simulator import DetectorSimulator
    from src.translations import TRANSLATIONS
    from src.material import MaterialManager
except ImportError:
    # Fallback: Try importing directly (for flattened PyInstaller builds)
    try:
        from simulator import DetectorSimulator
        from translations import TRANSLATIONS
        from material import MaterialManager
    except ImportError as e:
        # Re-raise original to see what's wrong if fallback fails
        # Try one more thing: maybe we are inside 'src'
        try:
            import simulator
            import translations
            import material
            
            DetectorSimulator = simulator.DetectorSimulator
            TRANSLATIONS = translations.TRANSLATIONS
            MaterialManager = material.MaterialManager
        except ImportError as e2:
            raise ImportError(f"Could not import modules. Base path: {base_path}. Error: {e}. Fallback Error: {e2}")

from PyQt6.QtCore import Qt, pyqtSignal

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class MaterialResultWidget(QWidget):
    """
    A widget to display simulation results for a single material.
    Contains Stats and Tabs (Spectrum, Band Diagram).
    """
    # Signal to notify parent when tab changes
    tab_changed = pyqtSignal(int)

    def __init__(self, material_name, current_lang='en'):
        super().__init__()
        self.material_name = material_name
        self.current_lang = current_lang
        self.init_ui()
        
    def init_ui(self):
        # ... (Layout setup remains same)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Group Box
        self.group = QGroupBox(f"{TRANSLATIONS[self.current_lang]['results_title']}{self.material_name}")
        self.group.setStyleSheet("QGroupBox { font-weight: bold; color: #2c3e50; }")
        group_layout = QVBoxLayout(self.group)
        
        # Stats Label (Compact)
        self.stats_label = QLabel("Running...")
        self.stats_label.setStyleSheet("font-size: 10px; color: #555;")
        self.stats_label.setWordWrap(True)
        self.stats_label.setFixedHeight(60)
        
        # Save Button
        self.save_btn = QPushButton(TRANSLATIONS[self.current_lang]['save_image_btn'])
        self.save_btn.setFixedSize(80, 24)
        self.save_btn.setStyleSheet("font-size: 10px; padding: 2px;")
        self.save_btn.clicked.connect(self.save_current_plot)
        
        # Header Layout
        top_row = QHBoxLayout()
        top_row.addWidget(self.stats_label)
        top_row.addWidget(self.save_btn, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        
        group_layout.addLayout(top_row)
        
        # Tabs
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.on_tab_changed) # Connect signal
        
        # Spectrum Tab
        self.spectrum_tab = QWidget()
        spec_layout = QVBoxLayout(self.spectrum_tab)
        spec_layout.setContentsMargins(0,0,0,0)
        self.canvas_spectrum = MplCanvas(self, width=4, height=3, dpi=80)
        spec_layout.addWidget(self.canvas_spectrum)
        self.tabs.addTab(self.spectrum_tab, TRANSLATIONS[self.current_lang]['tab_spectrum'])
        
        # Band Tab
        self.band_tab = QWidget()
        band_layout = QVBoxLayout(self.band_tab)
        band_layout.setContentsMargins(0,0,0,0)
        self.canvas_band = MplCanvas(self, width=4, height=3, dpi=80)
        band_layout.addWidget(self.canvas_band)
        self.tabs.addTab(self.band_tab, TRANSLATIONS[self.current_lang]['tab_band'])
        
        # Transport Tab
        self.transport_tab = QWidget()
        trans_layout = QVBoxLayout(self.transport_tab)
        trans_layout.setContentsMargins(0,0,0,0)
        self.canvas_trans = MplCanvas(self, width=4, height=3, dpi=80)
        trans_layout.addWidget(self.canvas_trans)
        self.tabs.addTab(self.transport_tab, TRANSLATIONS[self.current_lang]['tab_transport'])
        
        # Polarization Tab
        self.polar_tab = QWidget()
        pol_layout = QVBoxLayout(self.polar_tab)
        pol_layout.setContentsMargins(0,0,0,0)
        self.canvas_pol = MplCanvas(self, width=4, height=3, dpi=80)
        pol_layout.addWidget(self.canvas_pol)
        self.tabs.addTab(self.polar_tab, TRANSLATIONS[self.current_lang]['tab_polarization'])

        # Attenuation Tab
        self.atten_tab = QWidget()
        atten_layout = QVBoxLayout(self.atten_tab)
        atten_layout.setContentsMargins(0,0,0,0)
        self.canvas_atten = MplCanvas(self, width=4, height=3, dpi=80)
        atten_layout.addWidget(self.canvas_atten)
        self.tabs.addTab(self.atten_tab, TRANSLATIONS[self.current_lang]['tab_attenuation'])

        # Temp Scan Tab
        self.scan_tab = QWidget()
        scan_layout = QVBoxLayout(self.scan_tab)
        scan_layout.setContentsMargins(0,0,0,0)
        self.canvas_scan = MplCanvas(self, width=4, height=3, dpi=80)
        scan_layout.addWidget(self.canvas_scan)
        self.tabs.addTab(self.scan_tab, TRANSLATIONS[self.current_lang]['tab_tscan'])
        
        # Flux Analysis Tab
        self.flux_tab = QWidget()
        flux_layout = QVBoxLayout(self.flux_tab)
        flux_layout.setContentsMargins(0,0,0,0)
        self.canvas_flux = MplCanvas(self, width=4, height=3, dpi=80)
        flux_layout.addWidget(self.canvas_flux)
        self.tabs.addTab(self.flux_tab, TRANSLATIONS[self.current_lang]['tab_flux'])
        
        group_layout.addWidget(self.tabs)
        layout.addWidget(self.group)

    def update_language(self, lang):
        self.current_lang = lang
        t = TRANSLATIONS[lang]
        
        self.group.setTitle(f"{t['results_title']}{self.material_name}")
        self.save_btn.setText(t['save_image_btn'])
        self.tabs.setTabText(0, t['tab_spectrum'])
        self.tabs.setTabText(1, t['tab_band'])
        self.tabs.setTabText(2, t['tab_transport'])
        self.tabs.setTabText(3, t['tab_polarization'])
        self.tabs.setTabText(4, t['tab_attenuation'])
        self.tabs.setTabText(5, t['tab_tscan'])
        self.tabs.setTabText(6, t['tab_flux'])
        
        # Redraw plots if data is available
        if hasattr(self, 'last_result_args'):
            self.update_results(*self.last_result_args)
        if hasattr(self, 'last_scan_data'):
            self.update_temp_scan(self.last_scan_data)

    def on_tab_changed(self, index):
        # Emit signal so parent can sync others
        self.tab_changed.emit(index)

    def set_tab_index(self, index):
        # Set tab without emitting signal to avoid loops
        self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex(index)
        self.tabs.blockSignals(False)

    def save_current_plot(self):
        """Save the currently visible plot to a file."""
        t = TRANSLATIONS[self.current_lang]
        current_widget = self.tabs.currentWidget()
        # Find the canvas in the current tab
        canvas = current_widget.findChild(MplCanvas)
        
        if canvas:
            # Default filename suggestion
            default_name = f"{self.material_name}_{self.tabs.tabText(self.tabs.currentIndex())}.png"
            file_path, _ = QFileDialog.getSaveFileName(self, t['save_image_btn'], default_name, "PNG Images (*.png);;PDF Documents (*.pdf);;All Files (*)")
            
            if file_path:
                try:
                    canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                    QMessageBox.information(self, t['msg_save_success_title'], t['msg_save_success_text'].format(file_path))
                except Exception as e:
                    QMessageBox.critical(self, t['msg_save_error_title'], t['msg_save_error_text'].format(str(e)))
        else:
             QMessageBox.warning(self, t['msg_no_plot_title'], t['msg_no_plot_text'])

    def update_temp_scan(self, scan_data):
        self.last_scan_data = scan_data
        ax = self.canvas_scan.axes
        ax.cla()
        t_dict = TRANSLATIONS[self.current_lang]
        
        t = np.asarray(scan_data['T'], dtype=float)
        noise = np.asarray(scan_data['noise'], dtype=float)
        current = np.asarray(scan_data['dark_current'], dtype=float)
        
        # Colors for scientific plotting
        color_noise = '#1f77b4'  # Muted blue
        color_current = '#d62728' # Muted red
        
        # Dual axis plot
        # Ensure data is valid for plotting
        if len(t) > 0 and len(noise) > 0:
            ax.plot(t, noise, marker='o', color=color_noise, label=f"{t_dict['stats_noise']} (keV)", linestyle='-')
        ax.set_xlabel(t_dict['plot_scan_xlabel'])
        ax.set_ylabel(t_dict['plot_scan_ylabel'], color=color_noise, weight='bold')
        ax.tick_params(axis='y', labelcolor=color_noise, width=1.5)
        ax.spines['left'].set_color(color_noise)
        ax.spines['left'].set_linewidth(1.5)
        
        ax2 = ax.twinx()
        if len(t) > 0 and len(current) > 0:
            ax2.plot(t, current, marker='s', color=color_current, label=f"{t_dict['stats_idark']} (nA)", linestyle='--')
        ax2.set_ylabel(t_dict['plot_scan_ylabel2'], color=color_current, weight='bold')
        ax2.tick_params(axis='y', labelcolor=color_current, width=1.5)
        ax2.spines['right'].set_color(color_current)
        ax2.spines['right'].set_linewidth(1.5)
        
        # Combined legend - MOVED to upper center with more margin
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # Place legend ABOVE the plot
        ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left', bbox_to_anchor=(0, 1.05), ncol=2, fontsize='small', borderaxespad=0)
        
        ax.set_title(t_dict['plot_scan_title'], y=1.25)
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # Adjust layout to make room for top legend
        self.canvas_scan.figure.subplots_adjust(top=0.7, bottom=0.15)
        self.canvas_scan.draw()

    def update_results(self, result, band_data, transport_data, polar_data, atten_data, flux_data, temp, energy):
        self.last_result_args = (result, band_data, transport_data, polar_data, atten_data, flux_data, temp, energy)
        t_dict = TRANSLATIONS[self.current_lang]
        # Update Stats
        stats = result["noise_stats"]
        eff = result["interaction_efficiency"]
        
        stats_text = (
            f"<b>{t_dict['stats_eff']}:</b> {eff*100:.1f}% | <b>{t_dict['stats_noise']}:</b> {stats['fwhm_noise_keV']:.2f} keV<br>"
            f"<b>{t_dict['stats_idark']}:</b> {stats['dark_current_nA']:.2f} nA | <b>{t_dict['stats_enc']}:</b> {stats['enc_electrons']:.0f} e-<br>"
            f"<b>{t_dict['stats_res']}:</b> {stats['rho']:.1e} Ω·cm"
        )
        self.stats_label.setText(stats_text)
        
        # Update Spectrum
        ax = self.canvas_spectrum.axes
        ax.cla()
        
        e_axis = np.asarray(result["energy_axis"], dtype=float)
        spectrum = np.asarray(result["spectrum"], dtype=float)
        
        # Use scientific colors and fill
        color_spec = '#2ca02c' # Muted green
        if len(e_axis) > 0 and len(spectrum) == len(e_axis):
            ax.fill_between(e_axis, spectrum, color=color_spec, alpha=0.3)
            ax.plot(e_axis, spectrum, color=color_spec, lw=2, label=f'{self.material_name}')
        
        # Incident energy marker
        ax.axvline(x=energy, color='#d62728', linestyle='--', alpha=0.8, lw=1.5, label=f'E_inc = {energy} keV')
        
        ax.set_title(f"{t_dict['plot_spectrum_title']} ({self.material_name})")
        ax.set_xlabel(t_dict['plot_spectrum_xlabel'])
        ax.set_ylabel(t_dict['plot_spectrum_ylabel'])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        self.canvas_spectrum.draw()
        
        # Update Band Diagram
        self.draw_band_diagram(band_data, temp)
        
        # Update Transport Diagram
        self.draw_transport_diagram(transport_data)
        
        # Update Polarization Diagram
        self.draw_polarization_diagram(polar_data)
        
        # Update Attenuation Diagram
        self.draw_attenuation_diagram(atten_data)
        
        # Update Flux Diagram
        self.draw_flux_diagram(flux_data)
        
    def draw_flux_diagram(self, data):
        ax = self.canvas_flux.axes
        ax.cla()
        t_dict = TRANSLATIONS[self.current_lang]
        
        flux = np.asarray(data['flux_mcps_mm2'], dtype=float)
        fwhm = np.asarray(data['fwhm_keV'], dtype=float)
        throughput = np.asarray(data['throughput_mcps_mm2'], dtype=float)
        
        color_res = '#d62728' # Red for degradation
        color_thru = '#1f77b4' # Blue for throughput
        
        # Plot FWHM
        if len(flux) > 0:
            p1, = ax.plot(flux, fwhm, marker='o', color=color_res, linestyle='-', label=t_dict['legend_fwhm'])
            
        ax.set_xlabel(t_dict['plot_flux_xlabel'])
        ax.set_ylabel(t_dict['plot_flux_ylabel'], color=color_res, weight='bold')
        ax.tick_params(axis='y', labelcolor=color_res)
        ax.spines['left'].set_color(color_res)
        ax.spines['left'].set_linewidth(1.5)
        
        # Plot Throughput on Twin Axis
        ax2 = ax.twinx()
        if len(flux) > 0:
            p2, = ax2.plot(flux, throughput, marker='s', color=color_thru, linestyle='--', label=t_dict['legend_throughput'])
            
            # Add ideal line
            ax2.plot(flux, flux, color='gray', linestyle=':', alpha=0.5, label='Ideal')
            
        ax2.set_ylabel(t_dict['plot_flux_ylabel2'], color=color_thru, weight='bold')
        ax2.tick_params(axis='y', labelcolor=color_thru)
        ax2.spines['right'].set_color(color_thru)
        ax2.spines['right'].set_linewidth(1.5)
        
        # Combined Legend - MOVED to upper center with more margin
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # Place legend ABOVE the plot to avoid all overlap issues
        ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2, fontsize='small', borderaxespad=0)
        
        # Adjust title position to be even higher
        ax.set_title(t_dict['plot_flux_title'], y=1.2)
        ax.grid(True, alpha=0.3)
        
        # Adjust layout to make room for top legend
        self.canvas_flux.figure.subplots_adjust(top=0.75, bottom=0.15)
        self.canvas_flux.draw()
        
    def draw_attenuation_diagram(self, data):
        ax = self.canvas_atten.axes
        ax.cla()
        t_dict = TRANSLATIONS[self.current_lang]
        
        E = np.asarray(data['energy_keV'], dtype=float)
        mu = np.asarray(data['attenuation'], dtype=float)
        k_edges = data['k_edges']
        
        color_atten = '#ff7f0e' # Muted orange
        if len(E) > 0 and len(mu) == len(E):
            ax.plot(E, mu, color=color_atten, lw=2.5)
            # Only set log scale if data is valid and positive
            if np.any(mu > 0):
                ax.set_yscale('log') # Log scale for attenuation usually looks best
        
        # Mark K-edges
        y_min, y_max = ax.get_ylim()
        for elem, edge_E in k_edges.items():
            if E.min() < edge_E < E.max():
                ax.axvline(x=edge_E, color='#1f77b4', linestyle=':', alpha=0.7)
                ax.text(edge_E, y_max*0.5, f" {elem} K-edge\n {edge_E:.1f} keV", 
                        rotation=90, verticalalignment='top', fontsize=8, color='#1f77b4')
        
        ax.set_title(t_dict['plot_atten_title'])
        ax.set_xlabel(t_dict['plot_atten_xlabel'])
        ax.set_ylabel(t_dict['plot_atten_ylabel'])
        ax.grid(True, which="both", ls="-", alpha=0.2)
        self.canvas_atten.draw()

    def draw_polarization_diagram(self, data):
        ax = self.canvas_pol.axes
        ax.cla()
        t_dict = TRANSLATIONS[self.current_lang]
        
        t = np.asarray(data['time_s'], dtype=float)
        ef = np.asarray(data['field_strength'], dtype=float)
        
        color_pol = '#9467bd' # Muted purple
        if len(t) > 0 and len(ef) == len(t):
            ax.plot(t, ef, color=color_pol, lw=2.5)
            ax.fill_between(t, ef, 0, color=color_pol, alpha=0.1)
        
        ax.set_title(t_dict['plot_pol_title'])
        ax.set_xlabel(t_dict['plot_pol_xlabel'])
        ax.set_ylabel(t_dict['plot_pol_ylabel'])
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        self.canvas_pol.draw()

    def draw_transport_diagram(self, data):
        ax = self.canvas_trans.axes
        ax.cla()
        t_dict = TRANSLATIONS[self.current_lang]
        
        z = np.asarray(data['z_mm'], dtype=float)
        
        # Scientific styling for lines
        # Ensure arrays are valid
        if len(z) > 0:
            if len(data['cce_e']) == len(z):
                ax.plot(z, data['cce_e'], color='#1f77b4', linestyle='--', lw=1.5, label=t_dict['legend_cce_e'])
            if len(data['cce_h']) == len(z):
                ax.plot(z, data['cce_h'], color='#d62728', linestyle='--', lw=1.5, label=t_dict['legend_cce_h'])
            if len(data['cce_total']) == len(z):
                ax.plot(z, data['cce_total'], color='#333333', linestyle='-', lw=2.5, label=t_dict['legend_cce_total'])
        
        ax.set_title(t_dict['plot_trans_title'])
        ax.set_xlabel(t_dict['plot_trans_xlabel'])
        ax.set_ylabel(t_dict['plot_trans_ylabel'])
        ax.set_ylim(0, 1.05)
        
        # Add Electric Field on secondary axis
        if 'E_field_abs' in data:
            ax2 = ax.twinx()
            E_field = np.asarray(data['E_field_abs'], dtype=float)
            # Normalize field for display or show V/cm?
            if len(z) > 0 and len(E_field) == len(z):
                ax2.plot(z, E_field, color='#2ca02c', linestyle=':', lw=2, label=t_dict['legend_efield'])
            ax2.set_ylabel(t_dict['plot_trans_ylabel2'], color='#2ca02c')
            ax2.tick_params(axis='y', labelcolor='#2ca02c')
            # Add field to legend
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize='small')
        else:
            ax.legend(loc='lower left', fontsize='small')
            
        ax.grid(True, alpha=0.3)
        
        # Add annotation for Anode/Cathode
        ax.text(0, 1.06, t_dict['cathode'], ha='left', va='bottom', fontsize=8, color='#555')
        ax.text(max(z), 1.06, t_dict['anode'], ha='right', va='bottom', fontsize=8, color='#555')
        
        self.canvas_trans.draw()

    def draw_band_diagram(self, band_data, temp):
        ax = self.canvas_band.axes
        ax.cla()
        t_dict = TRANSLATIONS[self.current_lang]
        
        Ev = band_data['Ev']
        Ec = band_data['Ec']
        Ef = band_data['Ef']
        Eg = band_data['Eg']
        
        ax.set_ylim(-0.5, Eg + 0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_title(f"{t_dict['plot_band_title']} @ {temp}K")
        
        # Bands with gradients (simulated by alpha blocks for simplicity in mpl)
        # Valence Band
        rect_vb = patches.Rectangle((0, -10), 1, 10 + Ev, linewidth=0, facecolor='#7f7f7f', alpha=0.3)
        ax.add_patch(rect_vb)
        ax.axhline(y=Ev, color='black', lw=2)
        ax.text(0.05, Ev - 0.1, t_dict['valence_band'], va='top', fontsize=9, fontweight='bold')
        
        # Conduction Band
        rect_cb = patches.Rectangle((0, Ec), 1, 10, linewidth=0, facecolor='#7f7f7f', alpha=0.3)
        ax.add_patch(rect_cb)
        ax.axhline(y=Ec, color='black', lw=2)
        ax.text(0.05, Ec + 0.1, t_dict['conduction_band'], va='bottom', fontsize=9, fontweight='bold')
        
        # Fermi Level
        ax.axhline(y=Ef, color='#d62728', linestyle='--', lw=2, label=t_dict['fermi_level'])
        ax.text(0.95, Ef + 0.05, f"Ef = {Ef:.3f} eV", color='#d62728', ha='right', va='bottom', fontsize=9, fontweight='bold')
        
        # Gap Arrow
        ax.annotate(f"", xy=(0.5, Ev), xytext=(0.5, Ec), arrowprops=dict(arrowstyle='<->', color='#1f77b4', lw=1.5))
        ax.text(0.52, (Ev+Ec)/2, f"Eg = {Eg:.2f} eV", color='#1f77b4', verticalalignment='center', fontweight='bold', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        ax.grid(False) # Clean look for schematic
        self.canvas_band.draw()

class ComparisonSummaryWidget(QWidget):
    """
    A widget to display a comparative summary table and radar chart for selected materials.
    Evaluates suitability for PCCT (Photon-Counting CT).
    """
    def __init__(self, current_lang='en'):
        super().__init__()
        self.current_lang = current_lang
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        
        # Left: Table
        self.table_group = QGroupBox()
        table_layout = QVBoxLayout(self.table_group)
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table_layout.addWidget(self.table)
        
        # Right: Radar Chart
        self.radar_group = QGroupBox()
        radar_layout = QVBoxLayout(self.radar_group)
        self.canvas_radar = MplCanvas(self, width=5, height=4, dpi=80)
        radar_layout.addWidget(self.canvas_radar)
        
        layout.addWidget(self.table_group, stretch=3)
        layout.addWidget(self.radar_group, stretch=2)
        
        self.update_language(self.current_lang)
        
    def update_language(self, lang):
        self.current_lang = lang
        t = TRANSLATIONS[lang]
        
        self.table_group.setTitle(t['summary_title'])
        self.radar_group.setTitle(t['summary_radar_title'])
        
        headers = [
            t['col_material'], t['col_density'], t['col_z_eff'], t['col_eg'],
            t['col_mu_e'], t['col_res'], t['col_eff'], t['col_flux'], t['col_peak'], t['col_score']
        ]
        self.table.setHorizontalHeaderLabels(headers)
        
        # Redraw radar if data exists
        if hasattr(self, 'last_data'):
            self.update_data(self.last_data)

    def update_data(self, results_list):
        """
        results_list: List of tuples (material_name, result_dict, material_props, sim_params)
        """
        self.last_data = results_list
        self.table.setRowCount(len(results_list))
        t = TRANSLATIONS[self.current_lang]
        
        radar_data = []
        
        for row, (name, res, props, params) in enumerate(results_list):
            # 1. Extract Metrics
            density = props.density
            # Z_eff approximation (weighted average)
            if 'xraylib' in sys.modules:
                z_eff = 0.0
                for el, frac in props.atomic_composition.items():
                    try:
                        z_eff += xraylib.SymbolToAtomicNumber(el) * float(frac)
                    except Exception:
                        pass
            else:
                z_eff = 0
            eg = props.Eg
            mu_e = props.mu_e
            
            # Simulation results
            eff = res['interaction_efficiency']
            fwhm = res['noise_stats']['fwhm_noise_keV']
            incident_energy = float(params.get('energy', 0.0))
            e_axis = np.asarray(res.get('energy_axis', []), dtype=float)
            spectrum = np.asarray(res.get('spectrum', []), dtype=float)
            peak_ratio = 0.0
            if incident_energy > 0 and len(e_axis) > 0 and len(spectrum) == len(e_axis) and np.any(spectrum > 0):
                peak_energy = float(e_axis[int(np.argmax(spectrum))])
                peak_ratio = max(0.0, min(1.0, peak_energy / incident_energy))
            
            # Extract Max Throughput from flux data
            # flux_data is stored in sim result? No, it's passed separately in update_data.
            # We need to change how update_data is called in run_simulation.
            # In run_simulation, we are passing a tuple: 
            # (mat_name, result, sim.material, {'energy': energy, ...})
            # We should include flux_data in the tuple.
            
            # Check if params tuple has flux_data (hacky check for now, better to assume it's there if we updated run_simulation)
            # Actually, let's just use a default if missing.
            max_flux = 0.0
            if len(params) > 4 and 'flux_data' in params: # If we stored it in params dict
                 # But we store params as dict.
                 pass
            
            # Wait, in run_simulation I updated summary_data append to:
            # summary_data.append((mat_name, result, sim.material, {'energy': energy, ...}))
            # I need to add flux_data to this dictionary or tuple.
            
            # Let's assume params has 'max_throughput' key which we will add in run_simulation
            max_throughput = params.get('max_throughput', 0.0)
            
            # 2. PCCT Scoring (0-100)
            # High Z/Density (Stopping Power) -> Efficiency
            score_eff = min(eff * 1.2, 1.0) * 100 # >80% eff is max score
            
            # Speed (Mobility) -> High Flux
            # Si (~1400) is standard, CdTe (~1000) good, TlBr (~30) bad
            score_speed = min(mu_e / 1500.0, 1.0) * 100
            
            # Resolution (Energy) -> Low FWHM
            # < 1 keV is excellent (100), > 5 keV is poor (0)
            score_res = max(0, (5.0 - fwhm) / 5.0) * 100
            
            # Stability (Polarization penalty)
            # Hardcoded knowledge base penalty
            penalty_pol = 0
            if name in ["TlBr", "HgI2"]:
                penalty_pol = 40 # Severe polarization issues
            elif name in ["CdTe", "CZT"]:
                penalty_pol = 10 # Minor polarization at high flux
            score_stability = 100 - penalty_pol
            
            # Flux Score (New)
            # > 100 Mcps/mm2 is target. 
            score_flux = min(max_throughput / 100.0, 1.0) * 100

            score_peak = (peak_ratio ** 4) * 100
            
            # Total weighted score
            total_score = (
                0.14 * score_eff +
                0.14 * score_speed +
                0.14 * score_res +
                0.14 * score_stability +
                0.14 * score_flux +
                0.30 * score_peak
            )
            
            # 3. Fill Table
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(f"{density:.2f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{z_eff:.1f}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{eg:.2f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{mu_e:.0f}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{fwhm:.2f}"))
            self.table.setItem(row, 6, QTableWidgetItem(f"{eff*100:.1f}%"))
            self.table.setItem(row, 7, QTableWidgetItem(f"{max_throughput:.1f}")) # Flux Col
            self.table.setItem(row, 8, QTableWidgetItem(f"{peak_ratio*100:.1f}%"))
            
            item_score = QTableWidgetItem(f"{total_score:.1f}")
            item_score.setFont(self.table.font())
            # Color code score
            if total_score > 80:
                item_score.setBackground(QColor("#d4edda")) # Green
            elif total_score > 60:
                item_score.setBackground(QColor("#fff3cd")) # Yellow
            else:
                item_score.setBackground(QColor("#f8d7da")) # Red
            self.table.setItem(row, 9, item_score)
            
            # 4. Prepare Radar Data
            radar_data.append({
                'name': name,
                'scores': [score_eff, score_speed, score_res, score_stability, score_flux, score_peak]
            })
            
        self.draw_radar(radar_data)
        
    def draw_radar(self, data):
        ax = self.canvas_radar.axes
        ax.clear()
        
        # Radar chart setup is tricky in standard mpl axes, 
        # usually needs 'polar=True' at creation.
        # But MplCanvas created a standard subplot(111).
        # We need to replace it.
        self.canvas_radar.fig.clear()
        
        # Adjust layout for radar chart
        self.canvas_radar.fig.subplots_adjust(top=0.85, bottom=0.1)
        
        ax = self.canvas_radar.fig.add_subplot(111, polar=True)
        self.canvas_radar.axes = ax # Update reference
        
        t = TRANSLATIONS[self.current_lang]
        categories = [t['radar_efficiency'], t['radar_speed'], t['radar_resolution'], t['radar_stability'], t['radar_flux'], t['radar_peak']]
        N = len(categories)
        
        # Angles for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1] # Close loop
        
        # Colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, item in enumerate(data):
            values = item['scores']
            values += values[:1] # Close loop
            
            c = colors[i % len(colors)]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=item['name'], color=c)
            ax.fill(angles, values, color=c, alpha=0.1)
            
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Y labels
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=7)
        plt.ylim(0, 100)
        
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        ax.set_title(t['summary_radar_title'], y=1.08)
        
        self.canvas_radar.draw()

class ClickableLabel(QLabel):
    def __init__(self, text, tooltip_text, parent=None):
        super().__init__(text, parent)
        self.tooltip_text = tooltip_text
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("QLabel { color: #2980b9; font-weight: bold; } QLabel:hover { color: #3498db; text-decoration: underline; }")
    
    def update_content(self, text, tooltip_text):
        self.setText(text)
        self.tooltip_text = tooltip_text

    def mousePressEvent(self, event):
        QMessageBox.information(self.window(), "Parameter Explanation", self.tooltip_text)

class RepositoryDialog(QDialog):
    """
    Dialog to browse and select materials from a remote repository.
    """
    def __init__(self, parent=None, manager=None, lang='en'):
        super().__init__(parent)
        self.manager = manager
        self.lang = lang
        self.repo_data = {}
        
        self.init_ui()
        
    def init_ui(self):
        t = TRANSLATIONS[self.lang]
        self.setWindowTitle(t['repo_title'])
        self.resize(500, 400)
        
        layout = QVBoxLayout(self)
        
        # 1. Manual Entry Section (Moved to top, removed URL section)
        manual_group = QGroupBox(t['repo_group_manual'])
        manual_layout = QGridLayout(manual_group)
        
        # Fields for manual entry
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("e.g. MyMaterial")
        
        self.edit_Eg = QDoubleSpinBox()
        self.edit_Eg.setRange(0.1, 10.0)
        self.edit_Eg.setDecimals(2)
        self.edit_Eg.setValue(1.5)
        
        self.edit_rho = QDoubleSpinBox()
        self.edit_rho.setRange(0.1, 20.0)
        self.edit_rho.setValue(5.0)
        
        self.edit_mu_e = QDoubleSpinBox()
        self.edit_mu_e.setRange(1, 1e5)
        self.edit_mu_e.setValue(1000)
        
        self.edit_mu_h = QDoubleSpinBox()
        self.edit_mu_h.setRange(1, 1e5)
        self.edit_mu_h.setValue(100)
        
        self.edit_tau_e = QLineEdit("1e-6") # Scientific notation easier in text
        self.edit_tau_h = QLineEdit("1e-6")
        self.edit_W = QDoubleSpinBox()
        self.edit_W.setValue(4.5)
        self.edit_comp = QLineEdit()
        self.edit_comp.setPlaceholderText(t['repo_comp_placeholder'])
        self._setup_composition_completer()
        
        # Add to grid
        manual_layout.addWidget(QLabel(t['repo_lbl_name']), 0, 0)
        manual_layout.addWidget(self.edit_name, 0, 1)
        manual_layout.addWidget(QLabel("Eg (eV):"), 0, 2)
        manual_layout.addWidget(self.edit_Eg, 0, 3)
        
        manual_layout.addWidget(QLabel(t['repo_lbl_density']), 1, 0)
        manual_layout.addWidget(self.edit_rho, 1, 1)
        manual_layout.addWidget(QLabel("W (eV):"), 1, 2)
        manual_layout.addWidget(self.edit_W, 1, 3)
        
        manual_layout.addWidget(QLabel(t['repo_lbl_mu_e']), 2, 0)
        manual_layout.addWidget(self.edit_mu_e, 2, 1)
        manual_layout.addWidget(QLabel(t['repo_lbl_mu_h']), 2, 2)
        manual_layout.addWidget(self.edit_mu_h, 2, 3)
        
        manual_layout.addWidget(QLabel(t['repo_lbl_tau_e']), 3, 0)
        manual_layout.addWidget(self.edit_tau_e, 3, 1)
        manual_layout.addWidget(QLabel(t['repo_lbl_tau_h']), 3, 2)
        manual_layout.addWidget(self.edit_tau_h, 3, 3)

        manual_layout.addWidget(QLabel(t['repo_lbl_comp']), 4, 0)
        manual_layout.addWidget(self.edit_comp, 4, 1, 1, 3)
        
        parse_comp_btn = QPushButton(t['repo_btn_parse_comp'])
        parse_comp_btn.clicked.connect(self.parse_compound)
        manual_layout.addWidget(parse_comp_btn, 5, 2)

        add_manual_btn = QPushButton(t['repo_btn_add'])
        add_manual_btn.clicked.connect(self.add_manual_material)
        manual_layout.addWidget(add_manual_btn, 5, 3)
        
        layout.addWidget(manual_group)
        
        # 2. List of Materials
        self.list_widget = QListWidget()
        layout.addWidget(QLabel(t['repo_lbl_import']))
        layout.addWidget(self.list_widget)
        
        # 3. Description Area
        self.desc_label = QLabel(t['select_material_hint'])
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("color: #555; font-style: italic; margin: 5px;")
        layout.addWidget(self.desc_label)
        
        self.list_widget.currentItemChanged.connect(self.show_preview)
        
        # 4. Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        btn_box.button(QDialogButtonBox.StandardButton.Ok).setText(t['repo_btn_import'])
        layout.addWidget(btn_box)

    def _setup_composition_completer(self):
        symbols = get_xraylib_element_symbols()
        completer = QCompleter(symbols, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.activated.connect(self._insert_composition_completion)
        self.edit_comp.textEdited.connect(self._update_composition_completion_prefix)
        self.edit_comp.setCompleter(completer)
        self._composition_completer = completer

    def _update_composition_completion_prefix(self, text):
        cursor = self.edit_comp.cursorPosition()
        start = cursor
        while start > 0 and text[start - 1].isalpha():
            start -= 1
        prefix = text[start:cursor]
        if not prefix:
            return
        self._composition_completer.setCompletionPrefix(prefix)
        self._composition_completer.complete(self.edit_comp.cursorRect())

    def _insert_composition_completion(self, completion):
        text = self.edit_comp.text()
        cursor = self.edit_comp.cursorPosition()

        start = cursor
        while start > 0 and text[start - 1].isalpha():
            start -= 1

        end = cursor
        while end < len(text) and text[end].isalpha():
            end += 1

        new_text = text[:start] + completion + text[end:]
        self.edit_comp.setText(new_text)
        self.edit_comp.setCursorPosition(start + len(completion))

    def parse_compound(self):
        t = TRANSLATIONS[self.lang]
        try:
            formula = composition_text_to_formula(self.edit_comp.text())
            parsed = xraylib.CompoundParser(formula)
            elements = parsed.get("Elements", ())
            mass_fracs = parsed.get("massFractions", ())
            n_atoms = parsed.get("nAtoms", ())
            lines = [t['repo_parse_formula'].format(formula)]
            for i, z in enumerate(elements):
                sym = xraylib.AtomicNumberToSymbol(int(z))
                mf = float(mass_fracs[i]) if i < len(mass_fracs) else 0.0
                na = float(n_atoms[i]) if i < len(n_atoms) else 0.0
                lines.append(f"{sym}: mass={mf:.6f}, atoms={na:g}")
            mm = parsed.get("molarMass", None)
            if mm is not None:
                lines.append(t['repo_parse_molar_mass'].format(float(mm)))
            QMessageBox.information(self, t['repo_parse_title'], "\n".join(lines))
        except Exception as e:
            QMessageBox.warning(self, t['msg_input_error_title'], t['repo_parse_failed'].format(str(e)))
        
    def add_manual_material(self):
        try:
            t = TRANSLATIONS[self.lang]
            from src.material import MaterialProperties
            
            name = self.edit_name.text().strip()
            if not name:
                QMessageBox.warning(self, t['msg_input_error_title'], t['msg_name_req'])
                return
                
            # Parse scientific notation manually if needed, but float() handles '1e-6'
            try:
                tau_e = float(self.edit_tau_e.text())
                tau_h = float(self.edit_tau_h.text())
            except ValueError:
                QMessageBox.warning(self, t['msg_input_error_title'], t['msg_invalid_num'])
                return

            try:
                atomic_composition = parse_atomic_composition(self.edit_comp.text())
            except ValueError as e:
                QMessageBox.warning(self, t['msg_input_error_title'], t['msg_comp_invalid_text'].format(str(e)))
                return
            
            # Construct property object
            props = MaterialProperties(
                name=name,
                Eg=self.edit_Eg.value(),
                density=self.edit_rho.value(),
                W_pair=self.edit_W.value(),
                mu_e=self.edit_mu_e.value(),
                mu_h=self.edit_mu_h.value(),
                tau_e=tau_e,
                tau_h=tau_h,
                atomic_composition=atomic_composition
            )
            
            # Add to repo_data temporarily (so it shows in list)
            self.repo_data[name] = props
            
            # Add to list widget and select it
            item = self.list_widget.addItem(name)
            # Find the item and select it
            items = self.list_widget.findItems(name, Qt.MatchFlag.MatchExactly)
            if items:
                items[0].setSelected(True)
                
            QMessageBox.information(self, t['msg_added_title'], t['msg_added_text'].format(name))
            
        except Exception as e:
            QMessageBox.warning(self, t['msg_input_error_title'], str(e))

    def fetch_catalog(self):
        # Deprecated functionality, kept for API compatibility or future re-enablement
        pass

    def show_preview(self, current, previous):
        if not current:
            return
        # Handle " (Installed)" suffix
        name = current.text().replace(" (Installed)", "")
        
        if name in self.repo_data:
            mat = self.repo_data[name]
            info = (
                f"Name: {mat.name}\n"
                f"Bandgap: {mat.Eg} eV\n"
                f"Density: {mat.density} g/cm3\n"
                f"Mobility (e/h): {mat.mu_e} / {mat.mu_h}\n"
                f"Composition: {mat.atomic_composition}"
            )
            self.desc_label.setText(info)

    def get_selected_materials(self):
        # In QListWidget with single selection mode (default), only one.
        # Let's enable multi-selection for the list
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        
        selected_items = self.list_widget.selectedItems()
        selected_mats = {}
        for item in selected_items:
            name = item.text().replace(" (Installed)", "")
            if name in self.repo_data:
                selected_mats[name] = self.repo_data[name]
        return selected_mats

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        if APP_ICON_PATH:
            self.setWindowIcon(QIcon(APP_ICON_PATH))
        self.current_lang = 'en' # Default language
        self.material_manager = MaterialManager()
        self.current_result_widgets = []
        
        self.init_ui()
        self.update_ui_text()

    def init_ui(self):
        self.resize(1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel: Controls ---
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(320)
        
        # Config Group
        self.config_group = QGroupBox()
        form_layout = QFormLayout()
        
        # Material Selection (List Widget for Multi-select)
        self.material_list = QListWidget()
        self.material_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.material_list.setFixedHeight(120)
        self.refresh_material_list()
        self.material_list.itemSelectionChanged.connect(self.check_selection_limit)
        self.material_list.currentItemChanged.connect(self.update_material_info_from_item)
        
        # Material Details
        self.material_details = QLabel()
        self.material_details.setStyleSheet("font-size: 10px; color: #333;")
        self.material_details.setWordWrap(True)
        self.material_details.setFixedHeight(100)
        
        self.energy_spin = QDoubleSpinBox()
        self.energy_spin.setRange(10.0, 500.0)
        self.energy_spin.setValue(200.0)
        self.energy_spin.setSuffix(" keV")
        
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(50.0, 500.0)
        self.temp_spin.setValue(300.0)
        self.temp_spin.setSuffix(" K")
        
        # Temperature Conversion Label
        self.temp_conv_label = QLabel()
        self.temp_conv_label.setStyleSheet("color: #555; margin-left: 5px;")
        self.update_temp_label(self.temp_spin.value())
        self.temp_spin.valueChanged.connect(self.update_temp_label)
        
        # Container for Temp Spin + Label
        temp_container = QWidget()
        temp_layout = QHBoxLayout(temp_container)
        temp_layout.setContentsMargins(0, 0, 0, 0)
        temp_layout.addWidget(self.temp_spin)
        temp_layout.addWidget(self.temp_conv_label)
        
        self.bias_spin = QDoubleSpinBox()
        self.bias_spin.setRange(10.0, 5000.0)
        self.bias_spin.setValue(500.0)
        self.bias_spin.setSingleStep(50.0)
        self.bias_spin.setSuffix(" V")
        
        self.thick_spin = QDoubleSpinBox()
        self.thick_spin.setRange(0.1, 50.0)
        self.thick_spin.setValue(2.0)
        self.thick_spin.setSingleStep(0.1)
        self.thick_spin.setSuffix(" mm")
        
        # Labels (Instance variables for translation updates)
        self.lbl_mat = ClickableLabel("", "")
        self.lbl_energy = ClickableLabel("", "")
        self.lbl_temp = ClickableLabel("", "")
        self.lbl_bias = ClickableLabel("", "")
        self.lbl_thick = ClickableLabel("", "")
        
        form_layout.addRow(self.lbl_mat, self.material_list)
        
        self.details_group = QGroupBox()
        details_layout = QVBoxLayout()
        details_layout.addWidget(self.material_details)
        self.details_group.setLayout(details_layout)
        form_layout.addRow(self.details_group)
        
        form_layout.addRow(self.lbl_energy, self.energy_spin)
        form_layout.addRow(self.lbl_temp, temp_container)
        form_layout.addRow(self.lbl_bias, self.bias_spin)
        form_layout.addRow(self.lbl_thick, self.thick_spin)
        
        self.config_group.setLayout(form_layout)
        control_layout.addWidget(self.config_group)
        
        self.update_btn = QPushButton()
        self.update_btn.clicked.connect(self.update_materials_online)
        control_layout.addWidget(self.update_btn)
        
        # Reset Button
        self.reset_btn = QPushButton()
        self.reset_btn.setStyleSheet("background-color: #e0e0e0; color: #333; padding: 5px;")
        self.reset_btn.clicked.connect(self.reset_materials)
        control_layout.addWidget(self.reset_btn)

        self.delete_btn = QPushButton()
        self.delete_btn.setStyleSheet("background-color: #ffcccc; color: #cc0000; padding: 5px;")
        self.delete_btn.clicked.connect(self.delete_selected_material)
        control_layout.addWidget(self.delete_btn)
        
        self.run_btn = QPushButton()
        self.run_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #4CAF50; color: white;")
        self.run_btn.clicked.connect(self.run_simulation)
        control_layout.addWidget(self.run_btn)
        
        control_layout.addStretch()
        
        # Language Button
        self.lang_btn = QPushButton()
        self.lang_btn.setStyleSheet("font-size: 10px; color: #555; padding: 5px;")
        self.lang_btn.clicked.connect(self.toggle_language)
        control_layout.addWidget(self.lang_btn)
        
        # --- Right Panel: Grid Layout for Results ---
        # We now use a QTabWidget to hold the Grid View and the Summary View
        self.right_tabs = QTabWidget()
        
        # Tab 1: Detailed Results (Grid)
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_container = QWidget()
        self.results_grid = QGridLayout(self.results_container)
        self.results_grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.results_scroll.setWidget(self.results_container)
        
        # Tab 2: Comparison Summary
        self.summary_widget = ComparisonSummaryWidget(self.current_lang)
        
        # Add Tabs (Titles will be updated by update_ui_text)
        self.right_tabs.addTab(self.results_scroll, "Detailed Results") 
        self.right_tabs.addTab(self.summary_widget, "Comparison Summary")
        
        # Add panels to main layout
        main_layout.addWidget(control_panel)
        
        # Right Side Container
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        right_layout.addWidget(self.right_tabs, stretch=1)
        
        # Footer
        self.footer_label = QLabel()
        self.footer_label.setWordWrap(True)
        self.footer_label.setOpenExternalLinks(True)
        self.footer_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.footer_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-top: 1px solid #ccc;")
        
        # Add footer to left control panel instead of right panel
        control_layout.addStretch() # Ensure it's at bottom
        control_layout.addWidget(self.footer_label)
        
        main_layout.addWidget(right_panel, stretch=2)
        
        # Select first item by default
        if self.material_list.count() > 0:
            self.material_list.setCurrentRow(0)

    def toggle_language(self):
        self.current_lang = 'zh' if self.current_lang == 'en' else 'en'
        self.update_ui_text()
        
    def get_best_cjk_font(self):
        """Find the best available CJK font on the system."""
        # Candidates for different OS - ORDER MATTERS
        # Put generic sans-serif last
        candidates = [
            'PingFang SC',      # macOS (Modern)
            'Heiti SC',         # macOS (Legacy)
            'STHeiti',          # macOS
            'Microsoft YaHei',  # Windows
            'SimHei',           # Windows
            'SimSun',           # Windows
            'WenQuanYi Micro Hei', # Linux
            'Droid Sans Fallback', # Linux
            'Noto Sans CJK SC',    # Generic
            'Arial Unicode MS',    # Fallback with CJK
        ]
        
        # Fallback English fonts that are commonly available
        fallbacks = ['Arial', 'Helvetica', 'Verdana', 'Times New Roman', 'DejaVu Sans']
        
        # Get list of all available fonts
        try:
            available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
        except:
            available_fonts = set()
        
        found = []
        for font in candidates:
            if font in available_fonts:
                found.append(font)
        
        # Also check fallbacks
        found_fallbacks = []
        for font in fallbacks:
            if font in available_fonts:
                found_fallbacks.append(font)
                
        # IMPORTANT: Ensure at least one known CJK font is first if available
        # If no CJK found, Matplotlib will try to fallback, but better to be explicit
        if not found:
             # Try to force some common names even if not in ttflist (sometimes aliases work)
             # But 'sans-serif' is the ultimate fallback
             print("Warning: No specific CJK font found in system font list.")
        
        # Return all found, plus generic sans-serif
        # Note: We put CJK first, then English fallbacks, then generic
        final_list = found + found_fallbacks + ['sans-serif']
        
        return final_list

    def update_ui_text(self):
        # Safe translation helper to prevent KeyErrors
        def tr(key, default=None):
            t = TRANSLATIONS[self.current_lang]
            return t.get(key, default if default is not None else key)

        t = TRANSLATIONS[self.current_lang]
        
        self.setWindowTitle(tr('window_title'))
        self.config_group.setTitle(tr('config_group'))
        
        self.lbl_mat.update_content(tr('materials_label'), tr('materials_tooltip'))
        self.lbl_energy.update_content(tr('energy_label'), tr('energy_tooltip'))
        self.lbl_temp.update_content(tr('temp_label'), tr('temp_tooltip'))
        self.lbl_bias.update_content(tr('bias_label'), tr('bias_tooltip'))
        self.lbl_thick.update_content(tr('thick_label'), tr('thick_tooltip'))
        
        self.details_group.setTitle(tr('selected_props_group'))
        if not self.material_list.selectedItems():
            self.material_details.setText(tr('select_material_hint'))
            
        self.update_btn.setText(tr('add_custom_btn'))
        self.reset_btn.setText(tr('reset_btn'))
        self.delete_btn.setText(tr('delete_btn'))
        self.run_btn.setText(tr('run_btn'))
        self.lang_btn.setText(tr('lang_btn'))
        self.footer_label.setText(tr('footer_text'))
        
        # Update Main Tab Titles
        self.right_tabs.setTabText(0, tr('tab_results', 'Detailed Results'))
        self.right_tabs.setTabText(1, tr('tab_summary'))
        
        # Update Matplotlib fonts
        if self.current_lang == 'zh':
            # Dynamically find best font
            cjk_fonts = self.get_best_cjk_font()
            
            # CRITICAL FIX: Directly update rcParams AND the font manager's default family
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['font.sans-serif'] = cjk_fonts
            plt.rcParams['axes.unicode_minus'] = False 
            
            print(f"DEBUG: Set font family to {cjk_fonts}")
        else:
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Verdana', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = True

        # Update existing result widgets
        for widget in self.current_result_widgets:
            widget.update_language(self.current_lang)
            
        # Update Summary Widget
        self.summary_widget.update_language(self.current_lang)

    def refresh_material_list(self):
        self.material_list.clear()
        names = self.material_manager.get_all_names()
        self.material_list.addItems(names)
        
    def check_selection_limit(self):
        selected = self.material_list.selectedItems()
        if len(selected) > 4:
            t = TRANSLATIONS[self.current_lang]
            QMessageBox.warning(self, t['msg_limit_title'], t['msg_limit_text'])
            selected[-1].setSelected(False)

    def update_material_info_from_item(self, current, previous):
        if not current:
            return
        self.update_material_info(current.text())

    def update_material_info(self, material_name):
        mat = self.material_manager.get_material(material_name)
        if mat:
            info = (
                f"<b>{mat.name}</b><br>"
                f"Eg: {mat.Eg} eV | W: {mat.W_pair} eV<br>"
                f"Density: {mat.density} g/cm³<br>"
                f"μτ(e): {mat.mu_e * mat.tau_e:.1e} | μτ(h): {mat.mu_h * mat.tau_h:.1e}"
            )
            self.material_details.setText(info)
        else:
            self.material_details.setText("No Data")

    def update_temp_label(self, k_val):
        c_val = k_val - 273.15
        f_val = c_val * 9/5 + 32
        self.temp_conv_label.setText(f"({c_val:.1f}°C / {f_val:.1f}°F)")

    def update_materials_online(self):
        # New implementation using Repository Browser
        dlg = RepositoryDialog(self, self.material_manager, lang=self.current_lang)
        # Enable multi-select for the list widget inside dialog
        dlg.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        
        if dlg.exec():
            # User clicked Download
            to_import = dlg.get_selected_materials()
            if to_import:
                count = self.material_manager.import_materials(to_import)
                self.refresh_material_list()
                t = TRANSLATIONS[self.current_lang]
                QMessageBox.information(self, t['msg_import_success_title'], t['msg_import_success_text'].format(count))
            else:
                t = TRANSLATIONS[self.current_lang]
                QMessageBox.information(self, t['msg_no_mat_title'], t['msg_no_mat_text'])

    def delete_selected_material(self):
        t = TRANSLATIONS[self.current_lang]
        selected_items = self.material_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, t['msg_no_sel_title'], t['msg_no_sel_text'])
            return
            
        confirm = QMessageBox.question(self, t['msg_confirm_del_title'], 
                                     t['msg_confirm_del_text'].format(len(selected_items)),
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if confirm == QMessageBox.StandardButton.Yes:
            deleted_count = 0
            for item in selected_items:
                name = item.text()
                # Prevent deleting some core defaults if we wanted, but for now allow all
                if self.material_manager.delete_material(name):
                    deleted_count += 1
            
            self.refresh_material_list()
            self.material_details.setText(t['select_material_hint'])
            QMessageBox.information(self, t['msg_del_success_title'], t['msg_del_success_text'].format(deleted_count))

    def reset_materials(self):
        t = TRANSLATIONS[self.current_lang]
        confirm = QMessageBox.question(self, t['msg_confirm_reset_title'], 
                                     t['msg_confirm_reset_text'],
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if confirm == QMessageBox.StandardButton.Yes:
            # We can re-create the data/materials.json with initial defaults
            # Or better, let MaterialManager handle this if it had a reset method
            # For now, let's just write the hardcoded defaults back to file
            
            # Initial default materials
            defaults = {
                "Si": {
                    "name": "Si",
                    "Eg": 1.12,
                    "mu_e": 1400.0,
                    "mu_h": 450.0,
                    "tau_e": 1.0e-3,
                    "tau_h": 1.0e-3,
                    "W_pair": 3.6,
                    "density": 2.33,
                    "atomic_composition": {"Si": 1.0}
                },
                "Ge": {
                    "name": "Ge",
                    "Eg": 0.66,
                    "mu_e": 3900.0,
                    "mu_h": 1900.0,
                    "tau_e": 1.0e-3,
                    "tau_h": 1.0e-3,
                    "W_pair": 2.96,
                    "density": 5.32,
                    "atomic_composition": {"Ge": 1.0}
                },
                "CdTe": {
                    "name": "CdTe",
                    "Eg": 1.44,
                    "mu_e": 1100.0,
                    "mu_h": 100.0,
                    "tau_e": 3.0e-6,
                    "tau_h": 1.0e-6,
                    "W_pair": 4.43,
                    "density": 5.85,
                    "atomic_composition": {"Cd": 0.5, "Te": 0.5}
                },
                "CZT": {
                    "name": "CZT",
                    "Eg": 1.57,
                    "mu_e": 1000.0,
                    "mu_h": 80.0,
                    "tau_e": 5.0e-6,
                    "tau_h": 0.5e-6,
                    "W_pair": 4.6,
                    "density": 5.78,
                    "atomic_composition": {"Cd": 0.45, "Zn": 0.05, "Te": 0.5}
                },
                "TlBr": {
                    "name": "TlBr",
                    "Eg": 2.68,
                    "mu_e": 30.0,
                    "mu_h": 4.0,
                    "tau_e": 1.0e-5,
                    "tau_h": 1.0e-6,
                    "W_pair": 6.5,
                    "density": 7.56,
                    "atomic_composition": {"Tl": 0.5, "Br": 0.5}
                },
                "HgI2": {
                    "name": "HgI2",
                    "Eg": 2.13,
                    "mu_e": 100.0,
                    "mu_h": 4.0,
                    "tau_e": 1.0e-6,
                    "tau_h": 1.0e-5,
                    "W_pair": 4.2,
                    "density": 6.4,
                    "atomic_composition": {"Hg": 0.33, "I": 0.67}
                },
                "GaAs": {
                    "name": "GaAs",
                    "Eg": 1.42,
                    "mu_e": 8500.0,
                    "mu_h": 400.0,
                    "tau_e": 1.0e-5,
                    "tau_h": 1.0e-6,
                    "W_pair": 4.2,
                    "density": 5.32,
                    "atomic_composition": {"Ga": 0.5, "As": 0.5}
                },
                "Perovskite": {
                    "name": "Perovskite",
                    "Eg": 1.55,
                    "mu_e": 60.0,
                    "mu_h": 60.0,
                    "tau_e": 1.0e-6,
                    "tau_h": 1.0e-6,
                    "W_pair": 5.0,
                    "density": 4.16,
                    "atomic_composition": {"C": 1, "H": 6, "N": 1, "Pb": 1, "I": 3}
                },
                "SiC": {
                    "name": "SiC",
                    "Eg": 1.5,
                    "mu_e": 1000.0,
                    "mu_h": 100.0,
                    "tau_e": 1.0e-6,
                    "tau_h": 1.0e-6,
                    "W_pair": 4.5,
                    "density": 5.0,
                    "atomic_composition": {"Si": 0.5, "C": 0.5}
                },
                "4H-SiC": {
                    "name": "4H-SiC",
                    "Eg": 3.26,
                    "mu_e": 950.0,
                    "mu_h": 120.0,
                    "tau_e": 1.0e-6,
                    "tau_h": 1.0e-6,
                    "W_pair": 7.8,
                    "density": 3.21,
                    "atomic_composition": {"Si": 0.5, "C": 0.5}
                }
            }
            
            try:
                # Need to use the manager to overwrite
                # Convert dicts to MaterialProperties
                from src.material import MaterialProperties
                new_mats = {}
                for name, props in defaults.items():
                    new_mats[name] = MaterialProperties(**props)
                
                # Directly replace manager's dict and save
                self.material_manager.materials = new_mats
                self.material_manager.save_materials()
                
                self.refresh_material_list()
                QMessageBox.information(self, t['msg_reset_success_title'], t['msg_reset_success_text'])
            except Exception as e:
                QMessageBox.critical(self, t['msg_sim_error_title'], t['msg_sim_error_text'].format(str(e)))

    def run_simulation(self):
        t = TRANSLATIONS[self.current_lang]
        selected_items = self.material_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, t['msg_no_sel_title'], t['msg_no_sel_text'])
            return
            
        # Clear existing results
        while self.results_grid.count():
            item = self.results_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
                
        # Parameters
        energy = self.energy_spin.value()
        start_temp = self.temp_spin.value() # Use this as the "base" temp or scan range
        bias = self.bias_spin.value()
        thick = self.thick_spin.value()
        
        row = 0
        col = 0
        
        # Store widgets to manage synchronization
        self.current_result_widgets = []
        
        # Collect data for summary
        summary_data = []
        
        for item in selected_items:
            mat_name = item.text()
            
            try:
                # 1. Standard Point Simulation (at current T)
                # Use Monte Carlo if checked? Or by default?
                # Let's use MC by default for better realism now
                sim = DetectorSimulator(mat_name, thick, bias, start_temp)
                
                # Use MC simulation
                # Note: MC is slower, so we might want to make it optional or async in future
                # For now, 10000 photons is fast enough
                result = sim.run_monte_carlo_simulation(energy, n_photons=5000)
                
                band_data = sim.get_band_structure()
                transport_data = sim.get_transport_profile()
                polar_data = sim.get_polarization_profile()
                atten_data = sim.get_attenuation_profile(energy_range=np.linspace(10, 150, 500))
                
                # 1.5 Flux Performance
                flux_data = sim.calculate_high_flux_performance(max_flux_mcps_mm2=200)
                
                # 2. Temperature Scan Simulation (303K - 313K, step 1K)
                t_range = np.arange(303, 314, 1) 
                scan_results = {
                    "T": t_range,
                    "noise": [],
                    "dark_current": []
                }
                
                for t in t_range:
                    s = DetectorSimulator(mat_name, thick, bias, float(t))
                    res = s.calculate_noise_properties()
                    scan_results["noise"].append(res['fwhm_noise_keV'])
                    scan_results["dark_current"].append(res['dark_current_nA'])
                
                # Create Result Widget
                res_widget = MaterialResultWidget(mat_name, current_lang=self.current_lang)
                res_widget.update_results(result, band_data, transport_data, polar_data, atten_data, flux_data, start_temp, energy)
                res_widget.update_temp_scan(scan_results)  
                
                # Connect synchronization signal
                res_widget.tab_changed.connect(self.sync_tabs)
                self.current_result_widgets.append(res_widget)
                
                # Add to Grid
                self.results_grid.addWidget(res_widget, row, col)
                
                # Collect for summary
                # Extract max throughput from flux_data
                max_throughput = max(flux_data['throughput_mcps_mm2']) if len(flux_data['throughput_mcps_mm2']) > 0 else 0.0
                summary_data.append((mat_name, result, sim.material, {'energy': energy, 'bias': bias, 'thick': thick, 'temp': start_temp, 'max_throughput': max_throughput}))
                
                col += 1
                if col > 1:
                    col = 0
                    row += 1
                    
            except Exception as e:
                import traceback
                tb_str = traceback.format_exc()
                print(f"Error simulating {mat_name}: {e}\n{tb_str}")

        # Update Summary Widget
        if summary_data:
            self.summary_widget.update_data(summary_data)

    def sync_tabs(self, index):
        """Synchronize all result widgets to show the same tab index."""
        for widget in self.current_result_widgets:
            widget.set_tab_index(index)

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        if APP_ICON_PATH:
            app.setWindowIcon(QIcon(APP_ICON_PATH))
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        # Global exception handler for packaged app debugging
        import traceback
        error_msg = f"Critical Error:\n{str(e)}\n\n{traceback.format_exc()}"
        # If QApplication is not created yet, we can't use QMessageBox easily, but here it likely is
        try:
            QMessageBox.critical(None, "Fatal Error", error_msg)
        except:
            print(error_msg) # Fallback to console if GUI fails completely
        sys.exit(1)
