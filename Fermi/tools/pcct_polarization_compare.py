import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with contextlib.redirect_stdout(io.StringIO()):
    from src.simulator import DetectorSimulator


def _interp(x, xs, ys):
    return float(np.interp(float(x), np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)))


def main():
    materials = ["CdTe", "CZT", "Si", "GaAs"]

    energy_keV = 200.0
    temp_k = 300.0
    bias_v = 500.0
    thickness_mm = 2.0

    flux_points = [1, 20, 50, 100, 150, 200]

    print("# PCCT 极化效应对比（按程序高通量模型）")
    print("")
    print("## 设置（与 GUI 默认一致）")
    print(f"- 能量: {energy_keV} keV")
    print(f"- 温度: {temp_k} K")
    print(f"- 偏置: {bias_v} V")
    print(f"- 厚度: {thickness_mm} mm")
    print("")
    print("## 指标说明")
    print("- 这里的“极化效应差异”对应高通量模块中红线 FWHM 的随通量退化。")
    print("- FWHM_total(flux) = sqrt(FWHM_CCE(flux)^2 + FWHM_elec^2)，其中 FWHM_CCE 由极化畸变电场导致。")
    print("- ΔFWHM 以 1 Mcps/mm² 作为近似低通量基线：ΔFWHM(flux)=FWHM_total(flux)-FWHM_total(1)。")
    print("")

    header = ["材料"] + [f"{fp} Mcps/mm² FWHM/Δ(keV)" for fp in flux_points]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] + ["---:" for _ in flux_points]) + "|")

    for m in materials:
        sim = DetectorSimulator(m, thickness_mm, bias_v, temp_k)
        data = sim.calculate_high_flux_performance(max_flux_mcps_mm2=200)
        flux = data["flux_mcps_mm2"]
        fwhm = data["fwhm_keV"]
        f_base = _interp(1, flux, fwhm)

        row = [m]
        for fp in flux_points:
            f = _interp(fp, flux, fwhm)
            row.append(f"{f:.3f} / {f - f_base:+.3f}")
        print("| " + " | ".join(row) + " |")

    print("")
    print("## 重要结论（对应代码口径）")
    print("- CdTe/CZT 被设为“中等极化”（trapping_factor=5），Si 被设为“低极化”（0.1），GaAs 未被单独归类，走默认 1.0。")
    print("- 因此按程序设定，CdTe/CZT 的 ΔFWHM 退化应明显高于 GaAs，Si 最小。")


if __name__ == "__main__":
    main()

