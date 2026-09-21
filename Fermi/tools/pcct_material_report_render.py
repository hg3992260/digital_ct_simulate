import json
import os


def _fmt(x, digits=3):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _fmt_sci(x, digits=3):
    try:
        v = float(x)
        if v == 0:
            return "0"
        if abs(v) < 1e-2 or abs(v) >= 1e4:
            return f"{v:.{digits}e}"
        return f"{v:.{digits}f}"
    except Exception:
        return str(x)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path = os.path.join(root, "tools", "pcct_material_report_output.json")
    out_path = os.path.join(root, "tools", "pcct_material_report_output.md")

    with open(in_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    s = data["settings"]
    materials = data["materials"]
    results = data["results_by_pixel_mm"]

    lines = []
    lines.append("# PCCT 半导体材料对比（按程序模型输出）")
    lines.append("")
    lines.append("## 仿真设置")
    lines.append("")
    lines.append(f"- 入射能量: {s['energy_keV']} keV")
    lines.append(f"- 温度: {s['temp_k']} K")
    lines.append(f"- 偏置: {s['bias_v']} V")
    lines.append(f"- 厚度: {s['thickness_mm']} mm")
    lines.append(f"- 像素边长: {', '.join(str(p) for p in s['pixel_pitches_mm'])} mm（正方形像素）")
    lines.append(f"- 衰减模型: {'xraylib' if s['has_xraylib'] else 'MockXrayLib'}")
    lines.append("")

    for p in s["pixel_pitches_mm"]:
        key = str(p)
        rows = results[key]

        lines.append(f"## 像素 {p} mm")
        lines.append("")
        lines.append("| 材料 | Eg(eV) | ρ(g/cm³) | μe | eff% | peak% | FWHM@200keV(keV) | I_dark(nA) | ENC(e-) | τ_dead(ns) | max throughput (Mcps/mm²) | PCCT score |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        for r in rows:
            m = materials[r["material"]]
            lines.append(
                "| "
                + " | ".join(
                    [
                        r["material"],
                        _fmt(m["Eg"], 2),
                        _fmt(m["density"], 2),
                        _fmt(m["mu_e"], 0),
                        _fmt(r["eff"] * 100.0, 1),
                        _fmt(r["peak_ratio"] * 100.0, 1),
                        _fmt(r["fwhm_noise_keV"], 3),
                        _fmt_sci(r["dark_current_nA"], 3),
                        _fmt(r["enc_electrons"], 3),
                        _fmt(r["tau_dead_ns"], 1),
                        _fmt(r["max_throughput_mcps_mm2"], 3),
                        _fmt(r["pcct_score"], 2),
                    ]
                )
                + " |"
            )
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(out_path)


if __name__ == "__main__":
    main()
