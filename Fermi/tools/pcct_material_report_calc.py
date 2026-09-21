import math
import os
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.material import MaterialManager
from src.physics import FermiPhysics, k_B, q
import contextlib
import io

with contextlib.redirect_stdout(io.StringIO()):
    from src.simulator import HAS_XRAYLIB, MockXrayLib, RealXrayLib


def _attenuation_mu_cm_inv(material, energy_keV: float) -> float:
    if HAS_XRAYLIB:
        return float(RealXrayLib.get_attenuation(material, [energy_keV])[0])
    return float(MockXrayLib.get_attenuation(material, [energy_keV])[0])


def noise_for_pixel(material, *, thickness_mm: float, bias_v: float, temp_k: float, pixel_pitch_mm: float, energy_keV: float):
    d_cm = thickness_mm * 0.1
    e_field_v_per_cm = bias_v / d_cm

    ni = float(FermiPhysics.calculate_intrinsic_carriers(material.Eg, temp_k))
    rho = float(FermiPhysics.calculate_resistivity(ni, material.mu_e, material.mu_h))

    pixel_area_cm2 = (pixel_pitch_mm**2) * 0.01
    current_a = (e_field_v_per_cm / rho) * pixel_area_cm2

    tau_int = 1e-6
    enc_coulomb = math.sqrt(max(0.0, 2.0 * q * current_a * tau_int))
    enc_electrons = enc_coulomb / q if q > 0 else float("inf")

    fwhm_elec_ev = 2.355 * enc_electrons * material.W_pair
    fwhm_elec_keV = fwhm_elec_ev / 1000.0

    F = 0.11
    W_keV = material.W_pair / 1000.0
    sigma_fano_peak = math.sqrt(max(0.0, F * energy_keV * W_keV))
    sigma_elec = fwhm_elec_keV / 2.355
    sigma_total_peak = math.sqrt(sigma_elec * sigma_elec + sigma_fano_peak * sigma_fano_peak)
    fwhm_noise_keV = sigma_total_peak * 2.355

    return {
        "ni_cm3": ni,
        "rho_ohm_cm": rho,
        "pixel_area_cm2": pixel_area_cm2,
        "dark_current_nA": current_a * 1e9,
        "enc_electrons": enc_electrons,
        "fwhm_elec_keV": fwhm_elec_keV,
        "fwhm_noise_keV": fwhm_noise_keV,
    }


def simulate_spectrum_peak(material, *, thickness_mm: float, bias_v: float, temp_k: float, energy_keV: float, fwhm_elec_keV: float, n_bins: int = 200):
    d_cm = thickness_mm * 0.1
    e_field_v_per_cm = bias_v / d_cm

    z_steps = 1000
    z = np.linspace(0.0, d_cm, z_steps)
    dz = d_cm / z_steps

    mu = _attenuation_mu_cm_inv(material, energy_keV)

    prob_density = mu * np.exp(-mu * z)
    interaction_efficiency = 1.0 - math.exp(-mu * d_cm)

    if float(np.sum(prob_density)) > 0:
        prob_density = prob_density / (float(np.sum(prob_density)) * dz) * interaction_efficiency

    lambda_e = material.mu_tau_e * e_field_v_per_cm
    lambda_h = material.mu_tau_h * e_field_v_per_cm
    cce = FermiPhysics.hecht_equation(z, d_cm, lambda_e, lambda_h)

    measured_energies = energy_keV * cce
    hist, bin_edges = np.histogram(
        measured_energies,
        bins=n_bins,
        range=(0.0, energy_keV * 1.1),
        weights=prob_density * dz,
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    F = 0.11
    W_keV = material.W_pair / 1000.0
    sigma_elec = fwhm_elec_keV / 2.355
    bin_width = float(bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 1.0

    final_spectrum = np.zeros_like(hist, dtype=float)
    for i, count in enumerate(hist):
        if count <= 0:
            continue
        e_val = float(bin_centers[i])
        sigma_fano = math.sqrt(max(0.0, F * e_val * W_keV)) if e_val > 0 else 0.0
        sigma_total = math.sqrt(sigma_elec * sigma_elec + sigma_fano * sigma_fano)
        if sigma_total > 1e-6:
            gaussian = (1.0 / (sigma_total * math.sqrt(2.0 * math.pi))) * np.exp(
                -0.5 * ((bin_centers - e_val) / sigma_total) ** 2
            )
            final_spectrum += float(count) * gaussian * bin_width
        else:
            final_spectrum[i] += float(count)

    if float(np.max(final_spectrum)) <= 0:
        peak_ratio = 0.0
    else:
        peak_energy = float(bin_centers[int(np.argmax(final_spectrum))])
        peak_ratio = max(0.0, min(1.0, peak_energy / energy_keV))

    return {
        "interaction_efficiency": interaction_efficiency,
        "peak_ratio": peak_ratio,
    }


def throughput_for_pixel(material, *, thickness_mm: float, bias_v: float, pixel_pitch_mm: float, max_flux_mcps_mm2: float = 200.0, n_points: int = 20):
    d_cm = thickness_mm * 0.1
    e_avg = bias_v / d_cm

    mu_slow = min(float(material.mu_e), float(material.mu_h))
    if e_avg > 0 and mu_slow > 0:
        t_coll = d_cm / (mu_slow * e_avg)
    else:
        t_coll = 1e-6

    tau_asic = 20e-9
    tau_phys = 0.5 * t_coll
    tau_dead = max(tau_asic, tau_phys)

    pixel_area_cm2 = (pixel_pitch_mm**2) * 0.01

    flux_values = np.linspace(1.0, float(max_flux_mcps_mm2), int(n_points))
    a = 1e8 * pixel_area_cm2 * tau_dead
    throughput = flux_values * np.exp(-flux_values * a)

    idx = int(np.argmax(throughput))
    return {
        "tau_dead_s": tau_dead,
        "pixel_area_cm2": pixel_area_cm2,
        "flux_values_mcps_mm2": flux_values,
        "throughput_values_mcps_mm2": throughput,
        "max_throughput_mcps_mm2": float(throughput[idx]),
        "argmax_flux_mcps_mm2": float(flux_values[idx]),
    }


def pcct_score(material_name: str, *, eff: float, mu_e: float, fwhm_noise_keV: float, max_throughput: float, peak_ratio: float):
    score_eff = min(eff * 1.2, 1.0) * 100.0
    score_speed = min(mu_e / 1500.0, 1.0) * 100.0
    score_res = max(0.0, (5.0 - fwhm_noise_keV) / 5.0) * 100.0

    penalty_pol = 0.0
    if material_name in ["TlBr", "HgI2"]:
        penalty_pol = 40.0
    elif material_name in ["CdTe", "CZT"]:
        penalty_pol = 10.0
    score_stability = 100.0 - penalty_pol

    score_flux = min(max_throughput / 100.0, 1.0) * 100.0
    score_peak = (peak_ratio**4) * 100.0

    total = (
        0.14 * score_eff
        + 0.14 * score_speed
        + 0.14 * score_res
        + 0.14 * score_stability
        + 0.14 * score_flux
        + 0.30 * score_peak
    )
    return {
        "score_eff": score_eff,
        "score_speed": score_speed,
        "score_res": score_res,
        "score_stability": score_stability,
        "score_flux": score_flux,
        "score_peak": score_peak,
        "score_total": total,
    }


def main():
    energy_keV = 200.0
    temp_k = 300.0
    bias_v = 500.0
    thickness_mm = 2.0
    pixel_pitches_mm = [1.0, 0.8, 0.6, 0.4, 0.2]

    manager = MaterialManager()
    material_names = sorted(manager.get_all_names())

    report = {
        "settings": {
            "energy_keV": energy_keV,
            "temp_k": temp_k,
            "bias_v": bias_v,
            "thickness_mm": thickness_mm,
            "pixel_pitches_mm": pixel_pitches_mm,
            "has_xraylib": bool(HAS_XRAYLIB),
            "k_B_eV_per_K": k_B,
            "q_C": q,
        },
        "materials": {name: asdict(manager.get_material(name)) for name in material_names},
        "results_by_pixel_mm": {},
    }

    for p in pixel_pitches_mm:
        rows = []
        for name in material_names:
            mat = manager.get_material(name)
            noise = noise_for_pixel(
                mat,
                thickness_mm=thickness_mm,
                bias_v=bias_v,
                temp_k=temp_k,
                pixel_pitch_mm=p,
                energy_keV=energy_keV,
            )
            spec = simulate_spectrum_peak(
                mat,
                thickness_mm=thickness_mm,
                bias_v=bias_v,
                temp_k=temp_k,
                energy_keV=energy_keV,
                fwhm_elec_keV=noise["fwhm_elec_keV"],
            )
            flux = throughput_for_pixel(
                mat,
                thickness_mm=thickness_mm,
                bias_v=bias_v,
                pixel_pitch_mm=p,
            )
            score = pcct_score(
                name,
                eff=float(spec["interaction_efficiency"]),
                mu_e=float(mat.mu_e),
                fwhm_noise_keV=float(noise["fwhm_noise_keV"]),
                max_throughput=float(flux["max_throughput_mcps_mm2"]),
                peak_ratio=float(spec["peak_ratio"]),
            )
            rows.append(
                {
                    "material": name,
                    "eff": float(spec["interaction_efficiency"]),
                    "peak_ratio": float(spec["peak_ratio"]),
                    "dark_current_nA": float(noise["dark_current_nA"]),
                    "enc_electrons": float(noise["enc_electrons"]),
                    "fwhm_noise_keV": float(noise["fwhm_noise_keV"]),
                    "max_throughput_mcps_mm2": float(flux["max_throughput_mcps_mm2"]),
                    "argmax_flux_mcps_mm2": float(flux["argmax_flux_mcps_mm2"]),
                    "tau_dead_ns": float(flux["tau_dead_s"]) * 1e9,
                    "pcct_score": float(score["score_total"]),
                    "score_breakdown": score,
                }
            )

        rows.sort(key=lambda r: (-r["pcct_score"], -r["eff"], r["fwhm_noise_keV"]))
        report["results_by_pixel_mm"][str(p)] = rows

    import json

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
