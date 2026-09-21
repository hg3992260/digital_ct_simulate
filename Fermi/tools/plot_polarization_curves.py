import contextlib
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with contextlib.redirect_stdout(io.StringIO()):
    from src.simulator import DetectorSimulator


def main():
    materials = ["CdTe", "CZT", "Si", "GaAs"]

    thickness_mm = 2.0
    bias_v = 500.0
    temp_k = 300.0

    fig = plt.figure(figsize=(8.2, 4.8), dpi=140)
    ax = fig.add_subplot(111)

    for name in materials:
        sim = DetectorSimulator(name, thickness_mm, bias_v, temp_k)
        pol = sim.get_polarization_profile()
        t = np.asarray(pol["time_s"], dtype=float)
        ef = np.asarray(pol["field_strength"], dtype=float)
        ax.plot(t, ef, linewidth=2.2, label=name)

    ax.set_title("Polarization Display Curves (Field Strength vs Time)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative Field Strength")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polarization_curves_CdTe_CZT_Si_GaAs.png")
    fig.tight_layout()
    fig.savefig(out_path)
    print(out_path)


if __name__ == "__main__":
    main()

