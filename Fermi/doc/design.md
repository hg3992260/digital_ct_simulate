# Semiconductor Detector Simulation Design Document

## 1. Overview
This project simulates the performance of semiconductor X-ray detectors (specifically CZT and TlBr) under CT workload conditions (40-190 keV). It integrates microscopic Fermi band calculations with macroscopic X-ray response simulation.

## 2. Architecture
The system follows a modular architecture:

*   **`src/`**: Source code
    *   `material.py`: Material property definitions (CZT, TlBr).
    *   `physics.py`: Core physics calculations (Fermi-Dirac, Hecht equation, Absorption).
    *   `simulator.py`: Integration of modules to simulate detector response.
    *   `main.py`: CLI entry point.
*   **`data/`**: Output directory for simulation results (CSV/JSON).
*   **`kb/`**: Reference knowledge base.

## 3. Data Flow
1.  **Input**:
    *   Material: CZT or TlBr
    *   Temperature ($T$)
    *   Doping/Defect levels ($N_d, N_a, E_t$)
    *   X-ray Energy Range (40-190 keV)
    *   Bias Voltage / Electric Field ($E$)

2.  **Fermi Band Module**:
    *   Calculate intrinsic carrier concentration ($n_i$).
    *   Calculate Fermi level ($E_F$) position.
    *   Calculate resistivity ($\rho$) -> Leakage Current -> Noise.

3.  **X-ray Response Module**:
    *   Calculate Linear Attenuation Coefficient ($\mu$) for given energy.
    *   Calculate Photon Absorption Efficiency.
    *   (Simplified) K-edge effects.

4.  **Coupling & Output**:
    *   Calculate Charge Collection Efficiency (CCE) using Hecht equation.
    *   Calculate Signal (generated charge).
    *   Calculate SNR (Signal / Noise).
    *   Output Spectral Response.

## 4. Key Formulas

### 4.1 Fermi Band & Noise
*   **Intrinsic Carrier Concentration**:
    $$ n_i = \sqrt{N_c N_v} \exp\left(-\frac{E_g}{2kT}\right) $$
*   **Resistivity (simplified)**:
    $$ \rho = \frac{1}{q (\mu_n n + \mu_p p)} $$
*   **Dark Current Noise (Shot Noise)**:
    $$ I_{dark} \approx \frac{V \cdot Area}{\rho \cdot d} $$
    $$ \sigma_{noise} = \sqrt{2 q I_{dark} \tau_{int}} $$ (assuming integration time $\tau_{int}$)

### 4.2 Charge Transport (Hecht Equation)
$$ CCE = \frac{\lambda_e}{d} \left(1 - \exp\left(-\frac{x}{\lambda_e}\right)\right) + \frac{\lambda_h}{d} \left(1 - \exp\left(-\frac{d-x}{\lambda_h}\right)\right) $$
Where $\lambda = \mu \tau E$ (Drift length).

### 4.3 Signal Generation
$$ Q_{gen} = \frac{E_{absorbed}}{W_{pair}} \cdot q $$

## 5. Implementation Details
*   **Language**: Python 3
*   **Libraries**: `numpy`, `scipy`
*   **X-ray Data**: Since `xraylib` might be hard to install in this environment, we will use simplified lookup tables or analytical approximations for attenuation coefficients of Cd, Zn, Te, Tl, Br.
