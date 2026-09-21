import numpy as np

# Physical Constants
k_B = 8.617e-5  # Boltzmann constant (eV/K)
q = 1.602e-19   # Elementary charge (C)
m0 = 9.109e-31  # Electron mass (kg)
epsilon_0 = 8.854e-14 # Vacuum permittivity (F/cm)

class FermiPhysics:
    @staticmethod
    def calculate_intrinsic_carriers(Eg, T, Nc_300=2.2e17, Nv_300=1.8e18):
        """
        Calculate intrinsic carrier concentration.
        Nc_300, Nv_300 are effective density of states at 300K (approximate values for typical semiconductors).
        """
        # Temperature dependence of DOS (T/300)^1.5
        Nc = Nc_300 * (T / 300.0)**1.5
        Nv = Nv_300 * (T / 300.0)**1.5
        
        ni = np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * k_B * T))
        return ni

    @staticmethod
    def calculate_fermi_level(Eg, T, Nc_300=2.2e17, Nv_300=1.8e18):
        """
        Calculate intrinsic Fermi level relative to Valence Band (Ev=0).
        Ei = (Ec + Ev)/2 + (kT/2) * ln(Nv/Nc)
        """
        # Temperature dependence cancels out in ratio Nv/Nc if mass ratio is constant,
        # but let's be explicit
        Nc = Nc_300 * (T / 300.0)**1.5
        Nv = Nv_300 * (T / 300.0)**1.5
        
        # Ev = 0, Ec = Eg
        Ei = (Eg / 2.0) + (k_B * T / 2.0) * np.log(Nv / Nc)
        return Ei

    @staticmethod
    def calculate_resistivity(ni, mu_e, mu_h):
        """
        Calculate resistivity (Ohm-cm) assuming intrinsic (or near-intrinsic) material.
        rho = 1 / (q * (n*mu_e + p*mu_h))
        For intrinsic: n = p = ni
        """
        sigma = q * ni * (mu_e + mu_h)
        return 1.0 / sigma if sigma > 0 else float('inf')

    @staticmethod
    def hecht_equation(z, d, lambda_e, lambda_h):
        """
        Calculate Charge Collection Efficiency (CCE) using Hecht equation.
        z: Interaction depth from cathode (0 to d)
        d: Detector thickness
        lambda_e: Electron drift length (mu_e * tau_e * E)
        lambda_h: Hole drift length (mu_h * tau_h * E)
        """
        # Normalized drift lengths
        le = lambda_e / d
        lh = lambda_h / d
        
        # Avoid division by zero
        le = max(le, 1e-9)
        lh = max(lh, 1e-9)
        
        # CCE for electrons (moving to anode at z=d) - interaction at z
        # Distance to travel: d - z
        cce_e = le * (1 - np.exp(-(d - z) / (lambda_e + 1e-12)))
        
        # CCE for holes (moving to cathode at z=0) - interaction at z
        # Distance to travel: z
        cce_h = lh * (1 - np.exp(-z / (lambda_h + 1e-12)))
        
        return cce_e + cce_h

# K-Edge Energies (keV) for common detector elements
K_EDGE_ENERGIES = {
    "Si": 1.84,
    "Ge": 11.1,
    "Ga": 10.37,
    "As": 11.87,
    "Cd": 26.71,
    "Te": 31.81,
    "Zn": 9.66,
    "Se": 12.66,
    "Hg": 83.10,
    "I": 33.17,
    "Tl": 85.53,
    "Pb": 88.00,
    "Br": 13.47
}

class MockXrayLib:
    """
    Mock implementation of X-ray attenuation coefficients.
    Uses a simplified model: mu = rho * sum(w_i * (A * E^(-3) * Step + B))
    """
    @staticmethod
    def get_attenuation(material, energy_keV):
        """
        Calculate linear attenuation coefficient (1/cm) for a material.
        material: MaterialProperties object
        energy_keV: array-like or float
        """
        energies = np.array(energy_keV, dtype=float)
        mu_total = np.zeros_like(energies)
        
        # Base Compton scattering (approx constant/weakly energy dependent > 100keV, 
        # but for <100keV it's flatish compared to photo)
        # Klein-Nishina is better but let's use a simplified constant for this demo
        sigma_compton = 0.15 # cm2/g approx
        
        for element, fraction in material.atomic_composition.items():
            k_edge = K_EDGE_ENERGIES.get(element, 0)
            
            # Photoelectric effect ~ Z^4 / E^3
            # Z approximation: Si(14), Ge(32), Cd(48), Te(52), etc.
            # We can use a pre-factor roughly proportional to Z^4
            # Let's just use a lookup or simple heuristic if Z not available
            # Heuristic: K-edge energy is roughly proportional to Z^2 (Moseley's law)
            # So Z ~ sqrt(K_edge). A ~ Z^4 ~ K_edge^2
            
            if k_edge > 0:
                amplitude = 5e3 * (k_edge**2.5) # Empirical scaling for demo
            else:
                amplitude = 1e5 # Default for light elements like Si
            
            # Photoelectric term
            mu_photo = amplitude * (energies**-3.0)
            
            # K-edge Jump
            # If E < K_edge, absorption drops significantly (L-edge only)
            # Jump factor ~ 5-10
            jump_mask = energies < k_edge
            mu_photo[jump_mask] /= 8.0 
            
            # Add to total (mass attenuation)
            mu_element = mu_photo + sigma_compton
            mu_total += fraction * mu_element
            
        # Linear attenuation = Mass attenuation * Density
        return mu_total * material.density

    @staticmethod
    def get_k_edges(material):
        """Return dict of {Element: Energy} for K-edges in the material."""
        edges = {}
        for elem in material.atomic_composition:
            if elem in K_EDGE_ENERGIES:
                edges[elem] = K_EDGE_ENERGIES[elem]
        return edges

