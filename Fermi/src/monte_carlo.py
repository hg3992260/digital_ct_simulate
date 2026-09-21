import numpy as np
from src.physics import MockXrayLib

# Try to import xraylib for cross-sections
try:
    import xraylib
    HAS_XRAYLIB = True
except ImportError:
    HAS_XRAYLIB = False

class MonteCarloEngine:
    """
    A simplified 1D Monte Carlo engine for photon transport in semiconductor detectors.
    Simulates:
    1. Photoelectric absorption
    2. Compton scattering (simplified)
    3. Fluorescence escape (K-alpha)
    """
    
    def __init__(self, material, thickness_cm, n_photons=10000):
        self.material = material
        self.d = thickness_cm
        self.n_photons = n_photons
        
    def get_cross_sections(self, energy_keV):
        """
        Get partial cross sections (cm2/g) for Photoelectric and Compton.
        Returns: sigma_photo, sigma_compton, sigma_total
        """
        if HAS_XRAYLIB:
            # Construct formula
            formula = ""
            for elem, frac in self.material.atomic_composition.items():
                formula += f"{elem}{frac}"
            
            try:
                # Xraylib CS_Total includes Rayleigh, but we ignore Rayleigh for energy deposition
                # CS_Photo_CP: Photoelectric
                # CS_Compt_CP: Compton
                sig_pe = xraylib.CS_Photo_CP(formula, energy_keV)
                sig_co = xraylib.CS_Compt_CP(formula, energy_keV)
                return sig_pe, sig_co, sig_pe + sig_co
            except:
                pass
        
        # Fallback Mock Model
        # Photoelectric ~ E^-3
        sig_pe = 1e5 * (energy_keV**-3)
        # Compton ~ const (simplified)
        sig_co = 0.15 
        return sig_pe, sig_co, sig_pe + sig_co

    def run(self, incident_energy_keV):
        """
        Run the simulation.
        Returns: 
            deposited_energies: List of energy deposited in the detector per photon
            interaction_depths: List of primary interaction depths (for CCE calculation)
        """
        deposited_energies = []
        interaction_depths = []
        
        # Pre-calculate cross-sections at incident energy
        sig_pe, sig_co, sig_total = self.get_cross_sections(incident_energy_keV)
        
        # Linear attenuation coeff (cm^-1) = sigma * rho
        mu_total = sig_total * self.material.density
        mu_pe = sig_pe * self.material.density
        mu_co = sig_co * self.material.density
        
        # Probabilities relative to total interaction
        # Ensure scalar comparison for probability check
        if np.isscalar(mu_total):
             if mu_total > 0:
                 prob_pe = mu_pe / mu_total
             else:
                 prob_pe = 1.0
        else:
             # If mu_total is an array (shouldn't be for monoenergetic, but just in case)
             prob_pe = np.divide(mu_pe, mu_total, out=np.zeros_like(mu_pe), where=mu_total!=0)
        
        # 1. Generate interaction depths for all photons
        # P(z) = mu * exp(-mu * z)
        # Sample z = -ln(1 - rand) / mu
        # Or simply z = -ln(rand) / mu
        
        # Ensure mu_total is valid for division
        if np.isscalar(mu_total) and mu_total <= 0:
             # If mu_total is zero or negative (impossible physically but check for math safety)
             # Then no interaction -> z = infinity
             mu_total = 1e-9 # Avoid division by zero
        
        # Simulate N photons
        # Vectorized generation of random numbers
        r1 = np.random.random(self.n_photons)
        
        # Safe division
        with np.errstate(divide='ignore'):
             z_interact = -np.log(r1) / mu_total
             
        # Force cast to ensure standard numpy behavior
        z_interact = np.asarray(z_interact, dtype=float)
        
        # Filter photons that passed through without interaction (z > d)
        valid_mask = z_interact <= self.d
        
        # Explicitly use integer indices to filter z_interact
        valid_indices = np.where(valid_mask)[0].astype(int)
        z_interact = z_interact[valid_indices]
        n_interacted = len(z_interact)
        
        if n_interacted == 0:
             return np.array([]), np.array([])
        
        # 2. Determine interaction type (Photoelectric vs Compton)
        r2 = np.random.random(n_interacted)
        is_photoelectric = r2 < prob_pe
        
        # Arrays to store energy deposited
        E_deposited = np.zeros(n_interacted)
        
        # --- Handle Photoelectric Events ---
        
        # Ensure indices are integers
        pe_indices = np.where(is_photoelectric)[0].astype(int)
        # Use integer indices for assignment
        E_deposited[pe_indices] = 0.0 # Placeholder, will be overwritten

        # Usually deposits full energy, UNLESS fluorescence escape occurs
        # Fluorescence yield increases with Z. 
        # Simplified: If interaction is PE, check if K-shell vacancy creates X-ray that escapes.
        
        # Get heaviest element K-edge and Fluorescence yield (approx)
        # For simplicity, check the dominant heavy element (e.g. Cd or Te or Tl)
        heavy_elem = None
        max_z = 0
        for elem in self.material.atomic_composition:
            try:
                z = xraylib.SymbolToAtomicNumber(elem) if HAS_XRAYLIB else 0
                if z > max_z:
                    max_z = z
                    heavy_elem = elem
            except:
                pass
        
        k_edge = 0
        fluor_yield = 0
        fluor_energy = 0
        
        if HAS_XRAYLIB and heavy_elem:
            try:
                k_edge = xraylib.EdgeEnergy(max_z, xraylib.K_SHELL)
                fluor_yield = xraylib.FluorYield(max_z, xraylib.K_SHELL)
                # K-alpha energy approx
                fluor_energy = xraylib.LineEnergy(max_z, xraylib.KA1_LINE)
            except:
                pass
        
        # If incident energy > K-edge, fluorescence is possible
        if incident_energy_keV > k_edge and k_edge > 0:
            # Mask for PE events
            # IMPORTANT: Ensure pe_indices is an array of integers
            pe_indices = np.where(is_photoelectric)[0]
            n_pe = len(pe_indices)
            
            if n_pe > 0:
                # Check if fluorescence photon is emitted
                r_fluor = np.random.random(n_pe)
                emitted_mask = r_fluor < fluor_yield
                
                # Check if emitted photon escapes
                # Simplified: Isotropic emission. 
                # Escape prob approx exp(-mu_fluor * distance_to_surface)
                # 1D approx: distance is min(z, d-z). 50% chance up, 50% down.
                # Let's just give a fixed escape probability for near-surface events?
                # Better: Calculate re-absorption path
                
                # Get attenuation for fluorescence photon
                _, _, mu_fluor_total = self.get_cross_sections(fluor_energy)
                mu_fluor = mu_fluor_total * self.material.density
                
                # Random direction cosine (isotropic)
                # costheta = 2*rand - 1. Path length s = dist / |costheta|
                # Here we just check escape.
                # Distance to boundary:
                z_pe = z_interact[pe_indices]
                
                # 50% chance forward (d-z), 50% backward (z)
                r_dir = np.random.random(n_pe)
                dist_to_boundary = np.where(r_dir > 0.5, self.d - z_pe, z_pe)
                
                # Optical depth
                tau = mu_fluor * dist_to_boundary
                # Transmission prob = exp(-tau)
                r_escape = np.random.random(n_pe)
                is_escaped = (r_escape < np.exp(-tau)) & emitted_mask
                
                # Energy deposited = E_inc - E_fluor (if escaped)
                #                  = E_inc (if reabsorbed or no fluorescence)
                current_E = np.full(n_pe, incident_energy_keV, dtype=float)
                
                # Use boolean indexing directly
                # is_escaped is a boolean array of length n_pe
                # current_E is a float array of length n_pe
                # We iterate manually to be absolutely safe against numpy version quirks
                for i in range(n_pe):
                    if is_escaped[i]:
                        current_E[i] -= fluor_energy
                
                E_deposited[pe_indices] = current_E
            
        else:
            # No fluorescence possible or negligible
            E_deposited[is_photoelectric] = incident_energy_keV
            
        # --- Handle Compton Events ---
        # E_deposited = E_electron. Scattered photon E' escapes (simplified 1-step)
        # E_electron = E - E'
        # E' = E / (1 + (E/511)*(1 - cos theta))
        # Klein-Nishina distribution needed for angle sampling.
        # Simplified: Flat distribution or average energy transfer?
        # Let's use a rough approximation: Compton edge is max transfer.
        # Max energy transfer at 180 deg backscatter.
        
        compt_indices = np.where(~is_photoelectric)[0].astype(int)
        n_compt = len(compt_indices)
        
        if n_compt > 0:
            # Max energy transfer (Compton Edge)
            alpha = incident_energy_keV / 511.0
            E_backscatter = incident_energy_keV / (1 + 2*alpha)
            E_max_transfer = incident_energy_keV - E_backscatter
            
            # Sample energy transfer (0 to E_max_transfer)
            # Klein-Nishina favors forward scattering (low energy transfer) at high E
            # But at low E (diagnostic X-ray), it's more symmetric.
            # Uniform sampling for simplicity in this demo
            E_deposited[compt_indices] = np.random.random(n_compt) * E_max_transfer
            
        return E_deposited, z_interact
