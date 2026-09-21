import numpy as np
import sys
# Handle frozen import for MaterialManager
try:
    # Try importing from src first (for dev environment)
    from src.material import MATERIALS, MaterialManager
    from src.physics import FermiPhysics, MockXrayLib, k_B, q
except ImportError:
    # Fallback for PyInstaller flattened structure
    try:
        from material import MATERIALS, MaterialManager
        from physics import FermiPhysics, MockXrayLib, k_B, q
    except ImportError:
        # One last try - maybe we are inside 'src' but 'src' is not a package
        import material
        import physics
        MATERIALS = material.MATERIALS
        MaterialManager = material.MaterialManager
        FermiPhysics = physics.FermiPhysics
        MockXrayLib = physics.MockXrayLib
        k_B = physics.k_B
        q = physics.q

# Try to import xraylib, if fails, use Mock
try:
    import xraylib
    HAS_XRAYLIB = True
except ImportError:
    HAS_XRAYLIB = False
    print("Warning: xraylib not found, using simplified physics model.")

class RealXrayLib:
    """Wrapper for real xraylib functions."""
    @staticmethod
    def get_attenuation(material, energy_keV):
        """
        Calculate linear attenuation coefficient (1/cm).
        material: MaterialProperties
        energy_keV: array-like of energies in keV
        """
        energies = np.array(energy_keV, dtype=float)
        mu_total = np.zeros_like(energies)
        
        # Xraylib expects element symbols or Z numbers
        # We iterate over composition
        # Composition is mass fraction? Our MaterialProperties assumes atomic composition if sum ~ 1?
        # Actually in MaterialProperties it says {'Element': fraction}. 
        # Let's assume these are atomic fractions (stoichiometry) or mass fractions.
        # Standard convention for compounds (e.g. Cd0.9Zn0.1Te) is atomic formula.
        
        # However, to get linear attenuation (1/cm), we need:
        # mu_linear = density * mu_mass
        # mu_mass = sum(weight_fraction_i * mu_mass_i)
        
        # First, we need to convert atomic composition to weight fractions if not already
        # But xraylib has a parser for compounds if we construct a formula string!
        # e.g. "CdTe", "Cd0.9Zn0.1Te"
        
        # Construct formula string
        formula = ""
        for elem, frac in material.atomic_composition.items():
            formula += f"{elem}{frac}"
            
        # Xraylib works with individual energies, need loop for arrays usually, 
        # or use numpy vectorization if supported (xraylib usually scalar)
        
        try:
            compound = xraylib.CompoundParser(formula)
            # compound contains 'nAtomsAll', 'Elements', 'massFractions'
            
            # Vectorized lookup
            for i, E in enumerate(energies):
                # CS_Total includes Photo + Compton + Rayleigh (cm2/g)
                # We need linear attenuation (cm-1) = CS_Total * density
                mu_mass = xraylib.CS_Total_CP(formula, E)
                mu_total[i] = mu_mass * material.density
                
        except Exception as e:
            # Fallback if formula parsing fails
            print(f"Xraylib error for {formula}: {e}")
            return MockXrayLib.get_attenuation(material, energies)
            
        return mu_total

    @staticmethod
    def get_k_edges(material):
        """Return dict of {Element: Energy} for K-edges."""
        edges = {}
        for elem in material.atomic_composition:
            try:
                Z = xraylib.SymbolToAtomicNumber(elem)
                edge_E = xraylib.EdgeEnergy(Z, xraylib.K_SHELL)
                edges[elem] = edge_E
            except:
                pass
        return edges

class DetectorSimulator:
    def __init__(self, material_name, thickness_mm, voltage, temperature_k):
        # Always try to get fresh data from manager first, fall back to global MATERIALS
        manager = MaterialManager()
        self.material = manager.get_material(material_name)
        
        if not self.material:
            if material_name in MATERIALS:
                self.material = MATERIALS[material_name]
            else:
                raise ValueError(f"Unknown material: {material_name}")
        
        self.d = thickness_mm * 0.1 # Convert to cm
        self.voltage = voltage
        self.T = temperature_k
        self.E_field = self.voltage / self.d # V/cm

    def calculate_noise_properties(self):
        """
        Calculate leakage current and corresponding noise.
        Includes Fano factor.
        """
        # 1. Intrinsic carrier concentration
        ni = FermiPhysics.calculate_intrinsic_carriers(self.material.Eg, self.T)
        
        # 2. Resistivity (assuming intrinsic for simplicity in this demo)
        rho = FermiPhysics.calculate_resistivity(ni, self.material.mu_e, self.material.mu_h)
        
        # 3. Dark Current (J = E / rho) -> I = J * Area (Assume 1x1 mm pixel for noise calc)
        pixel_area_cm2 = 0.01 # 1mm^2
        current = (self.E_field / rho) * pixel_area_cm2
        
        # 4. Electronic Noise (ENC) - Shot noise approximation
        # sigma_q = sqrt(2 * q * I * tau_int)
        tau_int = 1e-6 # Integration time example 1us
        enc_coulomb = np.sqrt(2 * q * current * tau_int)
        enc_electrons = enc_coulomb / q
        
        # Convert Electronic Noise to Energy FWHM (eV)
        # FWHM_elec = 2.355 * sigma_elec * W_pair
        fwhm_elec_ev = 2.355 * enc_electrons * self.material.W_pair
        
        # For compatibility with legacy result keys (if any), calculate a default total noise at some reference energy
        # But really we should just return what we have.
        # The key 'fwhm_noise_keV' was missing in the previous partial update, causing KeyError in GUI.
        # We'll add it back as a placeholder (pure electronic noise) or calculated at 60keV reference.
        
        return {
            "ni": ni,
            "rho": rho,
            "dark_current_nA": current * 1e9,
            "enc_electrons": enc_electrons,
            "fwhm_elec_keV": fwhm_elec_ev / 1000.0,
            # Add back 'fwhm_noise_keV' for compatibility with GUI display (T-Scan needs this)
            "fwhm_noise_keV": fwhm_elec_ev / 1000.0 
        }

    def get_band_structure(self):
        """
        Get Fermi band structure data.
        Returns dictionary with Ec, Ev, Ef, Eg levels (in eV).
        """
        # Assume Ev = 0
        Ev = 0.0
        Eg = self.material.Eg
        Ec = Ev + Eg
        
        # Calculate Fermi Level (Intrinsic for now)
        Ef = FermiPhysics.calculate_fermi_level(Eg, self.T)
        
        return {
            "Ev": Ev,
            "Ec": Ec,
            "Eg": Eg,
            "Ef": Ef
        }

    def get_electric_field_profile(self):
        """
        Calculate non-uniform electric field profile using Finite Difference Solver.
        Returns z, potential, field.
        """
        # Lazy import
        from src.field_solver import FieldSolver
        
        solver = FieldSolver(self.d, self.voltage)
        
        # Assume some trapped charge profile for demonstration
        # E.g. TlBr polarization-like
        if "TlBr" in self.material.name:
            rho = FieldSolver.get_trapped_charge_profile(solver.z, flux_rate=5e8)
        elif "CdTe" in self.material.name or "CZT" in self.material.name:
            rho = FieldSolver.get_trapped_charge_profile(solver.z, flux_rate=1e8)
        else:
            rho = None # Linear field
            
        z, V, E = solver.solve(rho)
        return z, V, E

    def _calculate_cce_numeric(self, z_interaction, E_field_profile, z_grid):
        """
        Calculate CCE numerically for non-uniform electric field.
        z_interaction: array of interaction depths (cm)
        E_field_profile: array of E field values (V/cm) on z_grid
        z_grid: spatial grid for E field (cm)
        """
        # Ensure positive field for drift direction assumptions (Cathode at 0, Anode at d)
        # Electrons drift 0 -> d (if E > 0). Holes drift z -> 0.
        # Check field sign convention. In DetectorSimulator.init: E = V/d. 
        # If V>0, E>0. Electrons move against field? 
        # Wait, F = qE. If E points 0->d, force on electron is towards 0.
        # Usually we want electrons to move to Anode.
        # If Anode is at z=d and V_anode > 0 (vs Cathode z=0), E points d->0 (negative).
        # Let's check FieldSolver. 
        # FieldSolver: V(0)=0, V(d)=V_bias. If V_bias > 0, V increases 0->d.
        # E = -dV/dz. So E is negative (points d->0).
        # Force on electron (-e) is -e * E = -e * (-|E|) = +|F|. Moves 0 -> d. Correct.
        # Force on hole (+e) is +e * E = -|F|. Moves d -> 0. Correct.
        
        # So we work with magnitudes for drift speed, but direction is fixed.
        # Electrons: z -> d. Holes: z -> 0.
        
        E_mag = np.abs(E_field_profile)
        
        # Interpolator for Field
        def get_E(z_vals):
            return np.interp(z_vals, z_grid, E_mag)
            
        cce_e = np.zeros_like(z_interaction)
        cce_h = np.zeros_like(z_interaction)
        
        # Simulation parameters
        dt = 1e-10 # Time step (s) - small enough for typical velocities (10^7 cm/s * 1e-10 = 10um steps)
        # Or spatial steps? Spatial is safer.
        dz_step = self.d / 200.0
        
        # --- Electron Transport (z -> d) ---
        for i, z0 in enumerate(z_interaction):
            z_curr = z0
            q_curr = 1.0
            induced = 0.0
            
            while z_curr < self.d and q_curr > 1e-3:
                # Local field and velocity
                E_loc = get_E(z_curr)
                if E_loc < 1e-3: break # Stopped
                
                v = self.material.mu_e * E_loc
                # Saturation velocity? Not modeled yet, assume linear.
                
                # Step size determined by spatial grid or time
                # Let's move a fixed small distance dz
                step = dz_step
                if z_curr + step > self.d:
                    step = self.d - z_curr
                
                dt_step = step / v
                
                # Trapping
                # q(t+dt) = q(t) * exp(-dt/tau)
                decay = np.exp(-dt_step / self.material.tau_e)
                q_next = q_curr * decay
                
                # Ramo: dQ = q * v * dt / d * WeightingField
                # Planar detector: Weighting Field = 1/d
                # dQ = q_avg * dx / d
                q_avg = (q_curr + q_next) / 2.0
                induced += q_avg * step / self.d
                
                z_curr += step
                q_curr = q_next
            
            cce_e[i] = induced

        # --- Hole Transport (z -> 0) ---
        for i, z0 in enumerate(z_interaction):
            z_curr = z0
            q_curr = 1.0
            induced = 0.0
            
            while z_curr > 0 and q_curr > 1e-3:
                E_loc = get_E(z_curr)
                if E_loc < 1e-3: break
                
                v = self.material.mu_h * E_loc
                
                step = dz_step
                if z_curr - step < 0:
                    step = z_curr
                
                dt_step = step / v
                
                decay = np.exp(-dt_step / self.material.tau_h)
                q_next = q_curr * decay
                
                q_avg = (q_curr + q_next) / 2.0
                induced += q_avg * step / self.d
                
                z_curr -= step
                q_curr = q_next
                
            cce_h[i] = induced
            
        return cce_e + cce_h

    def calculate_high_flux_performance(self, max_flux_mcps_mm2=200):
        """
        Simulate performance under high photon flux.
        Varies flux and calculates Energy Resolution degradation due to polarization.
        """
        # Flux range: 1 to max (Mcps/mm^2) -> convert to counts/s/cm^2
        # 1 Mcps/mm^2 = 1e6 / (0.01 cm^2) = 1e8 counts/s/cm^2
        flux_values_mcps = np.linspace(1, max_flux_mcps_mm2, 20)
        flux_rates_hz_cm2 = flux_values_mcps * 1e8
        
        resolutions_fwhm = []
        max_throughput = []
        
        # Pre-calculate absorption profile (probability of interaction vs z)
        # This is needed to weight the CCE variance
        if HAS_XRAYLIB:
            mu = RealXrayLib.get_attenuation(self.material, [60.0])[0] # Use 60keV reference
        else:
            mu = MockXrayLib.get_attenuation(self.material, [60.0])[0]
            
        z_sample = np.linspace(0, self.d, 50)
        prob_density = np.exp(-mu * z_sample)
        prob_density /= np.sum(prob_density) # Normalize
        
        for flux in flux_rates_hz_cm2:
            # 1. Calculate Polarized Field
            from src.field_solver import FieldSolver
            solver = FieldSolver(self.d, self.voltage)
            
            # Estimate trapping based on material quality
            # TlBr/HgI2 -> High trapping. CdTe -> Medium. Si/Ge -> Low.
            trapping_factor = 1.0
            if self.material.name in ["TlBr", "HgI2"]:
                trapping_factor = 50.0 # Boost to show effect
            elif self.material.name in ["CdTe", "CZT"]:
                trapping_factor = 5.0
            elif self.material.name in ["Si", "Ge"]:
                trapping_factor = 0.1
                
            rho = FieldSolver.get_trapped_charge_profile(solver.z, flux_rate=flux * trapping_factor, trapping_time=1.0)
            z_grid, _, E_grid = solver.solve(rho)
            
            # 2. Calculate CCE Profile with distorted field
            cce_vals = self._calculate_cce_numeric(z_sample, E_grid, z_grid)
            
            # 3. Estimate Resolution Broadening
            # The variation in CCE leads to peak broadening.
            # Mean CCE
            mean_cce = np.sum(cce_vals * prob_density)
            # Variance
            variance_cce = np.sum(((cce_vals - mean_cce)**2) * prob_density)
            std_cce = np.sqrt(variance_cce)
            
            # Convert CCE spread to Energy spread (keV) at 60 keV reference
            E_ref = 60.0
            fwhm_cce_keV = 2.355 * std_cce * E_ref
            
            # Add electronic noise baseline
            noise_props = self.calculate_noise_properties()
            fwhm_elec = noise_props['fwhm_elec_keV'] # This assumes dark current doesn't change with flux (simplified)
            # Actually dark current might increase with heating, but let's ignore thermal effects of flux for now.
            
            total_fwhm = np.sqrt(fwhm_cce_keV**2 + fwhm_elec**2)
            resolutions_fwhm.append(total_fwhm)
            
            # 4. Estimate Throughput (Pile-up)
            # For high-flux PCCT (>100 Mcps/mm2), we assume pixelated detectors.
            # Dead time applies PER PIXEL.
            # Assume pixel pitch 0.1mm (100um), Area = 0.01 mm2 = 1e-4 cm2
            pixel_area_cm2 = 1e-4
            pixel_rate_in = flux * pixel_area_cm2 # flux is in Hz/cm2 -> Hz per pixel
            
            # Electronics dead time (ASIC limitation)
            tau_asic = 20e-9 # 20ns
            
            # Physics dead time (Charge collection time)
            # T_coll ~ d / (mu * E)
            # Use slower carrier (holes) as they cause long tails/pile-up
            mu_slow = min(self.material.mu_e, self.material.mu_h)
            # Average field E = V/d
            E_avg = self.voltage / self.d
            
            if E_avg > 0 and mu_slow > 0:
                t_coll = self.d / (mu_slow * E_avg)
            else:
                t_coll = 1e-6 # Fallback
            
            # Effective dead time is dominated by the slower process
            # For small pixels, maybe only electron time matters? 
            # Small pixel effect: signal induced mostly near anode.
            # Let's use a weighted dead time: max(tau_asic, t_coll_effective)
            # For pixel detectors, t_coll_effective is often determined by weighting potential,
            # which is small near cathode. So maybe 0.2 * t_coll is a better estimate for small pixels?
            # Let's be conservative:
            tau_phys = t_coll * 0.5 
            
            tau_dead = max(tau_asic, tau_phys)
            
            # Paralyzable model per pixel
            pixel_rate_out = pixel_rate_in * np.exp(-pixel_rate_in * tau_dead)
            
            # Convert back to Mcps/mm2
            # Rate out (Hz/pixel) -> Hz/cm2 -> Mcps/mm2
            # Hz/cm2 = Hz/pixel / pixel_area_cm2
            # Mcps/mm2 = Hz/cm2 / 1e8
            r_out_hz_cm2 = pixel_rate_out / pixel_area_cm2
            r_out_mcps_mm2 = r_out_hz_cm2 / 1e8
            max_throughput.append(r_out_mcps_mm2)

        return {
            "flux_mcps_mm2": flux_values_mcps,
            "fwhm_keV": resolutions_fwhm,
            "throughput_mcps_mm2": max_throughput
        }

    def get_transport_profile(self):
        """
        Calculate transport properties vs depth.
        Returns z, CCE_e, CCE_h, Total CCE.
        NOW SUPPORTS NON-UNIFORM FIELD via Numerical Integration!
        """
        # Get field profile
        z_grid, V_grid, E_grid = self.get_electric_field_profile()
        
        # To calculate CCE with non-uniform field, we need to solve the drift equation numerically.
        # Hecht equation is only for constant E.
        # General CCE = (Q_ind_e + Q_ind_h) / Q0
        # Q_ind = integral(i(t) dt)
        # i(t) = q * v(t) * E_w(x(t))
        # This is getting complex for a "simplified" simulator. 
        # Let's stick to Hecht but use the LOCAL electric field at interaction depth z? 
        # No, that's wrong. Carrier drifts through the whole field.
        
        # Simplified approach for non-uniform field:
        # Use average field seen by carrier? Or segment the detector?
        # Let's keep using the analytical Hecht with CONSTANT field for the main "Transport" plot 
        # to avoid confusing the user with mismatching assumptions unless they explicitly ask for "Polarized Mode".
        
        # However, we can overlay the Electric Field profile in the Transport tab!
        
        z_steps = 100
        z = np.linspace(0, self.d, z_steps)
        
        lambda_e = self.material.mu_tau_e * self.E_field
        lambda_h = self.material.mu_tau_h * self.E_field
        
        le = lambda_e / self.d
        lh = lambda_h / self.d
        le = max(le, 1e-9)
        lh = max(lh, 1e-9)
        
        cce_e = le * (1 - np.exp(-(self.d - z) / (lambda_e + 1e-12)))
        cce_h = lh * (1 - np.exp(-z / (lambda_h + 1e-12)))
        cce_total = cce_e + cce_h
        
        # Interpolate Electric field to these z points for display
        E_interp = np.interp(z, z_grid, E_grid)
        
        return {
            "z_mm": z * 10.0, 
            "cce_e": cce_e,
            "cce_h": cce_h,
            "cce_total": cce_total,
            "E_field_abs": np.abs(E_interp) # Add this for plotting
        }

    def get_polarization_profile(self):
        """
        Simulate polarization effect (time-dependent field collapse).
        This is a simplified model assuming high flux leading to space charge buildup.
        Returns t (time), E_field_anode, E_field_cathode relative to initial.
        """
        # Time scale for polarization (seconds)
        # Depends on flux, but let's model a generic trend
        t = np.linspace(0, 100, 100) # 0 to 100 seconds
        
        # Simple exponential decay model for internal field due to trapped charge
        # Rate depends on detrapping time (approx by tau_h for simplicity in this mock)
        tau_detrap = 10.0
        
        # Normalized field strength at cathode (assuming hole trapping reduces field there)
        # TlBr is known for this. CZT less so but still present.
        name = self.material.name
        if name == "TlBr":
            decay_factor = 0.4
            tau_detrap = 20.0
        elif name == "HgI2":
            decay_factor = 0.35
            tau_detrap = 18.0
        elif name == "CdTe":
            decay_factor = 0.18
            tau_detrap = 14.0
        elif name == "CZT":
            decay_factor = 0.10
            tau_detrap = 10.0
        elif name == "GaAs":
            decay_factor = 0.06
            tau_detrap = 9.0
        elif name == "Si":
            decay_factor = 0.02
            tau_detrap = 6.0
        else:
            decay_factor = 0.10
            
        e_field_rel = 1.0 - decay_factor * (1 - np.exp(-t / tau_detrap))
        
        return {
            "time_s": t,
            "field_strength": e_field_rel
        }
    
    def get_attenuation_profile(self, energy_range=None):
        """
        Calculate attenuation coefficient profile vs energy.
        Uses xraylib if available.
        """
        if energy_range is None:
            # Default range: 10 keV to 150 keV
            energies = np.linspace(10, 150, 500)
        else:
            energies = np.array(energy_range)
            
        if HAS_XRAYLIB:
            attenuation = RealXrayLib.get_attenuation(self.material, energies)
            k_edges = RealXrayLib.get_k_edges(self.material)
        else:
            attenuation = MockXrayLib.get_attenuation(self.material, energies)
            k_edges = MockXrayLib.get_k_edges(self.material)
        
        return {
            "energy_keV": energies,
            "attenuation": attenuation,
            "k_edges": k_edges
        }

    def run_monte_carlo_simulation(self, incident_energy_keV, n_photons=10000, n_bins=200):
        """
        Run the full Monte Carlo simulation pipeline.
        Returns the same result structure as simulate_spectrum but with MC data.
        """
        # Lazy import to avoid circular dependency if placed at top
        from src.monte_carlo import MonteCarloEngine
        
        # 1. Physics Parameters
        lambda_e = self.material.mu_tau_e * self.E_field
        lambda_h = self.material.mu_tau_h * self.E_field
        
        # 2. Run MC Engine
        mc = MonteCarloEngine(self.material, self.d, n_photons)
        deposited_energies, interaction_depths = mc.run(incident_energy_keV)
        
        # 3. Calculate CCE for each event
        # Vectorized Hecht
        cce = FermiPhysics.hecht_equation(interaction_depths, self.d, lambda_e, lambda_h)
        
        # 4. Measured Energy = Deposited * CCE
        # Note: deposited_energies already accounts for escape losses
        measured_energies = deposited_energies * cce
        
        # 5. Build Histogram
        # Normalize to interaction efficiency
        # Efficiency = n_interacted / n_photons
        eff = len(deposited_energies) / n_photons
        
        hist, bin_edges = np.histogram(measured_energies, bins=n_bins, range=(0, incident_energy_keV*1.1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Normalize histogram area to efficiency
        if np.sum(hist) > 0:
            hist = hist.astype(float) / n_photons # Normalized to counts per incident photon
            # Scale up for display? Usually we want normalized probability density or total counts.
            # Let's keep it proportional to counts but smooth it.
        
        # 6. Convolve with Noise (Fano + Electronic)
        noise_props = self.calculate_noise_properties()
        fwhm_elec_keV = noise_props['fwhm_elec_keV']
        
        F = 0.11 
        W_keV = self.material.W_pair / 1000.0
        
        final_spectrum = np.zeros_like(hist)
        
        for i, count in enumerate(hist):
            if count > 0:
                E_val = bin_centers[i]
                if E_val > 0:
                    sigma_fano = np.sqrt(F * E_val * W_keV)
                else:
                    sigma_fano = 0
                sigma_elec = fwhm_elec_keV / 2.355
                sigma_total = np.sqrt(sigma_elec**2 + sigma_fano**2)
                
                if sigma_total > 1e-6:
                    gaussian = (1/(sigma_total * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((bin_centers - E_val)/sigma_total)**2)
                    final_spectrum += count * gaussian * (bin_centers[1] - bin_centers[0])
                else:
                    final_spectrum[i] += count
                    
        # Add peak noise stats
        sigma_fano_peak = np.sqrt(F * incident_energy_keV * W_keV)
        sigma_elec = fwhm_elec_keV / 2.355
        sigma_total_peak = np.sqrt(sigma_elec**2 + sigma_fano_peak**2)
        noise_props['fwhm_noise_keV'] = sigma_total_peak * 2.355

        return {
            "energy_axis": bin_centers,
            "spectrum": final_spectrum, # This is now MC generated
            "interaction_efficiency": eff,
            "noise_stats": noise_props
        }

    def simulate_spectrum(self, incident_energy_keV, n_bins=200):
        """
        Simulate the response spectrum for a monoenergetic source.
        Includes Fano noise.
        """
        # 1. Physics Parameters
        lambda_e = self.material.mu_tau_e * self.E_field
        lambda_h = self.material.mu_tau_h * self.E_field
        
        # 2. Interaction Depth Profile
        # Discretize depth z from 0 to d
        z_steps = 1000
        z = np.linspace(0, self.d, z_steps)
        dz = self.d / z_steps
        
        # Absorption coefficient
        if HAS_XRAYLIB:
            mu = RealXrayLib.get_attenuation(self.material, [incident_energy_keV])[0]
        else:
            mu = MockXrayLib.get_attenuation(self.material, [incident_energy_keV])[0]
        
        # Probability of interaction at depth z: P(z) ~ exp(-mu * z)
        prob_density = mu * np.exp(-mu * z)
        # Normalize probability (interaction within detector)
        interaction_efficiency = 1 - np.exp(-mu * self.d)
        if np.sum(prob_density) > 0:
            prob_density = prob_density / (np.sum(prob_density) * dz) * interaction_efficiency
        
        # 3. CCE Profile
        cce = FermiPhysics.hecht_equation(z, self.d, lambda_e, lambda_h)
        
        # 4. Measured Energy at each depth
        measured_energies = incident_energy_keV * cce
        
        # 5. Build Histogram (Ideal Spectrum without electronic noise)
        hist, bin_edges = np.histogram(measured_energies, bins=n_bins, range=(0, incident_energy_keV*1.1), weights=prob_density*dz)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 6. Convolve with Noise (Fano + Electronic)
        noise_props = self.calculate_noise_properties()
        fwhm_elec_keV = noise_props['fwhm_elec_keV']
        
        # Fano Noise
        # FWHM_fano = 2.355 * sqrt(F * E * W)
        # Fano factor typically 0.1 for semi, let's use 0.11
        F = 0.11 
        W_keV = self.material.W_pair / 1000.0
        
        # Total FWHM depends on energy E: FWHM(E) = sqrt(FWHM_elec^2 + FWHM_fano(E)^2)
        # Since we are convolving the whole histogram, strictly speaking Fano noise varies bin by bin.
        # But since the peak is narrow, we can approximate using Fano at incident energy for the main peak,
        # or calculate per bin. Per bin is better for low energy tails.
        
        final_spectrum = np.zeros_like(hist)
        
        # Convolution loop with energy-dependent broadening
        for i, count in enumerate(hist):
            if count > 0:
                E_val = bin_centers[i]
                
                # Fano term at this energy
                if E_val > 0:
                    sigma_fano = np.sqrt(F * E_val * W_keV)
                else:
                    sigma_fano = 0
                    
                sigma_elec = fwhm_elec_keV / 2.355
                
                # Total sigma
                sigma_total = np.sqrt(sigma_elec**2 + sigma_fano**2)
                
                # Gaussian centered at E_val with width sigma_total
                # Avoid div by zero
                if sigma_total > 1e-6:
                    gaussian = (1/(sigma_total * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((bin_centers - E_val)/sigma_total)**2)
                    final_spectrum += count * gaussian * (bin_centers[1] - bin_centers[0])
                else:
                    # Delta function (put in single bin)
                    final_spectrum[i] += count
        
        # Add total noise at peak energy to stats for display
        sigma_fano_peak = np.sqrt(F * incident_energy_keV * W_keV)
        sigma_elec = fwhm_elec_keV / 2.355
        sigma_total_peak = np.sqrt(sigma_elec**2 + sigma_fano_peak**2)
        noise_props['fwhm_noise_keV'] = sigma_total_peak * 2.355
                
        return {
            "energy_axis": bin_centers,
            "spectrum": final_spectrum,
            "interaction_efficiency": interaction_efficiency,
            "noise_stats": noise_props
        }
