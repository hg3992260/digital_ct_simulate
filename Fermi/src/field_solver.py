import numpy as np

class FieldSolver:
    """
    1D Finite Difference Solver for Poisson's Equation.
    d^2V/dx^2 = -rho/epsilon
    """
    def __init__(self, thickness_cm, voltage, dielectric_constant=11.0):
        self.d = thickness_cm
        self.V_bias = voltage
        self.epsilon_r = dielectric_constant
        self.epsilon_0 = 8.854e-14 # F/cm
        self.epsilon = self.epsilon_r * self.epsilon_0
        
        # Grid settings
        self.nz = 200
        self.z = np.linspace(0, self.d, self.nz)
        self.dz = self.d / (self.nz - 1)
        
    def solve(self, rho_profile=None):
        """
        Solve for Potential phi(z) and Electric Field E(z).
        rho_profile: array-like of size nz, space charge density (C/cm^3)
                     If None, assumes rho=0 (Linear Field).
        """
        if rho_profile is None:
            rho_profile = np.zeros(self.nz)
            
        # Discretization matrix A for d2V/dz2
        # V[i+1] - 2V[i] + V[i-1] = -rho[i]/eps * dz^2
        
        # We use a simple tridiagonal solver or numpy's linalg for simplicity (N is small)
        A = np.zeros((self.nz, self.nz))
        b = np.zeros(self.nz)
        
        # Internal nodes (1 to N-2)
        for i in range(1, self.nz - 1):
            A[i, i-1] = 1
            A[i, i]   = -2
            A[i, i+1] = 1
            b[i] = -rho_profile[i] / self.epsilon * (self.dz**2)
            
        # Boundary Conditions
        # V(0) = 0 (Cathode, typically ground in this convention, or anode is V_bias)
        # Let's assume Cathode at z=0 (0V) and Anode at z=d (V_bias)
        # Or typical detector: Anode pixel at z=0 (+V) collecting electrons?
        # Convention: Electrons move against E field. 
        # If we apply +V at Anode (z=d), E points from d to 0. Electrons move 0 -> d.
        # Let's stick to: z=0 is Cathode (0V), z=d is Anode (+V_bias).
        
        # V(0) = 0
        A[0, 0] = 1
        b[0] = 0
        
        # V(d) = V_bias
        A[-1, -1] = 1
        b[-1] = self.V_bias
        
        # Solve A * V = b
        V = np.linalg.solve(A, b)
        
        # Calculate Electric Field E = -dV/dz
        # Central difference
        E = np.zeros_like(V)
        E[1:-1] = -(V[2:] - V[:-2]) / (2 * self.dz)
        # Boundaries (forward/backward)
        E[0] = -(V[1] - V[0]) / self.dz
        E[-1] = -(V[-1] - V[-2]) / self.dz
        
        return self.z, V, E

    @staticmethod
    def get_trapped_charge_profile(z, flux_rate=1e8, trapping_time=1e-6):
        """
        Generate a hypothetical space charge profile due to trapping.
        Simplistic model: more trapping near cathode (electron injection) or anode?
        Usually uniform illumination -> exponential absorption -> exponential trap filling.
        rho(z) ~ exp(-mu*z)
        """
        # Example: Positive space charge buildup due to hole trapping
        q = 1.6e-19
        rho = q * flux_rate * trapping_time * np.exp(-2.0 * z) # Arbitrary decay
        return rho
