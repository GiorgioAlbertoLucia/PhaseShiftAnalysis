import numpy as np
from scipy.integrate import quad
from scipy.special import gamma, hyp1f1
from alive_progress import alive_bar

import mpmath as mp

from multiprocessing import Pool
from tqdm import tqdm

from ROOT import TH1F, TFile

from include.wave_function_core import Constants

class LednickyCoulombWavefunction:
    """
    Implementation of the Lednicky-Lyuboshitz approximation for 
    Coulomb-corrected wavefunctions and correlation function calculation.
    """
    
    def __init__(self, m1, m2, charge1, charge2, f0s, d0s, f0t, d0t, clebsch_gordan_coefficients=(0.25, 0.75), scattering_length=None):
        """
        Initialize the wavefunction parameters.
        
        Parameters:
        -----------
        m1, m2 : float
            Masses of the two particles (in GeV/c^2 or consistent units)
        charge1, charge2 : float
            Charges of the particles (in units of e)
        f0 : float
            S-wave scattering length (in fm)
        d0 : float
            Effective range (in fm)
        scattering_length : float, optional
            Alternative parameterization for scattering length
        """
        self.m1 = m1
        self.m2 = m2
        self.charge1 = charge1
        self.charge2 = charge2
        
        self.f0s = f0s
        self.d0s = d0s
        self.f0t = f0t
        self.d0t = d0t
        self.clebsh_gordan_coefficients = clebsch_gordan_coefficients
        
        self.mu = (m1 * m2) / (m1 + m2)
        
        self.alpha = 1/137.036
        self.hbarc = 197.3  # MeV·fm
        
    def bohr_radius(self):
        """
        Calculate the Coulomb Bohr radius of the pair.
        
        Parameters:
        -----------
            
        Returns:
        --------
        a_C : float or array
            Bohr radius in fm
        """

        charge_product = abs(self.charge1 * self.charge2)
        if charge_product == 0:
            return np.inf

        return self.hbarc / (self.alpha * charge_product * self.mu)
    
    def eta(self, k):
        """
        Calculate the Sommerfeld parameter η = (k*a_C)^(-1).
        
        Parameters:
        -----------
        k : float or array
            Relative momentum
            
        Returns:
        --------
        eta : float or array
            Sommerfeld parameter
        """
        a_C = self.bohr_radius()
        return 1.0 / (k * a_C)
    
    def gamow_factor(self, eta):
        """
        Calculate the Gamow (Coulomb) penetration factor A_C(η).
        
        Parameters:
        -----------
        eta : float or array
            Sommerfeld parameter
            
        Returns:
        --------
        A_C : float or array
            Gamow factor
        """
        if np.any(np.abs(eta) < 1e-10):
            return np.ones_like(eta)
        return 2 * np.pi * eta / (np.exp(2 * np.pi * eta) - 1)
    
    def coulomb_function_F(self, eta, xi):
        """
        For the Lednicky formula, we need F(-iη, 1, iξ) which is
        the confluent hypergeometric function.
        
        Parameters:
        -----------
        eta : float or array
            Sommerfeld parameter
        xi : float or array
            ρ = k*r*
            ξ = ρ(1  costheta*)
            
        Returns:
        --------
        F : complex or array
            Coulomb function value
        """
        eta = np.atleast_1d(eta)
        xi = np.atleast_1d(xi)
        
        # F(-iη, 1, iξ) where ξ = ρ
        # Using confluent hypergeometric function 1F1
        
        result = np.zeros(eta.shape, dtype=complex)
        for i in range(len(eta)):
            result[i] = mp.hyp1f1(- 1j*eta[i], 1, 1j * xi[i])
            
        return result if len(result) > 1 else result[0]
    
    def coulomb_normalization(self, eta):
        """
        Coulomb wave function normalization constant C_L(η).
        For L=0: C_0(η) = 2^η / |Γ(1+iη)|
        """
        eta = np.atleast_1d(eta)
        result = np.zeros_like(eta)
        
        for i in range(len(eta)):
            gamma_val = gamma(1 + 1j * eta[i])
            result[i] = 2**eta[i] * np.exp(-np.pi * eta[i] / 2) / np.abs(gamma_val)
        
        return result if len(result) > 1 else result[0]
    
    @staticmethod
    def complex_quadrature(func, a, b, **kwargs):
        def real_func(x):
            return np.real(func(x))
        def imag_func(x):
            return np.imag(func(x))
        real_integral = quad(real_func, a, b, **kwargs)
        imag_integral = quad(imag_func, a, b, **kwargs)
        return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

    def RegularCoulomb(self, l, eta, rho):
        First = rho**(l+1)*2**l*np.exp(1j*rho-(np.pi*eta/2),dtype='complex_')/(abs(gamma(l+1+1j*eta)))
        integral = LednickyCoulombWavefunction.complex_quadrature(
            lambda t: np.exp(-2*1j*rho*t,dtype='complex_')*t**(l+1j*eta)*(1-t)**(l-1j*eta),0,1)[0]
        return np.array(First*integral,dtype='complex_')

    def h_function(self, eta):
        """
        Known function h(η) - related to Coulomb corrections.
        h(η) = Re[ψ(iη)] where ψ is the digamma function.
        
        Parameters:
        -----------
        eta : float or array
            Sommerfeld parameter
            
        Returns:
        --------
        h : float or array
        """
        from scipy.special import digamma
        
        eta = np.atleast_1d(eta)
        result = np.zeros_like(eta)
        
        for i in range(len(eta)):
            if np.abs(eta[i]) < 1e-10:
                result[i] = 0.0
            else:
                # h(η) = Re[ψ(iη)] where ψ is digamma function
                result[i] = np.real(digamma(1j * eta[i]))
        
        return result if len(result) > 1 else result[0]
    
    def scattering_amplitude_f_c(self, k, eta, state:str):
        """
        Calculate the strong scattering amplitude f_C(k*) including Coulomb.
        From equation (25).
        
        Parameters:
        -----------
        k : float or array
            Relative momentum
        eta : float or array
            Sommerfeld parameter
            
        Returns:
        --------
        f_c : complex array
            Scattering amplitude
        """

        if state == 'single':
            f0, d0 = self.f0s, self.d0s
        elif state == 'triplet':
            f0, d0 = self.f0s, self.d0s
        else:
            return 0

        a_C = self.bohr_radius()
        
        term1 = -1.0 / f0
        term2 = (d0 * k**2) / 2.0
        term3 = -1j * k * self.gamow_factor(eta)
        term4 = -2 * self.h_function(eta) / a_C
        
        return 1.0 / (term1 + term2 + term3 + term4)
    
    def G_function(self, rho, eta):
        """
        Function G̃(ρ,η) = √(F₀² + G₀²) combining regular (F₀) and 
        irregular (G₀) s-wave Coulomb functions.
        
        From the provided definition: G̃ = √(A_C(F₀² + G₀²))
        
        Parameters:
        -----------
        rho : float or array
            ρ = k*r*
        eta : float or array
            Sommerfeld parameter
            
        Returns:
        --------
        G_tilde : float or array
            Combined Coulomb function
        """
        eta = np.atleast_1d(eta)
        rho = np.atleast_1d(rho)
        
        result = np.zeros(rho.shape, dtype=complex)
        
        for i in range(len(rho)):
            # Get F₀ and G₀ from scipy (L=0 Coulomb functions)
            #F0 = self.RegularCoulomb(0, eta[i], rho[i])
            F0 = mp.coulombf(0, eta[i], rho[i])
            G0 = mp.coulombg(0, eta[i], rho[i])
            
            # Calculate G̃
            A_C = self.gamow_factor(eta[i])
            result[i] = np.sqrt(A_C) * (1j*F0 + G0)
        
        return result if len(result) > 1 else result[0]
    
    #def wavefunction(self, k_vec, r_vec):
    #    """
    #    Calculate the relative wavefunction ψ(k*, r*).
    #    From equation (24).
    #    
    #    Parameters:
    #    -----------
    #    k_vec : array (3,)
    #        Relative momentum vector
    #    r_vec : array (3,)
    #        Relative position vector
    #        
    #    Returns:
    #    --------
    #    psi : complex
    #        Wavefunction value
    #    """
    #    k = np.linalg.norm(k_vec)
    #    r = np.linalg.norm(r_vec)
    #    
    #    eta = self.eta(k)
    #    A_C = self.gamow_factor(eta)
    #    
    #    # Calculate ξ = ρ(1 + cos θ*) where θ* is angle between k* and r*
    #    cos_theta = np.dot(k_vec, r_vec) / (k * r) if k*r > 0 else 1.0
    #    rho = k * r
    #    xi = rho * (1 + cos_theta)
    #    
    #    # Scattering amplitude
    #    f_c = self.scattering_amplitude_f_c(k, eta)
    #    
    #    # Coulomb function
    #    F = self.coulomb_function_F(eta, xi)
    #    
    #    # G function
    #    G = self.G_function(rho, eta)
    #    
    #    # Phase factor
    #    phase = np.exp(1j * np.angle(gamma(1. + 1j*eta)))
    #    
    #    # Full wavefunction (equation 24)
    #    psi = phase * np.sqrt(A_C) * (np.exp(-1j * rho) * F + f_c * G / r)
    #    
    #    return psi
    #
    #def correlation_function(self, k, source_function, r_points, integration_method='trapz'):
    #    """
    #    Calculate correlation function C(k*) using Koonin-Pratt formula.
    #    C(k*) = ∫ d³r* S(r*)|ψ(r*,k*)|²
    #    
    #    Parameters:
    #    -----------
    #    k : float or array
    #        Relative momentum magnitude
    #    source_function : callable
    #        Source function S(r) as function of radius
    #    r_points : array
    #        Radial points for integration (in fm)
    #    integration_method : str
    #        'trapz' for trapezoidal, 'simpson' for Simpson's rule
    #        
    #    Returns:
    #    --------
    #    C : float or array
    #        Correlation function value(s)
    #    """
    #    k = np.atleast_1d(k)
    #    C = np.zeros_like(k)
    #    
    #    with alive_bar(len(k), title='Integration over k...') as bar:
    #        for i, k_val in enumerate(k):
    #            # k vector (choose along z-axis)
    #            k_vec = np.array([0., 0., k_val])
    #            
    #            integrand = np.zeros(len(r_points))
    #            
    #            for j, r in enumerate(r_points):
    #                # Integrate over angles (simplified: average over theta, phi)
    #                n_theta = 20
    #                n_phi = 20
    #                theta = np.linspace(0, np.pi, n_theta)
    #                phi = np.linspace(0, 2*np.pi, n_phi)
    #                
    #                psi_squared_avg = 0.0
    #                
    #                for th in theta:
    #                    for ph in phi:
    #                        r_vec = r * np.array([np.sin(th)*np.cos(ph), 
    #                                            np.sin(th)*np.sin(ph), 
    #                                            np.cos(th)])
    #                        psi = self.wavefunction(k_vec, r_vec)
    #                        psi_squared_avg += np.abs(psi)**2 * np.sin(th)
    #                
    #                psi_squared_avg /= (n_theta * n_phi) / (2 * np.pi)
    #                
    #                # d³r = r² dr dΩ, integrate over r with proper weighting
    #                integrand[j] = source_function(r) * psi_squared_avg * r**2
    #            
    #            # Integrate over r (4π included in dΩ integration above)
    #            if integration_method == 'trapz':
    #                C[i] = 4 * np.pi * np.trapz(integrand, r_points)
    #            else:
    #                # Simpson's rule would require scipy
    #                C[i] = 4 * np.pi * np.trapz(integrand, r_points)
    #            bar()
    #    
    #    return C if len(C) > 1 else C[0]

    def wavefunction_vectorized(self, k_vec, r_array):
        """
        Vectorized version - compute wavefunction for multiple r positions at once.
        
        Parameters:
        -----------
        k_vec : array (3,)
            Relative momentum vector
        r_array : array (N, 3)
            Array of N relative position vectors
            
        Returns:
        --------
        psi : array (N,) complex
            Wavefunction values
        """
        k = np.linalg.norm(k_vec)
        r = np.linalg.norm(r_array, axis=1)
        
        eta = self.eta(k)
        A_C = self.gamow_factor(eta)
        
        # Vectorized dot products
        cos_theta = np.dot(r_array, k_vec) / (k * r)
        cos_theta = np.where(k*r > 0, cos_theta, 1.0)
        
        rho = k * r
        xi = rho * (1 + cos_theta)
        
        # Scattering amplitude (same for all r)
        f_c_s = self.scattering_amplitude_f_c(k, eta, state='singlet')
        f_c_t = self.scattering_amplitude_f_c(k, eta, state='triplet')
        
        # Vectorized special functions
        F = np.array([complex(mp.hyp1f1(-1j*eta, 1, 1j*xi_i)) for xi_i in xi])
        G = np.array([complex(np.sqrt(A_C) * (1j*mp.coulombf(0, eta, rho_i) + mp.coulombg(0, eta, rho_i))) 
                    for rho_i in rho])
        
        phase = np.exp(1j * np.angle(gamma(1. + 1j*eta)))
        psi_s = phase * np.sqrt(A_C) * (np.exp(-1j * rho) * F + f_c_s * G / r)
        psi_t = phase * np.sqrt(A_C) * (np.exp(-1j * rho) * F + f_c_t * G / r)
        
        return psi_s, psi_t

    def correlation_function_optimized(self, k, source_function, r_points, n_theta=20, n_phi=20):
        """
        Optimized correlation function calculation with vectorization.
        """
        k = np.atleast_1d(k)
        C = np.zeros_like(k)
        C_s = np.zeros_like(k)
        C_t = np.zeros_like(k)
        
        # Pre-compute angular grid
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
        sin_theta = np.sin(theta_grid).ravel()
        
        # Angular integration weights (trapezoidal)
        dtheta = theta[1] - theta[0]
        dphi = phi[1] - phi[0]
        angular_weight = sin_theta * dtheta * dphi
        
        # Pre-compute all r_vec directions (unit vectors)
        x = np.sin(theta_grid) * np.cos(phi_grid)
        y = np.sin(theta_grid) * np.sin(phi_grid)
        z = np.cos(theta_grid)
        directions = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)  # (n_angles, 3)
        
        #with alive_bar(len(k), title='Integration over k...') as bar:
        for i, k_val in enumerate(k):
            k_vec = np.array([0., 0., k_val])
            
            integrand_s = np.zeros(len(r_points))
            integrand_t = np.zeros(len(r_points))
            
            for j, r in enumerate(r_points):
                # Create all r_vec for this radius
                r_vecs = r * directions  # (n_angles, 3)
                
                # Vectorized wavefunction calculation
                psi_s, psi_t = self.wavefunction_vectorized(k_vec, r_vecs)
                
                psi_s_squared = np.abs(psi_s)**2
                psi_s_squared_avg = np.sum(psi_s_squared * angular_weight) / (4 * np.pi)
                integrand_s[j] = source_function(r) * psi_s_squared_avg * r**2

                psi_t_squared = np.abs(psi_t)**2
                psi_t_squared_avg = np.sum(psi_t_squared * angular_weight) / (4 * np.pi)
                integrand_t[j] = source_function(r) * psi_t_squared_avg * r**2
                
            C_s[i] = 4 * np.pi * np.trapz(integrand_s, r_points)
            C_t[i] = 4 * np.pi * np.trapz(integrand_t, r_points)
            C[i] = self.clebsh_gordan_coefficients[0]*C_s[i] + self.clebsh_gordan_coefficients[1]*C_t[i]
            #bar()
        
        return C if len(C) > 1 else C[0]
    


def _compute_single_k(args):
    k_val, wf, R, r_points = args
    def source(r):
        return (1/(2 * R * np.sqrt(np.pi)))**3 * np.exp(-r**2/(2 * R)**2)
    C_val = wf.correlation_function_optimized(k_val, source, r_points,
                                              n_theta=10, n_phi=10)
    return C_val

# In main:

if __name__ == "__main__":
    
    # Scattering parameters (typical values)
    a0s_pd = -2.73  # fm - scattering length singlet
    r0s_pd = 2.27  # fm - effective range singlet

    a0t_pd = -11.88  # fm - scattering length triplet
    r0t_pd = 2.63  # fm - effective range triplet

    wf = LednickyCoulombWavefunction(Constants.M_PROTON, Constants.M_DEUTERON, 
                                     Constants.Z_PROTON, Constants.Z_DEUTERON,
                                     a0s_pd, r0s_pd, a0t_pd, r0t_pd,
                                     clebsch_gordan_coefficients=(1./3., 2./3.))
    
    outfile = TFile.Open('output/lednicky_integration.root', 'recreate')
    
    source = lambda r: (1/(2 * R * np.sqrt(np.pi)))**3 * np.exp(-r**2/(2 * R)**2)
    
    kmin, kmax, nbins = 1, 401, 200
    k_values = np.linspace(kmin, kmax, nbins)  # MeV/c (or 0.01-0.2 in 1/fm)
    k_values += 0.5 # center the bin centers in 1.5, ...
    
    k_values_fm = k_values / Constants.HBARC  # Convert MeV/c to 1/fm
    r_points = np.linspace(0.1, 50, 100)  # fm

    #R = 1.29  # fm (source size)
    for R in [1.059, 1.2, 2.0]:
        
        hist = TH1F(f'hCk_R{R}', f'R = {R} fm; k* (MeV/c); C(k*)', nbins, kmin, kmax)


        #C_k = wf.correlation_function(k_values_fm, source, r_points)
        #C_k = wf.correlation_function_optimized(k_values_fm, source, r_points)
        with Pool(processes=8) as pool:
            args_list = [(k_val, wf, R, r_points) for k_val in k_values_fm]
            #C_k = pool.map(_compute_single_k, args_list)
            C_k = list(tqdm(pool.imap_unordered(_compute_single_k, args_list),
                            total=len(args_list), desc=f'R = {R} fm'))

        for k, C in zip(k_values, C_k):
            hist.Fill(k, C)

        outfile.cd()
        hist.Write()
    
    outfile.Close()
    
    print("\nResults:")
    print("k (MeV/c) | C(k)")
    print("-" * 25)
    for k, C in zip(k_values, C_k):
        print(f"{k:8.1f}  | {C:8.4f}")