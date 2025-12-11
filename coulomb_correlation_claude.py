import numpy as np
from mpmath import gamma, exp, arg, coulombf
from alive_progress import alive_bar
from scipy.integrate import quad
from scipy.special import legendre
from multiprocessing import Pool, cpu_count

from ROOT import TH1F, TFile

from include.wave_function_core import Constants


def precompute_sigmal(eta, l_max):
    """Precompute all sigma_l values for a given eta."""
    return np.array([float(arg(gamma(l + 1. + 1j*eta))) for l in range(l_max)])


def coulomb_wavefunction_sq(r, k, eta, sigmal, l_max=30, use_angular=True):
    """
    |ψ_C(r, θ)|^2 using the proper expansion in spherical harmonics.
    
    If use_angular=False, returns the spherically averaged |ψ_C(r)|^2.
    This uses the orthogonality of Legendre polynomials:
    ∫_{-1}^{1} P_l(cosθ) P_{l'}(cosθ) d(cosθ) = 2/(2l+1) δ_{ll'}
    """
    kr = k * r
    
    if not use_angular:
        # Spherically averaged: sum over l of |F_l|^2 weighted by (2l+1)
        psic2_sum = 0.0
        for l in range(l_max):
            Fl = float(coulombf(l, eta, kr))
            psic2_sum += (2*l + 1.) * Fl * Fl
            
            # Convergence check
            if l > 5 and abs(psic2_sum) > 0:
                Fl_sq = Fl * Fl
                contribution = (2*l + 1.) * Fl_sq
                if abs(contribution / psic2_sum) < 1e-7:
                    break
        
        return psic2_sum / (kr * kr)
    
    else:
        # Full angular dependence - returns a function of cosθ
        def psic2_angular(costheta):
            psic = 0.0 + 0.0j
            for l in range(l_max):
                Fl = float(coulombf(l, eta, kr))
                Pl = legendre(l)(costheta)
                psic += (2*l + 1.) * (1j)**l * exp(1j*sigmal[l]) * Fl * Pl
            
            psic = psic / kr
            return float(abs(psic)**2)
        
        return psic2_angular


def gaussian_source(r, R):
    """3D Gaussian source density (normalized)."""
    return (2*np.pi)**(-1.5) / (R**3) * np.exp(-(r**2) / (2*R**2))


def integrand_spherical_avg(r, k, eta, R_natural_units, sigmal, l_max):
    """
    Integrand using spherically averaged Coulomb wavefunction.
    This is faster and appropriate if the source is spherically symmetric.
    """
    return r**2 * gaussian_source(r, R_natural_units) * \
           coulomb_wavefunction_sq(r, k, eta, sigmal, l_max, use_angular=False)


def integrand_full_angular(r, costheta, k, eta, R_natural_units, sigmal, l_max):
    """
    Full 2D integrand over r and cosθ.
    More accurate but slower.
    """
    psic2_func = coulomb_wavefunction_sq(r, k, eta, sigmal, l_max, use_angular=True)
    psic2 = psic2_func(costheta)
    return r**2 * gaussian_source(r, R_natural_units) * psic2


def correlation_gaussian_spherical(k, eta, R_natural_units, sigmal, l_max, r_max=None):
    """
    Coulomb correlation using spherically averaged wavefunction.
    Fastest method for spherically symmetric sources.
    """
    if r_max is None:
        r_max = 10 * R_natural_units * Constants.HBARC  # 10σ cutoff
    
    integral, error = quad(
        lambda r: integrand_spherical_avg(r, k, eta, R_natural_units, sigmal, l_max),
        0, r_max,
        limit=50,
        epsrel=1e-6
    )
    
    return 4 * np.pi * integral


def correlation_gaussian_full(k, eta, R_natural_units, sigmal, l_max, r_max=None):
    """
    Coulomb correlation with full angular integration.
    More accurate but slower.
    """
    if r_max is None:
        r_max = 10 * R_natural_units * Constants.HBARC
    
    # Nested integration: first over cosθ, then over r
    def integrand_r(r):
        integral_theta, _ = quad(
            lambda costheta: integrand_full_angular(r, costheta, k, eta, R_natural_units, sigmal, l_max),
            -1, 1,
            limit=30,
            epsrel=1e-5
        )
        return integral_theta
    
    integral, error = quad(integrand_r, 0, r_max, limit=50, epsrel=1e-6)
    
    return 2 * np.pi * integral  # 2π from φ integration


def compute_single_k(args):
    """Helper function for parallel processing."""
    k, eta, R_natural_units, sigmal, l_max, use_full_angular = args
    
    if use_full_angular:
        Ck = correlation_gaussian_full(k, eta, R_natural_units, sigmal, l_max)
    else:
        Ck = correlation_gaussian_spherical(k, eta, R_natural_units, sigmal, l_max)
    
    return k, Ck


def main(use_full_angular=False, use_parallel=True):
    """
    Main computation.
    
    Parameters:
    -----------
    use_full_angular : bool
        If True, use full angular integration (slower, more accurate)
        If False, use spherically averaged (faster, good for spherical sources)
    use_parallel : bool
        If True, use multiprocessing
    """
    R = 6.23  # fm
    R_natural_units = R / Constants.HBARC  # MeV^-1
    mu = (Constants.M_PROTON * Constants.M_HE3) / (Constants.M_PROTON + Constants.M_HE3)
    
    l_max = 30  # Reduced from 50 for speed
    kmin, kmax, kstep = 1.5, 1000.5, 1  # MeV
    nbins = int((kmax - kmin) / kstep)
    ks = np.arange(kmin, kmax, kstep)
    
    method = "full_angular" if use_full_angular else "spherical_avg"
    h_Ck_Coulomb = TH1F(
        f'hist_Ck_Coulomb_{method}',
        f'Coulomb ({method});#it{{k}}* (MeV/#it{{c}}); #it{{C}}(#it{{k}}*)',
        nbins, kmin, kmax
    )
    
    # Prepare arguments for computation
    args_list = []
    for k in ks:
        eta = Constants.ALPHA_EM * Constants.Z_PROTON * Constants.Z_HE3 * mu / k
        sigmal = precompute_sigmal(eta, l_max)
        args_list.append((k, eta, R_natural_units, sigmal, l_max, use_full_angular))
    
    if use_parallel:
        # Parallel computation
        n_processes = max(1, cpu_count() - 1)
        print(f"Using {n_processes} processes")
        
        with Pool(processes=n_processes) as pool:
            with alive_bar(len(ks), title=f'Computing Ck ({method})...') as bar:
                for k, Ck in pool.imap_unordered(compute_single_k, args_list):
                    h_Ck_Coulomb.SetBinContent(h_Ck_Coulomb.FindBin(k), float(Ck))
                    bar()
    else:
        # Sequential computation
        with alive_bar(len(ks), title=f'Computing Ck ({method})...') as bar:
            for args in args_list:
                k, Ck = compute_single_k(args)
                h_Ck_Coulomb.SetBinContent(h_Ck_Coulomb.FindBin(k), float(Ck))
                bar()
    
    outfile = TFile(f'output/coulomb_python_{method}.root', 'recreate')
    h_Ck_Coulomb.Write()
    outfile.Close()
    
    print(f"\nOutput saved to: output/coulomb_python_{method}.root")


if __name__ == "__main__":
    # Use spherically averaged method (fast) - appropriate for Gaussian source
    # main(use_full_angular=False, use_parallel=True)
    
    # Uncomment to compare with full angular integration (slower but more accurate)
    main(use_full_angular=True, use_parallel=True)