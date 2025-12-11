import mpmath as mp
from dataclasses import dataclass

# Set precision (adjust as needed)
mp.dps = 50  # decimal places

@dataclass(frozen=True)
class _Constants:
    HBARC: float = 197.3269804          # MeV fm
    M_PROTON: float = 938.272           # MeV
    M_DEUTERON: float = 1875.61294257   # MeV
    M_HE3: float = 2808.391             # MeV
    E_CHARGE_SQ: float = 1.44
    Z_PROTON: float = 1.0
    Z_DEUTERON: float = 1.0
    Z_HE3: float = 2.0
    ALPHA_EM: float = 0.0072973525643
Constants = _Constants()

def u0(r, k, eta):
    """
    Coulomb radial solution
    @param r: radius, in natural units (MeV^-1)
    @param k: momentum (MeV)
    @param eta: Sommerfeld parameter
    """
    
    result = (mp.exp(-mp.pi*eta/2) * 
              mp.gamma(1 + 1j*eta) * 
              mp.exp(1j*k*r) * 
              k*r * 
              mp.hyp1f1(1 + 1j*eta, 2, -2*1j*k*r))
    
    return complex(result)

def ul(r,k, eta, l):
    """
    Coulomb radial solution (with quantum number l)
    @param r: radius, in natural units (MeV^-1)
    @param k: momentum (MeV)
    @param eta: Sommerfeld parameter
    """
    
    result = (2.**l * mp.exp(-mp.pi*eta/2) * mp.gamma(1 + l + 1j*eta) / mp.fac(2*l +1)
              * mp.exp(1j*k*r) * k*r * 
              mp.hyp1f1(1 + l + 1j*eta, 2 + l, -2*1j*k*r))
    
    return complex(result)

def psi0(r, k, eta) -> complex:
    """
    Coulomb solution for l=0
    @param r: radius, in natural units (MeV^-1)
    @param k: momentum (MeV)
    @param eta: Sommerfeld parameter
    """
    
    return complex(u0(r, k, eta) / (k*r))

def psi(r, k, eta) -> complex:
    """
    Coulomb solution (full sum)
    @param r: radius, in natural units (MeV^-1)
    @param k: momentum (MeV)
    @param eta: Sommerfeld parameter
    """

    psi_value, psi_previous = 0, 0
    for l in range(15):

        psi_value += (2*l + 1) * 1j**l * ul(r, k, eta, l) / (k*r) * mp.legendre(l, 0.)    # assume always parallel emission for 1d source

        if psi_value == 0:
            continue
        if mp.re((psi_value - psi_previous)/psi_value) < 1e-7:
            break
        if l == 14:
            print(f'convergence not reached. {(psi_value - psi_previous)/psi_value=}')
        psi_previous = psi_value

    return complex(psi_value)

def Coulomb_penetration_factor(eta):
    """Coulomb penetration factor"""
    
    return 2*mp.pi*eta / (mp.exp(2*mp.pi*eta) - 1)

def h_lambda(eta, lambda_val):
    """Helper function for H_lambda calculation"""
    
    #h = 0
    #h_previous = 0
    #
    #for n in range(1, 100):
    #    
    #    h += eta**2 / (n**2 + eta**2)
    #    if h > 0 and (h - h_previous)/h < 1e-7:
    #        break  # convergence reached
    #    if n == 14 and (h - h_previous)/h > 1e-7:
    #        print(f'convergence not reached: {(h - h_previous)/h=}')
    #    
    #    h_previous = h
    #
    #h = h - mp.log(lambda_val*eta) - mp.euler
    #
    #return float(h)

    return 1/(12*eta**2) + 1/(120*eta**4)

def H_lambda(eta, lambda_val):
    """
    @param eta: Sommerfeld parameter
    @param lambda_val: can be ±1
    """

    return h_lambda(eta, lambda_val) + 1j*Coulomb_penetration_factor(eta)/(2*eta)

def scattering_amplitude(k, eta, mu, a0, r0, lambda_val=1.0):
    """
    Scattering amplitude expressed with the effective range approximation
    @param k: momentum (MeV)
    @param eta: Sommerfeld parameter
    @param mu: reduced mass of the system (MeV)
    @param a0: scattering length in natural units (MeV^-1)
    @param r0: effective range in natural units (MeV^-1)
    @param lambda_val: can be ±1
    """

    H = H_lambda(eta, lambda_val)
    inverse_f = -1/a0 + 0.5*r0*k**2 - 2*lambda_val*Constants.ALPHA_EM*mu*H
    return 1 / inverse_f

def T_matrix_element(k, eta, mu, a0, r0, lambda_val=1.0):
    """T-matrix element"""

    f_SC = scattering_amplitude(k, eta, mu, a0, r0, lambda_val)
    Ac = Coulomb_penetration_factor(eta)
    e_2isigma0 = mp.gamma(1 + 1j*eta) / mp.gamma(1 - 1j*eta)
    
    result = -2*mp.pi/mu * Ac * e_2isigma0 * f_SC
    return complex(result)

def Coulomb_propagator(r, k, eta, mu):
    """
    G_C - Coulomb propagator
    """
    
    result = (1j*mu*k/mp.pi * 
              mp.exp(1j*k*r) * 
              mp.gamma(1 + 1j*eta) * 
              mp.hyperu(1 + 1j*eta, 2, -2*1j*k*r))
    
    return complex(result)

def phi0(r, k, eta, mu, a0, r0, lambda_val=1.0):
    """Full wavefunction"""
    
    psi0_value = psi0(r, k, eta)
    T_SC = T_matrix_element(k, eta, mu, a0, r0, lambda_val)
    G_C = Coulomb_propagator(r, k, eta, mu)
    
    result = psi0_value + T_SC * G_C / \
            (mp.exp(-mp.pi*eta/2)*mp.gamma(1 + 1j*eta))
    
    return complex(result)


# Example usage:
if __name__ == "__main__":
    # Test with some values
    r = 1.0
    k = 100.0
    eta = 0.5
    mu = 500.0
    a0 = 1.0
    r0 = 0.5
    
    print(f"psi0(r={r}, k={k}, eta={eta}) = {psi0(r, k, eta)}")
    print(f"phi0(r={r}, k={k}, eta={eta}, mu={mu}, a0={a0}, r0={r0}) = {phi0(r, k, eta, mu, a0, r0)}")