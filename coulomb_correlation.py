import numpy as np
from mpmath import hyp1f1, gamma, exp, sqrt, quad, pi, exp, conj, factorial, arg, legendre, coulombf
from alive_progress import alive_bar

from ROOT import TH1F, TFile

from include.wave_function_core import Constants

## Cached variables


#def coulomb_wavefunction_sq(r, k, eta, cl:list, sigmal:list, l_max:int=30, costheta:float=1.):
#    """
#    |ψ_C(r)|^2 for s-wave Coulomb wavefunction.
#    Uses the exact confluent hypergeometric form.
#    """
#
#    psic = 0
#    psic_last = 0
#
#    for l in range(l_max):
#
#        #Fl = cl[l] * exp(1j * k * r) * (k*r)**(l+1) * hyp1f1(l + 1 + 1j*eta, 2.*l + 2., -2.*1j*k*r)
#        Fl = coulombf(l, eta, k*r)
#        psic += (2*l + 1.) * 1j**l * exp( 1j*sigmal[l] ) * Fl * legendre(l, costheta)
#
#        #error = (psic*conj(psic) - psic_last*conj(psic_last))/psic*conj(psic)
#        error = abs(psic - psic_last)/abs(psic)
#
#        if abs(error) < 1e-7:
#            break
#        #elif abs(error)  > 1e-7 and l == 49:
#        #    print(f'no convergence: {error=}, {l=}')
#
#        psic_last = psic
#
#    psic = psic / (k*r)
#
#    #z = 1j * k * r
#    #F = hyp1f1(-1j*eta, 1, z)
#    #psic = exp(-pi*eta/2.) * gamma(1 + 1j*eta) * exp(z) * F
#
#    return abs(psic * conj(psic))

def coulomb_wavefunction_sq(r, k, eta, cl:list, sigmal:list, l_max:int=30, costheta:float=1.):
    """
    |ψ_C(r)|^2 for s-wave Coulomb wavefunction.
    Uses the exact confluent hypergeometric form.
    """

    psic2 = 0
    psic2_last = 0

    for l in range(l_max):

        Fl = coulombf(l, eta, k*r)
        psic2 += (2*l + 1.) * Fl * conj(Fl)
        error = abs(psic2 - psic2_last)/abs(psic2)

        if abs(error) < 1e-7:
            break

        psic2_last = psic2

    return psic2 / (k*r)**2


def gaussian_source(r, R):
    """ 3D Gaussian source density (normalized). """
    return ((2*pi)**(-1.5) / (R**3)) * exp(-(r**2) / (2*R**2))


def integrand(r, k, eta, R_natural_units, cl:list, sigmal:list, l_max:int=30, costheta=1.):
    """ r^2 * S(r) * |ψ_C(r)|^2 """
    return r**2 * gaussian_source(r, R_natural_units) * coulomb_wavefunction_sq(r, k, eta, cl, sigmal, l_max, costheta)


def correlation_gaussian(k, eta, R_natural_units, cl:list, sigmal:list, l_max):
    """
    Coulomb correlation function for a Gaussian source.
    R in fm, masses in GeV, k in GeV.
    """


    #integral = quad(lambda r, costheta: integrand(r, k, mu, charge1, charge2, R_natural_units, costheta),
    #                [0, np.inf], [-1, 1])
    #Ck = 2 * pi * integral

    integral = quad(lambda r: integrand(r, k, eta, R_natural_units, cl, sigmal, l_max),
                    [0, np.inf])
    Ck = 4 * pi * integral

    return Ck


# Example usage
if __name__ == "__main__":
    
    R = 6.23            # fm
    R_natural_units = R / Constants.HBARC   # MeV^-1
    mu = (Constants.M_PROTON * Constants.M_HE3) / (Constants.M_PROTON + Constants.M_HE3)
    
    l_max = 50
    kmin, kmax, kstep = 1.5, 1000.5, 1 # MeV
    nbins = int((kmax-kmin)/kstep)
    ks = np.arange(kmin, kmax, kstep)

    h_Ck_Coulomb = TH1F('hist_Ck_Coulomb', 'Coulomb;#it{k}* (MeV/#it{c}); #it{C}(#it{k}*)',
                         nbins, kmin, kmax)
    
    cl, sigmal = np.zeros(l_max), np.zeros(l_max)
    
    with alive_bar(len(ks), title='Compute Ck for several values of k...') as bar:
        for ik, k in enumerate(ks):

            eta = Constants.ALPHA_EM * Constants.Z_PROTON * Constants.Z_HE3 * mu / k

            for l in range(l_max):
                #cl[l] = 2**l * exp(-pi*eta/2.) * abs(gamma(l + 1 + 1j*eta)) / factorial(2*l +1)
                sigmal[l] = arg( gamma(l + 1. + 1j*eta) )
                
            Ck = correlation_gaussian(k, eta, R_natural_units, cl, sigmal, l_max)
            h_Ck_Coulomb.SetBinContent(h_Ck_Coulomb.FindBin(k), Ck)
            bar()

    outfile = TFile('output/coulomb_python_costheta.root', 'recreate')
    h_Ck_Coulomb.Write()
    outfile.Close()
    