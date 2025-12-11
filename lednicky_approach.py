import numpy as np
import mpmath as mp
from alive_progress import alive_bar
from include.wave_function_core import psi0, phi0, psi, Constants

from ROOT import TFile, TH1F

def reduced_mass(m1, m2):
    """Reduced mass calculation"""
    return m1 * m2 / (m1 + m2)

def sommerfeld_parameter(k, mu, z1, z2):
    """Sommerfeld parameter calculation"""
    return Constants.ALPHA_EM * mu * z1 * z2 / k

def draw_wave_function(outfile:TFile):
    """Draw wave functions psi0 and phi0"""
    
    # Parameters
    k = 100  # MeV - relative momentum
    Rsource = 7  # fm - typical radius in central PbPb collisions
    mu = reduced_mass(Constants.M_PROTON, Constants.M_HE3)
    eta = sommerfeld_parameter(k, mu, Constants.Z_HE3, Constants.Z_PROTON)
    
    a0 = 11.1  # fm - scattering length
    a0_natural_units = a0 / Constants.HBARC
    r0 = 1.8  # fm - effective range
    r0_natural_units = r0 / Constants.HBARC
    
    step = 0.1
    nbins = int(Rsource / step)
    hist_psi0_real = TH1F("psi0_real", ";#it{r}* (fm); Re[#psi_{0}](#it{r})", nbins, 0., Rsource)
    hist_psi0_imag = TH1F("psi0_imag", ";#it{r}* (fm); Im[#psi_{0}](#it{r})", nbins, 0., Rsource)
    hist_phi0_real = TH1F("phi0_real", ";#it{r}* (fm); Re[#phi_{0}](#it{r})", nbins, 0., Rsource)
    hist_phi0_imag = TH1F("phi0_imag", ";#it{r}* (fm); Im[#phi_{0}](#it{r})", nbins, 0., Rsource)
        
    for iter, r in enumerate(np.arange(0.1, Rsource, step)):
        r_natural_units = r / Constants.HBARC
        
        psi0_value = psi0(r_natural_units, k, eta)  # pure Coulomb
        phi0_value = phi0(r_natural_units, k, eta, mu, a0_natural_units, r0_natural_units, 1.0)
        
        hist_psi0_real.SetBinContent(hist_psi0_real.FindBin(r), psi0_value.real)
        hist_psi0_imag.SetBinContent(hist_psi0_imag.FindBin(r), psi0_value.imag)
        hist_phi0_real.SetBinContent(hist_phi0_real.FindBin(r), phi0_value.real)
        hist_phi0_imag.SetBinContent(hist_phi0_imag.FindBin(r), phi0_value.imag)
    
    outfile.cd()
    for hist in [hist_psi0_real, hist_psi0_imag,
                 hist_phi0_real, hist_phi0_imag]:
        hist.Write()

def compute_correlation_function(outfile:TFile):

    Rsource = 7  # fm - typical radius in central PbPb collisions
    mu = reduced_mass(Constants.M_PROTON, Constants.M_HE3)
    
    a0 = 11.1  # fm - scattering length
    a0_natural_units = a0 / Constants.HBARC
    r0 = 1.8  # fm - effective range
    r0_natural_units = r0 / Constants.HBARC

    kmin, kmax, kstep = 1.5, 400.5, 1 # MeV
    Rsource = 7 # fm
    N_ITERATIONS = 100

    nbins = int((kmax-kmin)/kstep)
    h_Ck_strong = TH1F('hist_Ck_strong', 's-wave;#it{k}* (MeV/#it{c}); #it{C}(#it{k}*)',
                         nbins, kmin, kmax)
    h_Ck_Coulomb = TH1F('hist_Ck_Coulomb', 'Coulomb;#it{k}* (MeV/#it{c}); #it{C}(#it{k}*)',
                         nbins, kmin, kmax)
    
    with alive_bar(nbins, title='Computing correlation function...') as bar:
        for k in np.arange(kmin, kmax, kstep):

            eta = sommerfeld_parameter(k, mu, Constants.Z_HE3, Constants.Z_PROTON)
            radii = np.random.normal(0., 2*Rsource, N_ITERATIONS)
            actual_iterations = 0

            for iter in range(N_ITERATIONS):
                
                r = radii[iter]
                r_natural_units = r / Constants.HBARC
                if (r_natural_units < 1e-4):    # for too small radii, the wavefunction misbehaves
                    continue
                
                psi0_value = psi0(r_natural_units, k, eta)
                phi0_value = phi0(r_natural_units, k, eta, mu,
                                a0_natural_units, r0_natural_units)
                psi_value = psi(r_natural_units, k, eta)
                
                strong_modulus_squared = phi0_value*mp.conj(phi0_value) - psi0_value*mp.conj(psi0_value)
                Coulomb_modulus_squared = psi_value*mp.conj(psi_value)
                
                if mp.re(phi0_value*mp.conj(phi0_value)) > 1 or \
                   mp.re(psi0_value*mp.conj(psi0_value)) > 1 or \
                   mp.re(strong_modulus_squared) > 1:
                    print(f'{phi0_value*mp.conj(phi0_value)=}')
                    print(f'{psi0_value*mp.conj(psi0_value)=}')
                    print(f'{strong_modulus_squared=}')
                    print(f'{r=}, {r_natural_units=}')

                if (mp.im(strong_modulus_squared) > 1e-7):
                    print(f'{mp.im(strong_modulus_squared)=}')
                
                h_Ck_strong.Fill(k, mp.re(strong_modulus_squared))
                h_Ck_Coulomb.Fill(k, mp.re(Coulomb_modulus_squared))
                actual_iterations += 1

            kbin = h_Ck_strong.FindBin(k)
            h_Ck_strong.SetBinContent(kbin, h_Ck_strong.GetBinContent(kbin)/actual_iterations)
            h_Ck_strong.SetBinError(kbin, h_Ck_strong.GetBinError(kbin)/actual_iterations)
            h_Ck_Coulomb.SetBinContent(kbin, h_Ck_Coulomb.GetBinContent(kbin)/actual_iterations)
            h_Ck_Coulomb.SetBinError(kbin, h_Ck_Coulomb.GetBinError(kbin)/actual_iterations)
            
            bar()

    outfile.cd()
    h_Ck_strong.Write()
    h_Ck_Coulomb.Write()

    
if __name__ == "__main__":
    
    outfile = TFile.Open("output/lednicky.root", "recreate")
    draw_wave_function(outfile)
    compute_correlation_function(outfile)
    outfile.Close()