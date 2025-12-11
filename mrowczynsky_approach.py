
import sympy as sm
import numpy as np
from scipy.special import erf
from scipy.integrate import quad
import mpmath as mp

from ROOT import TH1F, TFile, TCanvas, TLegend, gStyle, \
                 kOrange, kBlue, kGreen, kRed, kViolet, kCyan, kMagenta, kTeal, kGray

from include.wave_function_core import Constants
from torchic.utils.root import set_root_object
from torchic.core.histogram import scale_hist_axis

def evaluate_integral():

    r, R, k = sm.symbols('r R k', positive=True, real=True)
    
    result = sm.integrate(sm.exp(-3*r**2/(8*R**2)) * sm.sin(2*k*r), (r, 0, sm.oo))
    
    print("Symbolic result:")
    print(result)
    
    R_values = np.linspace(0.5, 3.0, 10)
    k_values = np.linspace(10, 150, 10)

    for R_val, k_val in zip(R_values, k_values):
        R_val_natural_units = R_val / Constants.HBARC
        num_result = result.subs({R: R_val_natural_units, k: k_val}).evalf()
        print(f"R={R_val_natural_units:.2f}, k={k_val:.2f}: {num_result=}, {type(num_result)=}")

def sommerfeld_parameter(k, mu, z1, z2):
    """Sommerfeld parameter calculation"""

    return Constants.ALPHA_EM * mu * z1 * z2 / k

def reduced_mass(m1, m2):
    """Reduced mass calculation"""
    
    return m1 * m2 / (m1 + m2)

def Coulomb_penetration_factor(eta):
    """Coulomb penetration factor"""
    
    return 2*np.pi*eta / (np.exp(2*np.pi*eta) - 1)

def correlation_function_strong(Rsource, a0, k):

    first_term = 3./(4 * Rsource**2) * a0**2 / ( 1 + k**2 * a0**2 ) * np.exp(- 8 * k**2 * Rsource**2 / 3)
    second_term = (- 3./4 * 1j*a0 / (k * Rsource**2 * ( 1 + k**2 * a0**2 )) * np.exp(- 8 * k**2 * Rsource**2 / 3) 
                   * mp.erf(2. * np.sqrt(6) * 1j * Rsource * k/ 3))

    return 1. + first_term + second_term

def correlation_function_CATS(Rsource, a0, r0, k, eta):

    Ac = Coulomb_penetration_factor(eta)
    h_eta = 0.5 * (mp.digamma(1j*eta) + mp.digamma(-1j*eta) - np.log(eta*eta))
    inv_f = -1./a0 + 0.5*r0*k**2 - 1j*Ac*k - 2.*k*eta*h_eta
    f = 1./inv_f
    Dawson_arg = 2*k*Rsource
    Dawson = mp.quad(lambda x: mp.exp(-Dawson_arg**2+x**2), [0, Dawson_arg])

    Ck = Ac * ( 1 + 0.5*f*mp.conj(f)/(Rsource*Rsource) * (1 - r0/(2*mp.pi*Rsource) + 0.5*(Ac - 1)**2*(1. - mp.exp(-4*k*k*Rsource*Rsource)) )
               +  mp.re(f)*Dawson/(mp.sqrt(mp.pi)*k*Rsource*Rsource) - mp.im(f)*(1 - mp.exp(-Dawson_arg*Dawson_arg))/(2.*k*Rsource*Rsource)
               + 2.*(Ac - 1.)*mp.cos(k*Rsource)*mp.exp(-k*k*Rsource*Rsource) )
    
    return Ck

def correlation_function_paper(Rsource, a0, r0, k, eta):

    Ac = Coulomb_penetration_factor(eta)
    h_eta = 0.5 * (mp.digamma(1j*eta) + mp.digamma(-1j*eta) - np.log(eta*eta))
    inv_f = -1./a0 + 0.5*r0*k**2 - 1j*Ac*k - 2.*k*eta*h_eta
    f = 1./inv_f
    Dawson_arg = 2*k*Rsource
    Dawson = mp.quad(lambda x: mp.exp(-Dawson_arg**2+x**2), [0, Dawson_arg])

    Ck = Ac * ( 1 + 0.5*f*mp.conj(f)/(Rsource*Rsource) * (1 + 0.5*(Ac**2 - 1)*(1. - mp.exp(-4*k*k*Rsource*Rsource)) )
               +  mp.re(f)*Dawson/(mp.sqrt(mp.pi)*k*Rsource*Rsource) - mp.im(f)*(1 - mp.exp(-Dawson_arg*Dawson_arg))/(2.*k*Rsource*Rsource) )
    
    return Ck



def plot_strong_interaction_Coulomb(outfile:TFile, mass1, mass2, z1, z2, a0s, a0t, clebsch_gordan_coef=(0.25, 0.75)):

    mu = reduced_mass(mass1, mass2)

    a0s_natural_units = a0s / Constants.HBARC
    a0t_natural_units = a0t / Constants.HBARC

    kmin, kmax, kstep = 1.5, 1000.5, 1 # MeV
    nbins = int((kmax-kmin)/kstep)

    h_Ck_strong = TH1F('hist_Ck_strong', 's-wave;#it{k}* (MeV/#it{c}); #it{C}(#it{k}*)',
                         nbins, kmin, kmax)
    h_Cks_strong = TH1F('hist_Cks_strong', 's-wave - singlet;#it{k}* (MeV/#it{c}); #it{C}(#it{k}*)',
                         nbins, kmin, kmax)
    h_Ckt_strong = TH1F('hist_Ckt_strong', 's-wave - triplet;#it{k}* (MeV/#it{c}); #it{C}(#it{k}*)',
                         nbins, kmin, kmax)
    h_Ck_Coulomb = TH1F('hist_Ck_Coulomb', 'Coulomb;#it{k}* (MeV/#it{c}); #it{C}(#it{k}*)',
                         nbins, kmin, kmax)
    h_Ck_Coulomb_swave = TH1F('hist_Ck_Coulomb_swave', 'Coulomb+s-wave;#it{k}* (MeV/#it{c}); #it{C}(#it{k}*)',
                         nbins, kmin, kmax)
    h_Ck_Coulomb_swave_CATS = TH1F('hist_Ck_Coulomb_swave_CATS', 'Coulomb+s-wave;#it{k}* (MeV/#it{c}); #it{C}(#it{k}*)',
                         nbins, kmin, kmax)
    h_Ck_Coulomb_swave_paper = TH1F('hist_Ck_Coulomb_swave_paper', 'Coulomb+s-wave;#it{k}* (MeV/#it{c}); #it{C}(#it{k}*)',
                         nbins, kmin, kmax)
    
    for radius in [1.5, 1.73, 2., 2.5, 3., 5.08, 6.23, 7, 7.19]:

        radius_natural_units = radius / Constants.HBARC

        for k in np.arange(kmin, kmax, kstep):

            eta = sommerfeld_parameter(k, mu, z1, z2)
            Ac = Coulomb_penetration_factor(eta)

            Cks = correlation_function_strong(radius_natural_units, a0s_natural_units, k)
            Ckt = correlation_function_strong(radius_natural_units, a0t_natural_units, k)
            Ck = clebsch_gordan_coef[0] * Cks + clebsch_gordan_coef[1] * Ckt
            
            h_Cks_strong.SetBinContent(h_Ck_strong.FindBin(k), mp.re(Cks))
            h_Ckt_strong.SetBinContent(h_Ck_strong.FindBin(k), mp.re(Ckt))
            h_Ck_strong.SetBinContent(h_Ck_strong.FindBin(k), mp.re(Ck))
            h_Ck_Coulomb.SetBinContent(h_Ck_Coulomb.FindBin(k), Ac)
            h_Ck_Coulomb_swave.SetBinContent(h_Ck_Coulomb_swave.FindBin(k), Ac*mp.re(Ck))

            Cks_CATS = correlation_function_CATS(radius_natural_units, a0s_natural_units, 0., k, eta)
            Ckt_CATS = correlation_function_CATS(radius_natural_units, a0t_natural_units, 0., k, eta)
            Ck_CATS = clebsch_gordan_coef[0] * Cks_CATS + clebsch_gordan_coef[1] * Ckt_CATS
            h_Ck_Coulomb_swave_CATS.SetBinContent(h_Ck_Coulomb_swave_CATS.FindBin(k), mp.re(Ck_CATS))

            Cks_paper = correlation_function_paper(radius_natural_units, a0s_natural_units, 0., k, eta)
            Ckt_paper = correlation_function_paper(radius_natural_units, a0t_natural_units, 0., k, eta)
            Ck_paper = clebsch_gordan_coef[0] * Cks_paper + clebsch_gordan_coef[1] * Ckt_paper
            h_Ck_Coulomb_swave_paper.SetBinContent(h_Ck_Coulomb_swave_paper.FindBin(k), mp.re(Ck_paper))

        outdir = outfile.mkdir(f'r={radius}_fm')
        outdir.cd()
        for hist in [h_Cks_strong, h_Ckt_strong, h_Ck_strong,
                    h_Ck_Coulomb, h_Ck_Coulomb_swave, h_Ck_Coulomb_swave_CATS,
                    h_Ck_Coulomb_swave_paper]:

            hist.Write()
            hist_GeV = scale_hist_axis(hist, 1_000, name=f'{hist.GetName()}_GeV', title=';#it{k}* (GeV/#it{c}); #it{C}(#it{k}*)')
            hist_GeV.Write()

            hist.Reset()

def draw_strong_interaction_Coulomb(outfile:TFile, pdf_path:str, suffix:str='', approach:str='', **kwargs):

    required_name = f'hist_Ck_Coulomb_swave{approach}'

    canvas = TCanvas('strong', 'Coulomb + s-wave; #it{k}* (GeV/#it{c}); C(#it{k}*)')
    canvas.DrawFrame(0., 0., 150, kwargs.get('ymax', 3.6), 'Coulomb + s-wave; #it{k}* (GeV/#it{c}); C(#it{k}*)')
    hists = {}
    colors = [kOrange-3, kBlue-2, kGreen+2, kRed+1, kViolet+1, kCyan+3, kMagenta, 4]
    
    for ikey, key in enumerate(outfile.GetListOfKeys()):

        #if ikey == 0 and approach == '':
        #    hists['Coulomb'] = outfile.Get(f'{key.GetName()}/hist_Ck_Coulomb')
        #    set_root_object(hists['Coulomb'], name=f'Coulomb-only', line_color=colors[-1], line_width=2)

        r = key.GetName()[2:-3]

        if r == '1.73' or r == '7.19' or r == '5.08':
            continue

        hists[r] = outfile.Get(f'{key.GetName()}/{required_name}')
        set_root_object(hists[r], name=f'r = {r} fm', line_color=colors[ikey], line_width=2)

    legend = TLegend(*kwargs.get('legend_position', (0.4, 0.4, 0.8, 0.7)))
    legend.SetFillColor(0)
    legend.SetBorderSize(0)
    legend.SetNColumns(2)
    legend.SetTextSize(0.04)
    
    for ihist, hist in enumerate(hists.values()):
        legend.AddEntry(hist, hist.GetName(), 'l')
        hist.Draw('l same')

    legend.Draw()
    canvas.SaveAs(f'{pdf_path}/Coulomb_swave{suffix}{approach}.pdf')


def compare_strong_interaction_Coulomb(outfile:TFile, pdf_path:str,  **kwargs):

    canvas = TCanvas('strong', 'Coulomb + s-wave; #it{k}* (GeV/#it{c}); C(#it{k}*)')
    canvas.DrawFrame(0., 0., 400, kwargs.get('ymax', 3.6), 'Coulomb + s-wave; #it{k}* (GeV/#it{c}); C(#it{k}*)')
    hists = {}
    colors = [kOrange-3, kBlue-2, kGreen+2, kRed+1, kViolet+1, kCyan+3, kMagenta, kTeal+2]
    
    for ihist, (title, name) in enumerate({ 'Mrowczynski': 'hist_Ck_Coulomb_swave',
                                            'CATS': 'hist_Ck_Coulomb_swave_CATS',
                                            'Torres-Rincon': 'hist_Ck_Coulomb_swave_paper'}.items()):
        hists[title] = outfile.Get(f'r=6.23_fm/{name}')
        set_root_object(hists[title], name=name, line_color=colors[ihist], line_width=2)

    hists['Mrowczynski (#it{R}_{s} = 5.08 fm)'] = outfile.Get(f'r=5.08_fm/hist_Ck_Coulomb_swave')
    set_root_object(hists['Mrowczynski (#it{R}_{s} = 5.08 fm)'], line_color=colors[len(hists)-1], line_width=2)

    coulomb_file = TFile.Open('/Users/glucia/Projects/CATS/phemto/output/CATS_CF_LS_6p23fm.root')
    hists['Coulomb-only'] = coulomb_file.Get(f'hHe3_p_Coul_CF')
    set_root_object(hists['Coulomb-only'], line_color=colors[len(hists)-1], line_width=2)

    gamow_file = TFile.Open('/Users/glucia/Projects/CATS/phemto/output/CATS_phe_radii.root')
    hists['Gamow factor'] = gamow_file.Get(f'hCF_GamowFactor')
    set_root_object(hists['Gamow factor'], line_color=kGray+2, line_width=2)

    legend = TLegend(*kwargs.get('legend_position', (0.35, 0.3, 0.85, 0.6)))
    legend.SetFillColor(0)
    legend.SetBorderSize(0)
    legend.SetNColumns(2)
    legend.SetTextSize(0.04)
    
    for title, hist in hists.items():
        legend.AddEntry(hist, title, 'l')
        hist.Draw('l same')

    legend.Draw()
    canvas.SaveAs(f'{pdf_path}/compare_Coulomb_swave.pdf')


if __name__ == '__main__':

    #evaluate_integral()

    gStyle.SetOptStat(0)
    do_computation = True

    # values taken from https://arxiv.org/abs/2001.11351
    a0s_phe3 = 11.1  # fm - scattering length singlet
    a0t_phe3 = 9.05  # fm - scattering length triplet

    # values taken from the NLO calculations of https://arxiv.org/pdf/2507.16250
    #a0s_phe3 = 11.26  # fm - scattering length singlet
    #a0t_phe3 = 9.06  # fm - scattering length triplet

    if do_computation:
        outfile = TFile.Open('output/mrowczynsky_approach_pHe3.root', 'recreate')
        plot_strong_interaction_Coulomb(outfile, Constants.M_PROTON, Constants.M_HE3, 
                                        Constants.Z_PROTON, Constants.Z_HE3, a0s_phe3, a0t_phe3,
                                        clebsch_gordan_coef=(0.25, 0.75))
    else:
        outfile = TFile.Open('output/mrowczynsky_approach_pHe3.root')

    draw_strong_interaction_Coulomb(outfile, 'output', '_pHe3', legend_position=(0.55, 0.5, 0.85, 0.8))
    draw_strong_interaction_Coulomb(outfile, 'output', '_pHe3', '_CATS', ymax=1.05, legend_position=(0.55, 0.2, 0.85, 0.45))
    draw_strong_interaction_Coulomb(outfile, 'output', '_pHe3', '_paper', ymax=1.1, legend_position=(0.55, 0.2, 0.85, 0.45))
    compare_strong_interaction_Coulomb(outfile, 'output', ymax=1.05)


    outfile.Close()
    exit(0)

    ########### proton - deuteron


    # values taken from 
    a0s_pd = -0.13  # fm - scattering length singlet
    a0t_pd = 14.7  # fm - scattering length triplet

    a0s_pd = -1.3  # fm - scattering length singlet
    a0t_pd = -11.4  # fm - scattering length triplet

    if do_computation:
        outfile = TFile.Open('output/mrowczynsky_approach_pd.root', 'recreate')
        plot_strong_interaction_Coulomb(outfile, Constants.M_PROTON, Constants.M_DEUTERON, 
                                        Constants.Z_PROTON, Constants.Z_DEUTERON, a0s_pd, a0t_pd,
                                        clebsch_gordan_coef=(1./3., 2./3.))
    else:
        outfile = TFile.Open('output/mrowczynsky_approach_pd.root')

    draw_strong_interaction_Coulomb(outfile, 'output', '_pd', ymax=9)
    draw_strong_interaction_Coulomb(outfile, 'output', '_pd', '_CATS', ymax=3)
    draw_strong_interaction_Coulomb(outfile, 'output', '_pd', '_paper', ymax=3.3)

    outfile.Close()
    