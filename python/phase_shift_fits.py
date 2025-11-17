'''
    Fit of the phase shift data using the effective range expansion.
'''

from ROOT import TF1, TFile, TH1F
import ROOT
from ctypes import c_double

import numpy as np
from torchic.core.graph import create_graph
from torchic.utils.root import set_root_object

# from 2.25 MeV to 5.55 MeV (4 data points) taken from https://doi.org/10.1103/PhysRevC.82.034002
# from 6.82 MeV to 10.77 MeV (3 data points) taken from https://doi.org/10.1103/PhysRevC.1.888

DATA = {
    'energy': 
    {
        '1S0': np.array([2.25, 3.15, 4.00, 5.55, 6.82, 
                         #8.82, 
                         10.77]),  # incident proton kinetic in MeV,
        '3S1': np.array([2.25, 3.15, 4.00, 5.55, 6.82, 8.82, 10.77]),  # incident proton kinetic in MeV,
    },
    'delta': # phase shifts in degrees
    {
        '1S0': np.array([-39.1, -48.7, -56.3, -67.8, -66.6, 
                         #-78.2, 
                         -90.0,]),
        '3S1': np.array([-34.5, -42.9, -49.3, -58.6, -67.6, -78.2, -87.2]),  # in degrees
        '1P1': np.array([8]),  # in degrees
        '3P0': np.array([5]),  # in degrees
        '3P1': np.array([17]),  # in degrees
        '3P2': np.array([16.5]),  # in degrees
    },
    'delta_err': # phase shifts in degrees
    {
        '1S0': np.array([1.7, 0.9, 0.6, 0.9, 0, 
                         #8.3, 
                         0]),   # in degrees
        '3S1': np.array([0.7, 0.09, 0.5, 0.3, 0, 1.4, 0]),  # in degrees
        '1P1': np.array([]),  # in degrees
        '3P0': np.array([]),  # in degrees
        '3P1': np.array([]),  # in degrees
        '3P2': np.array([]),  # in degrees
    }
}

# points 4.55, 5.51, 6.52, 7.51, 8.51, 9.51, 10.38, 11.45 MeV from https://doi.org/10.1016/0029-5582(64)90234-2
NEW_DATA = {
    'energy': {'1S0': np.array([4.55, 5.51, 6.52, 7.51, 8.51, 9.51, 10.38, 11.45]), },
    # paper
    #'delta': {'1S0': np.array([-57.3, -62.5, -68.0, -73.5, -79.0, -83.5, -88.5, -94.2]), }, 
    #'delta_err': {'1S0': np.array([0, 0, 0, 0, 0, 0, 0, 0]), }
    # my fits
    'delta': {'1S0': np.array([-57.1, -61.97, -66.33, -70.69, -74.45, -77.70, -80.20, -83.70]), }, 
    'delta_err': {'1S0': np.array([0.53, 0.5, 0.46, 0.41, 0.38, 0.36, 0.35, 0.33]), }
    
}
for data_variable, data in NEW_DATA.items():
    for ll in NEW_DATA[data_variable].keys():
        if ll in DATA[data_variable]:
            DATA[data_variable][ll] = [*DATA[data_variable][ll], *NEW_DATA[data_variable][ll]]
for ll in NEW_DATA['energy'].keys():
    DATA['energy'][ll], DATA['delta'][ll], DATA['delta_err'][ll] = zip(*sorted(zip(DATA['energy'][ll], DATA['delta'][ll], DATA['delta_err'][ll])))
    DATA['energy'][ll], DATA['delta'][ll], DATA['delta_err'][ll] = list(DATA['energy'][ll]), list(DATA['delta'][ll]), list(DATA['delta_err'][ll])


M_PROTON = 938.2720813  # MeV/c^2
M_HELIUM3 = 2808.391383  # MeV/c^2
M_REDUCED = (M_PROTON * M_HELIUM3) / (M_PROTON + M_HELIUM3)  # MeV/c^2
HBAR = 197.3269804  # MeV*fm

def convert_incident_momentum_to_kstar(momentum):

    incident_p_mu = ROOT.Math.PxPyPzMVector(0., 0., momentum, M_PROTON)
    target_p_mu = ROOT.Math.PxPyPzMVector(0., 0., 0., M_HELIUM3)

    p_total_beta_vector = (incident_p_mu + target_p_mu).BoostToCM()
    beta_px, beta_py, beta_pz = 0., 0., 0.
    p_total_beta_vector.GetCoordinates(c_double(beta_px), c_double(beta_py), c_double(beta_pz))
    p_total_boost = ROOT.Math.Boost(beta_px, beta_py, beta_pz)

    incident_p_mu_to_boost = ROOT.Math.PxPyPzMVector(0., 0., momentum, M_PROTON)
    target_p_mu_to_boost = ROOT.Math.PxPyPzMVector(0., 0., 0., M_HELIUM3)
    p1mu_star = p_total_boost(incident_p_mu_to_boost)
    p2mu_star = p_total_boost(target_p_mu_to_boost)

    kmu_star = (p1mu_star - p2mu_star)
    return 0.5 * kmu_star.P()

def vectorized_convert_incident_momentum_to_kstar(momentum_vector):
    kstar_vector = []
    for momentum in momentum_vector:
        kstar = convert_incident_momentum_to_kstar(momentum)
        kstar_vector.append(kstar)
    return kstar_vector

def fit_effective_range_expansion(outfile: TFile, spectroscopic_index: str):

    plab_data = [np.sqrt(2 * energy / M_PROTON)*M_PROTON for energy in DATA['energy'][spectroscopic_index]]
    kstar_data = vectorized_convert_incident_momentum_to_kstar(plab_data)
    delta_data = DATA['delta'][spectroscopic_index]
    energy_err = np.zeros_like(kstar_data)
    delta_err = DATA['delta_err'][spectroscopic_index]
    delta_graph = create_graph(list(kstar_data), list(delta_data), list(energy_err), list(delta_err), 
                               name=f'phase_shift_{spectroscopic_index}', title=';E (MeV);#delta(^{1}S_{0}) (degrees)')
    set_root_object(delta_graph, marker_style=20, marker_size=1.2, line_color=1)

    k_data = np.array(kstar_data) / HBAR  # in fm^-1
    k_cot_delta = 1 / np.tan(np.radians(delta_data)) * k_data  # in fm^-1
    k_cot_delta_err = np.abs(k_data * delta_err / np.sin(np.radians(delta_data))**2)  # in fm^-1
    k_cot_delta_graph = create_graph(list(k_data), list(k_cot_delta), list(np.zeros_like(k_data)), list(k_cot_delta_err), 
                                     name=f'k_cot_delta_{spectroscopic_index}', title=';#it{k} (fm^{-1});#it{k} cot(#delta) (fm^{-1})')
    set_root_object(k_cot_delta_graph, marker_style=21, marker_size=1.2, line_color=1)

    fit = TF1("fit", "1/[0] + 0.5*[1]*(x^2)", 0, 5) # Effective range expansion: k cot(delta) = 1/a_0 + 0.5 r_0 k^2
    fit.SetParNames("a_0", "r_0")
    fit.SetParameters(np.sin(np.radians(delta_data[0]))**2, 1.0)  # Initial guess for a_0, r_0
    k_cot_delta_graph.Fit(fit, "RMS+")

    outfile.cd()
    delta_graph.Write()
    k_cot_delta_graph.Write()
    fit.Write()

def fit_delta(outfile: TFile, spectroscopic_index: str):

    plab_data = [np.sqrt(2 * energy / M_PROTON)*M_PROTON for energy in DATA['energy'][spectroscopic_index]]
    kstar_data = vectorized_convert_incident_momentum_to_kstar(plab_data)
    k_data = np.array(kstar_data) / HBAR  # in fm^-1
    delta_data = DATA['delta'][spectroscopic_index]
    delta_err = DATA['delta_err'][spectroscopic_index]
    delta_graph = create_graph(list(k_data), list(np.radians(delta_data)), list(np.zeros_like(k_data)), list(np.radians(delta_err)), 
                               name=f'phase_shift_delta_{spectroscopic_index}', title=';k (fm^{-1});#delta(^{1}S_{0}) (radians)')
    set_root_object(delta_graph, marker_style=20, marker_size=1.2, line_color=1)

    fit = TF1("fit_delta", "TMath::ATan(x/(1/[0] + 0.5*[1]*(x^2)))") # Effective range expansion: delta = atan(k / (1/a_0 + 0.5 r_0 k^2))
    fit.SetParNames("a_0", "r_0")
    fit.SetParameters(np.sin(np.radians(delta_data[0]))**2, 1.0)  # Initial guess for a_0, r_0
    delta_graph.Fit(fit, "RMS+")

    print(f'{fit.GetChisquare()=}')
    print(f'{fit.GetNDF()=}')

    outfile.cd()
    delta_graph.Write()
    delta_graph.Write()
    fit.Write()

def save_phase_shift_histograms(outfile, spectroscopic_index):

    plab_data = [np.sqrt(2 * energy / M_PROTON)*M_PROTON for energy in DATA['energy'][spectroscopic_index]]
    kstar_data = vectorized_convert_incident_momentum_to_kstar(plab_data)
    bin_centers = kstar_data
    bin_edges = [(bin_centers[ibin+1] - bin_centers[ibin])/2 + bin_centers[ibin] for ibin in range(len(bin_centers)-1)]
    first_bin_edge = bin_centers[0] - (bin_edges[0] - bin_centers[0])
    bin_edges.insert(0, first_bin_edge)

    hist_phase_shift = TH1F('hPhaseShifts', ';#it{k}* (MeV);#delta (deg.)', len(bin_edges)-1, np.array(bin_edges, dtype=np.float64))
    for ibin in range(1, hist_phase_shift.GetNbinsX()+1):
        hist_phase_shift.SetBinContent(ibin, DATA['delta'][spectroscopic_index][ibin-1])
        hist_phase_shift.SetBinError(ibin, DATA['delta_err'][spectroscopic_index][ibin-1])

    delta_radians = np.radians(DATA['delta'][spectroscopic_index])
    delta_err_radians = np.radians(DATA['delta_err'][spectroscopic_index])
    hist_phase_shift_radians = TH1F('hPhaseShiftsRadians', ';#it{k}* (MeV);#delta (rad.)', len(bin_edges)-1, np.array(bin_edges, dtype=np.float64))
    for ibin in range(1, hist_phase_shift_radians.GetNbinsX()+1):
        hist_phase_shift_radians.SetBinContent(ibin, delta_radians[ibin-1])
        hist_phase_shift_radians.SetBinError(ibin, delta_err_radians[ibin-1])

    outfile.cd()
    hist_phase_shift.Write()
    hist_phase_shift_radians.Write()



if __name__ == "__main__":

    outfile = TFile("output/phase_shift_fits.root", "RECREATE")

    for spectroscopic_index in ['1S0', '3S1']:
        outdir = outfile.mkdir(spectroscopic_index)
        fit_effective_range_expansion(outdir, spectroscopic_index)
        fit_delta(outdir, spectroscopic_index)
        save_phase_shift_histograms(outdir, spectroscopic_index)

    outfile.Close()
