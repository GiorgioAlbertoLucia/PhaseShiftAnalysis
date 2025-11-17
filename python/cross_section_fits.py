import numpy as np
from scipy.constants import hbar, e, c, alpha
from ROOT import TFile, TF1, gInterpreter, TCanvas, TLatex

import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INCLUDE_DIR = os.path.join(CURRENT_DIR, 'include')
gInterpreter.ProcessLine(f'#include "{INCLUDE_DIR}/CrossSectionPartialWave.h"')
from ROOT import CrossSectionPartialWave

from torchic.core.graph import create_graph
from torchic.utils.root import set_root_object

# --- Data dictionary from the table ---
DATA = {
    'theta': [
        27.64, 31.97, 36.71, 44.85, 54.75, 63.44, 70.13, 76.28, 82.63, 90.00,
        96.03, 103.80, 110.55, 116.57, 125.27, 133.48, 140.79, 147.21, 153.90,
        162.14, 165.67, 166.59
    ],
    'cross-section': {  # mb
        4.55: [200.5, 198.2, 205.3, 185.3, 160.3, 134.3, 111.8, 96.00, 80.97, 70.85,
               63.36, 66.56, 74.24, 85.96, 111.1, 140.7, 168.7, 195.8, 220.8, 244.5,
               252.9, 263.5],
        5.51: [236.5, 243.2, 238.0, 217.1, 183.2, 147.9, 121.9, 100.7, 79.90, 63.40,
               54.71, 51.46, 56.05, 66.61, 88.57, 118.1, 141.9, 167.6, 196.1, 217.5,
               225.1, 232.6],
        6.52: [256.5, 261.8, 259.1, 230.6, 182.2, None, 121.3, 97.08, 77.33, 58.56,
               47.44, 41.63, 43.00, 48.58, 65.63, 91.51, 116.2, 138.6, 166.6, 188.7,
               196.4, 205.3],
        7.51: [258.7, 261.7, 256.9, 226.1, 185.2, 147.2, 118.6, 95.69, 72.70, 54.14,
               40.31, 33.33, 32.86, 35.86, 50.34, 72.02, 94.47, 120.2, 137.8, 158.7,
               165.2, 173.7],
        8.51: [255.6, 265.1, 250.9, 216.5, 179.8, 141.0, 112.0, 90.48, 68.21, 48.12,
               36.19, 26.57, 24.42, 27.42, 40.19, 57.44, 76.36, 95.23, 115.7, 139.6,
               143.5, 146.6],
        9.51: [243.6, 243.1, 240.3, None, 167.8, 132.0, 105.6, 84.46, 64.32, 44.19, 
               31.85, 23.02, 19.95, 21.48, 30.99, 46.71, 63.79, 82.46, 99.28, 119.9, 
               126.7, 132.7],
        10.38: [235.9, 237.8, 232.7, None, 159.1, None, 99.86, 80.48, 59.18, 42.64, 
                29.51, 19.42, 16.83, 17.67, 24.90, 39.81, 55.10, 72.15, 88.83, 104.5, 
                112.8, 115.7],
        11.48: [223.1, 222.0, 211.9, None, None, None, None, None, 54.27, 36.76, 
                25.70, 16.78, 13.21, 13.21, 20.26, 32.21, 45.95, 58.85, 75.46, 92.72, 
                97.70, 101.1]
    }
}

M_PROTON = 938.2720813  # MeV/c^2
M_HELIUM3 = 2808.391383  # MeV/c^2
M_REDUCED = (M_PROTON * M_HELIUM3) / (M_PROTON + M_HELIUM3)  # MeV/c^2
HBAR_C = 197.3269804  # MeV*fm
Z_PROTON = 1.
Z_HE3 = 2.

def CrossSectionPartialWaveDegree(xs, pars):
    xs[0] = np.radians(xs[0])
    return CrossSectionPartialWave(xs, pars)

def plot_cross_section(data_dict, output_file:TFile):
    
    thetas = np.array(data_dict['theta'])
    shifts, shifts_errors = [], []
    
    for energy, cross_section_values in data_dict['cross-section'].items():
        
        xs, ys, eys = [], [], []
        for theta, cross_section in zip(thetas, cross_section_values):
            if cross_section is None:
                continue
            ey = cross_section * 0.028 if (theta > 90 and theta < 135) else cross_section * 0.022
            xs.append(np.radians(theta))
            ys.append(cross_section / 1_000) # b/sr
            eys.append(ey / 1_000)

        n = len(xs)
        graph = create_graph(xs, ys, np.zeros(n), eys)
        set_root_object(graph, title=f'Energy = {energy} MeV; #theta (rad.); d#sigma/d#Omega (b/sr)',
                        marker_style=20, marker_color=797)


        # Theoretical curve
        v = np.sqrt(2*energy/M_PROTON)          # relative speed
        k = M_REDUCED * v / (HBAR_C)            # wave number
        eta = (Z_PROTON * Z_HE3 * alpha) / v    # Sommerfeld parameter
        nL = 3                                  # number of l indices to consider
        print(f'{energy=} MeV, {k=} fm^-1, {eta=}')
        npars = 3 + nL
        
        func = TF1(f"theory_{energy}", CrossSectionPartialWave, min(xs)-0.1, max(xs)+0.1, npars)
        func.FixParameter(0, k)
        func.FixParameter(1, eta)
        func.FixParameter(2, nL)
        set_values = (-np.radians(57.3), np.radians(32.2), -np.radians(1.2), -np.radians(1.2),  -np.radians(1.2))
        for par in range(3, npars):
            func.SetParameter(par, set_values[par-3])
            #func.SetParLimits(par, set_values[par-3]-0.1, set_values[par-3]+0.1)
        set_root_object(func, line_color=2, line_width=2)

        graph.Fit(func, 'RMS+')
        shifts_enegy = [np.rad2deg(func.GetParameter(3 + ishift)) for ishift in range(nL)]
        shifts_error_energy = [np.rad2deg(func.GetParError(3 + ishift)) for ishift in range(nL)]
        shifts.append(shifts_enegy)
        shifts_errors.append(shifts_error_energy)

        canvas = TCanvas(f'c_Energy{energy}MeV', '')
        graph.Draw('ap')
        func.Draw('same')
        tlatex = TLatex()
        tlatex.SetNDC()
        tlatex.DrawLatex(0.27, 0.85, f'#chi^{{2}} / NDF = {func.GetChisquare():.2f} / {func.GetNDF()}')
        for ishift, (shift, shift_error) in enumerate(zip(shifts_enegy, shifts_error_energy)):
            tlatex.DrawLatex(0.27, 0.85-0.05*(ishift+1), f'#delta_{{{ishift}}} = ({shift:.2f} #pm {shift_error:.2f})^{{#circ}}')

        
        output_file.cd()
        canvas.Write()

        del graph, canvas, func, shifts_enegy, shifts_error_energy

    return shifts, shifts_errors

def plot_shifts(data_dict, shifts, shifts_errors, output_file):

    shifts = np.asarray(shifts).T
    shifts_errors = np.asarray(shifts_errors).T
    energy = list(data_dict['cross-section'].keys())
    graphs = []

    canvas = TCanvas('c_Shifts', '')
    canvas.DrawFrame(4, -120, 12, 70, ';#it{p}_{lab} (MeV/#it{c});#delta (deg.)')

    for iL in range(3):
        graph = create_graph(energy, list(shifts[iL]), np.zeros(len(energy)), list(shifts_errors[iL]))
        set_root_object(graph, marker_style=20+iL, marker_color=2+iL, title=f'l={iL}')
        graph.Draw('p same')
        graphs.append(graph)

    legend = canvas.BuildLegend(0.55, 0.3, 0.65, 0.44)
    legend.SetBorderSize(0)

    output_file.cd()
    canvas.Write() 

    

if __name__ =='__main__':

    output_file = TFile.Open('output/cross_section_fits.root', 'recreate')
    fits_dir = output_file.mkdir('fits')
    shifts, shifts_errors = plot_cross_section(DATA, fits_dir)
    plot_shifts(DATA, shifts, shifts_errors, output_file)
    output_file.Close()
