import numpy as np
from itertools import product

from ROOT import TCanvas, TFile, TH1D

from torchic.core.graph import load_graph
from torchic.utils.colors import get_color
from torchic.utils.root import set_alice_global_style, set_root_object, init_legend

def graph_to_hist(graph):
    
    bin_centers = np.ascontiguousarray(graph.GetX())
    bin_contents = np.ascontiguousarray(graph.GetY())
    bin_errors = np.ascontiguousarray(graph.GetEY())
    
    n_bins = len(bin_centers)
    bin_edges = []
    for i in range(n_bins):
        if i == 0:
            bin_edges.append(bin_centers[i] - (bin_centers[i + 1] - bin_centers[i]) / 2)
        elif i == n_bins - 1:
            bin_edges.append(bin_centers[i] + (bin_centers[i] - bin_centers[i - 1]) / 2)
        else:
            bin_edges.append((bin_centers[i] + bin_centers[i + 1]) / 2)
    bin_edges.append(bin_centers[-1] + (bin_centers[-1] - bin_centers[-2]) / 2)
    hist = TH1D('hist', 'hist', n_bins, np.array(bin_edges, dtype=np.float64))
    hist.SetDirectory(0)  # Detach the histogram from any ROOT file
    for i in range(n_bins):
        hist.SetBinContent(i + 1, bin_contents[i])
        #hist.SetBinError(i + 1, bin_errors[i])
        hist.SetBinError(i + 1, 0.0)  # Set errors to zero if not available
    return hist

def scale_x_axis(hist, scale_factor):
    hist.SetTitle(hist.GetTitle() + f" ({hist.GetXaxis().GetTitle()} in GeV)")
    hist.GetXaxis().SetTitle(hist.GetXaxis().GetTitle() + " (GeV)")
    hist.GetXaxis().SetLimits(hist.GetXaxis().GetXmin() / scale_factor, hist.GetXaxis().GetXmax() / scale_factor)
    return hist

def convert_and_rebin_histogram(hist, suffix:str):
    '''
        Convert the x scale from MeV to GeV and rebin the histogram
    '''
    
    scale_factor = 1000.0  # MeV to GeV

    hist_scaled = hist.Clone(hist.GetName() + suffix)
    bins = np.array([hist_scaled.GetBinLowEdge(i)/scale_factor for i in range(1, hist_scaled.GetNbinsX() + 2)], dtype=np.float64)
    nbins = len(bins) - 1
    
    #scale_x_axis(hist_scaled, scale_factor=scale_factor)

    tmp_hist = TH1D("tmp_hist", "tmp_hist", nbins, bins)
    for i in range(1, nbins+1):
        tmp_hist.SetBinContent(i, hist.GetBinContent(i))
        tmp_hist.SetBinError(i, hist.GetBinError(i))

    tmp_hist.SetDirectory(0)  # Detach the histogram from any ROOT file
    tmp_hist.SetName(hist.GetName() + suffix)

    return tmp_hist

if __name__ == "__main__":

    set_alice_global_style()

    infile_path = '/home/galucia/PhaseShiftAnalysis/numerical_lednicky/pHe/output/pHe3_LL_radius_fit_with26.root'
    outfile = TFile('/home/galucia/PhaseShiftAnalysis/numerical_lednicky/pHe/output/pHe3_LL_radius_fit_with26_GeV.root', 'RECREATE')
    
    # Radii from parameterisation (mT scaling)
    ### radii = [4.33, 4.26, 4.41,
    ###          3.46, 3.43, 3.49] # fm --> Rs / sqrt(2)
    # Radii from Mrowczynski 
    ### radii = [5.34, 5.34 - 0.11, 5.34 + 0.10,
    ###          4.24, 4.24 - 0.06, 4.24 + 0.06] # fm --> Rs / sqrt(2)
    # Radii from fit (free radius)
    ### radii = [6.93, 6.93 - 0.88, 6.93 + 0.88,
    ###          6.35, 6.35 - 0.72, 6.35 + 0.72] # fm --> Rs / sqrt(2)
    # Radii from fit (free radius with 26 data)
    radii = [6.62, 6.62 - 0.79, 6.62 + 0.79,
             5.10, 5.10 - 0.54, 5.10 + 0.54] # fm --> Rs / sqrt(2)
    # Scan over a range of radii
    ### radii = np.arange(3.5, 8.1, 0.1) # fm --> Rs / sqrt(2)
    
    radii = radii/np.sqrt(2)  # Convert to Rs
    radii_str = [f'{radius:.2f}' for radius in radii]
    print(f'{radii_str=}')

    for radius in radii:
        # {n, n-e, n+e} variations; all (aRe, r) combinations are used (nominal first)
        aRe_variations = {'1s0': [-11.26, -11.30, -11.22], # fm
                          '3s1': [-9.06, -9.10, -9.02]     # fm
                          }
        r_variations = {'1s0': [1.65, 1.36, 1.94], # fm
                        '3s1': [1.36, 1.11, 1.61]  # fm
                        }
        combinations = {state: list(product(aRe_variations[state], r_variations[state])) for state in aRe_variations}
        aRe_dict = {state: [aRe for aRe, _ in combs] for state, combs in combinations.items()}
        r_dict = {state: [r for _, r in combs] for state, combs in combinations.items()}
        graphs = {}
        hists = {}
        hists_GeV = {}
        
        outdir = outfile.mkdir(f'Rs{radius:.2f}')
        
        for state in aRe_dict.keys():
            aRe_list = aRe_dict[state]
            r_list = r_dict[state]
            graphs[state] = []
            hists[state] = []
            hists_GeV[state] = []
            
            legend = init_legend(0.4, 0.2, 0.7, 0.4)
            
            for icomb, combination in enumerate(zip(aRe_list, r_list)):
                aRe, r = combination
                graph = load_graph(infile_path, f'g{radius:.2f}_aRe{aRe:.2f}_aIm0.00_r{r:.2f}')
                graphs[state].append(graph)
                
                hist = graph_to_hist(graph)
                set_root_object(hist, color=get_color(icomb), line_color=get_color(icomb), line_width=1,
                                name=f'h{radius:.2f}_aRe{aRe:.2f}_aIm0.00_r{r:.2f}', 
                                title=f'Rs={radius:.2f} fm, aRe={aRe:.2f} fm, r={r:.2f} fm; #it{{k}}* (MeV/#it{{c}}); #it{{C}}(#it{{k}}*)')
                hists[state].append(hist)
                
                hist_GeV = convert_and_rebin_histogram(hist, suffix='_GeV')
                set_root_object(hist_GeV, color=get_color(icomb), line_color=get_color(icomb), line_width=1,
                                name=f'h{radius:.2f}_aRe{aRe:.2f}_aIm0.00_r{r:.2f}_GeV', 
                                title=f'Rs={radius:.2f} fm, aRe={aRe:.2f} fm, r={r:.2f} fm; #it{{k}}* (GeV/#it{{c}}); #it{{C}}(#it{{k}}*)')
                hists_GeV[state].append(hist_GeV)

                legend.AddEntry(hist, f'a={aRe:.2f} fm, r={r:.2f} fm', 'l')
                
            canvas = TCanvas(f'canvas_{state}', f'canvas_{state}', 800, 600)
            for hist in hists[state]:
                hist.Draw('HIST SAME')
            legend.Draw()
            
            outdir.mkdir(state)
            outdir.cd(state)
            for hist in hists[state]:
                hist.Write()
            for hist in hists_GeV[state]:
                hist.Write()
            canvas.Write()
            
        hists['combined'] = []
        hists_GeV['combined'] = []
        weights = {'1s0': 1./4, '3s1': 3./4}
        
        legend_combined = init_legend(0.3, 0.2, 0.7, 0.5)
        outdir.mkdir('combined')
        
        hist_dicts = [hists, hists_GeV]
        for hist_dict in hist_dicts:
            for ihist1s0, hist in enumerate(hist_dict['1s0']):
                for ihist3s1, hist3s1 in enumerate(hist_dict['3s1']):
                    
                    # Only combine variation of one with the nominal of the other
                    if ihist1s0 != 0 and ihist3s1 != 0:
                        continue  
                        
                    name1s0 = hist_dict['1s0'][ihist1s0].GetName()
                    name3s1 = hist_dict['3s1'][ihist3s1].GetName()
                    combined_name = f'combined_{name1s0}_{name3s1}'
                    combined_hist = hist_dict['1s0'][ihist1s0].Clone(combined_name)
                    combined_hist.SetDirectory(0)
                    combined_hist.Scale(weights['1s0'])
                    combined_hist.Add(hist_dict['3s1'][ihist3s1], weights['3s1'])
                    
                    set_root_object(combined_hist, 
                                    color=get_color(ihist1s0+len(hist_dict['3s1'])+ihist3s1), 
                                    line_color=get_color(ihist1s0+len(hist_dict['3s1'])+ihist3s1), 
                                    line_width=1, name=combined_name,
                                    #title=f'Rs={radius:.2f} fm, aRe={aRe_list[ihist1s0]:.2f} fm, r={r_list[ihist1s0]:.2f} fm; #it{{k}}* (GeV/#it{{c}}); #it{{C}}(#it{{k}}*)'
                                    )
                    
                    hist_dict['combined'].append(combined_hist)
                    legend_combined.AddEntry(combined_hist, f'f_{{0}}={aRe_dict["1s0"][ihist1s0]:.2f} fm, r_{{0}}={r_dict["1s0"][ihist1s0]:.2f} fm; f_{{1}}={aRe_dict["3s1"][ihist3s1]:.2f} fm, r_{{1}}={r_dict["3s1"][ihist3s1]:.2f} fm', 'l')
            
            canvas_combined = TCanvas(f'canvas_combined_{radius:.2f}', f'canvas_combined_{radius:.2f}', 800, 600)
            for hist in hist_dict['combined']:
                hist.Draw('HIST SAME')
            legend_combined.Draw()
            
            outdir.cd('combined')
            canvas_combined.Write()
            for hist in hist_dict['combined']:
                hist.Write()
            
            del canvas_combined
        