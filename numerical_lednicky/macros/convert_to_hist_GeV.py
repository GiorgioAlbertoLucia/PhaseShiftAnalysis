import numpy as np

from ROOT import TCanvas, TFile, TH1D

from torchic.core.graph import load_graph
from torchic.utils.colors import get_color
from torchic.utils.root import set_alice_global_style, set_root_object, init_legend

def graph_to_hist(graph):
    
    bin_centers = np.ascontiguousarray(graph.GetX())
    bin_contents = np.ascontiguousarray(graph.GetY())
    #bin_errors = np.ascontiguousarray(graph.GetEY())
    bin_errors = np.zeros(len(bin_contents), dtype=np.float64)  # Set errors to zero if not available
    
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
        hist.SetBinError(i + 1, bin_errors[i])
    
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

    infile_path = '/home/galucia/PhaseShiftAnalysis/numerical_lednicky/pHe/output/pHe3_LL_radius_corrected.root'
    outfile = TFile('/home/galucia/PhaseShiftAnalysis/numerical_lednicky/pHe/output/pHe3_LL_radius_corrected_GeV.root', 'RECREATE')
    
    infile = TFile(infile_path, 'READ')
    
    for graph_key in infile.GetListOfKeys():
        
        hist_name = graph_key.GetName().replace('g', 'h')
        
        graph = infile.Get(graph_key.GetName())

        hist = graph_to_hist(graph)
        set_root_object(hist, title=f'{graph.GetTitle()}; #it{{k}}* (MeV/#it{{c}}); #it{{C}}(#it{{k}}*)',
                                name=hist_name)
        
        hist_GeV = convert_and_rebin_histogram(hist, suffix='_GeV')
        set_root_object(hist_GeV, title=f'{graph.GetTitle()}; #it{{k}}* (GeV/#it{{c}}); #it{{C}}(#it{{k}}*)',
                        name=hist_name)
        hist_GeV.SetDirectory(outfile)  # Detach the histogram from any ROOT file
        
        
        #for ibin in range(1, hist_GeV.GetNbinsX() + 1):
        #    print(f"Bin {ibin}: k* = {hist_GeV.GetBinCenter(ibin):.3f} GeV/c, C(k*) = {hist_GeV.GetBinContent(ibin):.3f} ± {hist_GeV.GetBinError(ibin):.3f}")
        #exit(0)

        outfile.cd()
        hist_GeV.Write()
    
    infile.Close()
    outfile.Close()
        