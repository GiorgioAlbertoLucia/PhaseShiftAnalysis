import numpy as np

from ROOT import TFile, TGraphAsymmErrors, TCanvas

from torchic.utils.root import set_alice_global_style, set_root_object
from torchic.core.histogram import load_hist
from torchic.utils.colors import get_color

def combine_histograms_with_weights(hist1, hist2, weight1, weight2, name="combined_hist"):
    """Combine two histograms with given weights."""
    combined_hist = hist1.Clone(name)
    combined_hist.Scale(weight1)
    
    combined_hist.Add(hist2, weight2)
    return combined_hist

def compute_bin_by_bin_std(nominal_hist, variation_hists):
    """Compute the bin-by-bin standard deviation from the nominal histogram."""
    n_bins = nominal_hist.GetNbinsX()
    n_var = len(variation_hists)
    
    std_hist = nominal_hist.Clone()
    std_hist.Reset()
    
    for b in range(1, n_bins + 1):
        nom_val = nominal_hist.GetBinContent(b)
        sq_sum = sum((h.GetBinContent(b) - nom_val) ** 2 for h in variation_hists)
        std = np.sqrt(sq_sum / n_var) if n_var > 0 else 0.0
        std_hist.SetBinContent(b, std)
        std_hist.SetBinError(b, 0.0)
    
    return std_hist

def create_band(x_values, yvalues, ex_values, ey_low, ey_high):
    """
    Build a TGraphAsymmErrors whose central values come from model_curve
    (e.g. the fitted background TGraph/RooCurve) and whose up/down widths
    come from the relative bandwidth of the background variations.
    """
    graph = TGraphAsymmErrors(len(x_values))
    for i, (x, ex, y, dlo, dhi) in enumerate(zip(x_values, ex_values, yvalues, ey_low, ey_high)):
        graph.SetPoint(i, x, y)
        graph.SetPointError(i, ex, ex, dlo, dhi)
    return graph

def create_band_from_hist(h_nominal, h_sigma):
    """
    Build a TGraphAsymmErrors whose central values come from the nominal histogram
    and whose up/down widths come from the standard deviation histogram.
    """
    x_values = [h_nominal.GetBinCenter(i) for i in range(1, h_nominal.GetNbinsX() + 1)] 
    y_values = [h_nominal.GetBinContent(i) for i in range(1, h_nominal.GetNbinsX() + 1)]
    ex_values = [h_nominal.GetBinWidth(i) / 2 for i in range(1, h_nominal.GetNbinsX() + 1)]
    ey_low = [(h_sigma.GetBinContent(i)) for i in range(1, h_nominal.GetNbinsX() + 1)]
    ey_high = [(h_sigma.GetBinContent(i)) for i in range(1, h_nominal.GetNbinsX() + 1)]
    
    return create_band(x_values, y_values, ex_values, ey_low, ey_high)


if __name__ == "__main__":

    set_alice_global_style()
    
    radii = [4.33, 3.46]  # fm --> Rs / sqrt(2)
    input_file  = "/home/galucia/PhaseShiftAnalysis/numerical_lednicky/pHe/output/comparison_pHe3_LL.root"

    w_1s0, w_3s1 = 0.25, 0.75

    outfile_path = "/home/galucia/PhaseShiftAnalysis/numerical_lednicky/pHe/output/pHe3_LL_bands.root"
    outfile = TFile.Open(outfile_path, "RECREATE")
    
    for iradius, radius in enumerate(radii):
        outdir = outfile.mkdir(f"Rs{radius:.2f}")

        combined_hists = {}  # (i, j) -> TH1
        radius_dir = f"Rs{radius:.2f}"

        hist_1s0 = [f"h{radius:.2f}_aRe-11.26_aIm0.00_r1.65", f"h{radius:.2f}_aRe-11.30_aIm0.00_r1.65",
                    f"h{radius:.2f}_aRe-11.22_aIm0.00_r1.65", f"h{radius:.2f}_aRe-11.26_aIm0.00_r1.36",
                    f"h{radius:.2f}_aRe-11.26_aIm0.00_r1.94"]

        hist_3s1 = [f"h{radius:.2f}_aRe-9.06_aIm0.00_r1.36", f"h{radius:.2f}_aRe-9.10_aIm0.00_r1.36",
                    f"h{radius:.2f}_aRe-9.14_aIm0.00_r1.36", f"h{radius:.2f}_aRe-9.06_aIm0.00_r1.11",
                    f"h{radius:.2f}_aRe-9.06_aIm0.00_r1.61"]

        for ihist, hist_1s0_name in enumerate(hist_1s0):
            h1 = load_hist(input_file, f"{radius_dir}/1s0/{hist_1s0_name}")
            for jhist, hist_3s1_name in enumerate(hist_3s1):
                h3 = load_hist(input_file, f"{radius_dir}/3s1/{hist_3s1_name}")
                hist_combined = combine_histograms_with_weights(h1, h3, w_1s0, w_3s1, name=f"combined_{ihist}_{jhist}")
                combined_hists[(ihist, jhist)] = hist_combined

        nominal = combined_hists[(0, 0)]
        nominal.SetName("nominal")

        outdir.cd()
        for h in combined_hists.values():
            h.Write()

        variations = [h for key, h in combined_hists.items() if key != (0, 0)]
        n_var = len(variations)

        h_sigma = compute_bin_by_bin_std(nominal, variations)
        set_root_object(h_sigma, name=f"h{radius:.2f}_sigma", title="Bin-by-bin std-dev from nominal;"
                                                              " #it{k}{*} (MeV/#it{c}); #sigma_{#it{C}}")
        
        g_band = create_band_from_hist(nominal, h_sigma)
        set_root_object(g_band, name=f"g{radius:.2f}_band", line_color=get_color(iradius), 
                        fill_color=get_color(iradius), fill_style=1001,
                        title="Band around nominal; #it{k}{*} (MeV/#it{c}); #it{C}(#it{k}{*})")
        
        canvas = TCanvas(f"canvas_Rs{radius:.2f}", f"canvas_Rs{radius:.2f}", 800, 600)
        hframe = canvas.DrawFrame(0, 0, 400, 1.2, 
                                  f"Combined Correlation Function with Band; #it{{k}}* (MeV/#it{{c}}); #it{{C}}(#it{{k}}*)")
        g_band.Draw("l3 same")
        
        outdir.cd()
        h_sigma.Write()
        g_band.Write()
        canvas.Write()

    