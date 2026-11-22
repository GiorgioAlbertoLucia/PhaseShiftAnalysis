# combine_cf_delta.py — python3 combine_cf_delta.py piminus_d.root piminus_d_Delta_rsca1.00_rdec0.20_sq60.root out.root 1.0 1.0
import sys
from ROOT import TFile, TGraph

cf_file, delta_file, out_file = sys.argv[1:4]
lam_sca, lam_dec = float(sys.argv[4]), float(sys.argv[5])

fcf, fd = TFile(cf_file), TFile(delta_file)
g_dec = fd.Get("gDeltaDec")
fout = TFile(out_file, "RECREATE")

for key in fcf.GetListOfKeys():
    g = key.ReadObj()
    if not g.InheritsFrom("TGraph"):
        continue
    R = g.GetName().split("_")[0][1:]            # "g1.20_aRe..." -> "1.20"
    g_sca = fd.Get(f"gDeltaSca_R{R}")
    if not g_sca:
        continue
    out = TGraph(g.GetN())
    for i in range(g.GetN()):
        k, c = g.GetPointX(i), g.GetPointY(i)
        out.SetPoint(i, k, c + lam_sca * g_sca.Eval(k) + lam_dec * g_dec.Eval(k))
    out.SetName(g.GetName() + f"_withDelta_ls{lam_sca:g}_ld{lam_dec:g}")
    out.Write()
fout.Close()