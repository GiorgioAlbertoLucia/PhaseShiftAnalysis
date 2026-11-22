"""Delta(1232) contribution for any pi-nucleus system, built from pi-p and pi-n results.

Every nucleus can take up a proton or a neutron from Delta -> pi N, so two channels contribute:
    pi+ :  Delta++ -> pi+ p  (BR 1)     Delta+ -> pi+ n  (BR 1/3)
    pi- :  Delta0  -> pi- p  (BR 1/3)   Delta- -> pi- n  (BR 1)
Default weights (equal production of the Delta charge states, nucleon taken up with probability Z/A or N/A):
    decay term     : w_N = BR(Delta -> pi N) * (Z/A or N/A)
    scattering term: w_N = (Z/A or N/A)   (the I = 3/2 isospin factor is already in the pi-N amplitude)
Weights are normalised to sum to 1; the overall amplitude (lambda, P) is a fit parameter.
Override them with --w-proton / --w-neutron.

Scattering: the pi-N Coulomb is removed before mapping (C_{Coul+Delta} - C_{Coul}),
since the pi-nucleus Coulomb is already in the s-wave pi-A calculation. Output is Delta C(k*_piA),
to be added to the pi-A correlation function with a fitted weight.
Decay: pass the --res-decay-only outputs and no references.

Usage (pi+ d, scattering):
  python3 delta_for_system.py --system d --pion-charge +1 \
      --proton P/output/TheoCF_PipP_Res_x1.00_rreg1.00.root --neutron N/output/TheoCF_PipN_Res_x0.33_rreg1.00.root \
      --proton-ref P/output/TheoCF_PipP.root --neutron-ref N/output/TheoCF_PipN.root \
      --sigma-q 60 --output DeltaSca_PipD.root
"""
import argparse
import numpy as np
from ROOT import TFile, TGraph

from map_piN_to_piA import map_curve

NUCLEI = {"d": (2, 1), "t": (3, 1), "he3": (3, 2)}  # (A, Z)


def default_weights(system, pion_charge, decay):
    A, Z = NUCLEI[system]
    fp, fn = Z / A, (A - Z) / A
    if decay:
        br_p, br_n = (1.0, 1.0 / 3.0) if pion_charge > 0 else (1.0 / 3.0, 1.0)
        fp, fn = fp * br_p, fn * br_n
    s = fp + fn
    return fp / s, fn / s


def read_graphs(path):
    graphs = {}
    if path is None:
        return graphs
    f = TFile(path, "READ")
    for key in f.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom("TGraph"):
            k, y = np.array(obj.GetX(), float), np.array(obj.GetY(), float)
            o = np.argsort(k)
            graphs[obj.GetName()] = (k[o], y[o])
    f.Close()
    return graphs


def strip_params(name):
    # g<size>_aRe..._aIm..._r... -> g<size> (pi-N files differ only in the resonance settings)
    return name.split("_")[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=NUCLEI, required=True)
    p.add_argument("--pion-charge", type=float, required=True)
    p.add_argument("--proton", required=True)
    p.add_argument("--neutron", required=True)
    p.add_argument("--proton-ref", help="Coulomb-only pi-p file (scattering)")
    p.add_argument("--neutron-ref", help="Coulomb-only pi-n file (scattering)")
    p.add_argument("--decay", action="store_true", help="inputs are --res-decay-only outputs")
    p.add_argument("--sigma-q", type=float, default=0.0, help="internal nucleon momentum, MeV/c per component")
    p.add_argument("--w-proton", type=float)
    p.add_argument("--w-neutron", type=float)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    A, _ = NUCLEI[a.system]
    wp, wn = default_weights(a.system, a.pion_charge, a.decay)
    if a.w_proton is not None: wp = a.w_proton
    if a.w_neutron is not None: wn = a.w_neutron
    print(f"system {a.system}, pion charge {a.pion_charge:+g}: w_p = {wp:.3f}, w_n = {wn:.3f}")

    chans = []
    for path, ref, w in ((a.proton, a.proton_ref, wp), (a.neutron, a.neutron_ref, wn)):
        g = {strip_params(n): v for n, v in read_graphs(path).items()}
        r = {strip_params(n): v for n, v in read_graphs(ref).items()}
        if not a.decay and not r:
            raise SystemExit("scattering mode needs Coulomb-only references (--proton-ref/--neutron-ref)")
        chans.append((g, r, w))

    fout = TFile(a.output, "RECREATE")
    for name in sorted(set(chans[0][0]) & set(chans[1][0])):
        total = None
        for g, r, w in chans:
            k, y = g[name]
            if not a.decay:
                kr, yr = r[name]
                y = y - np.interp(k, kr, yr)
            m = w * map_curve(k, y, A, a.sigma_q)
            total = m if total is None else total + m
        out = TGraph(len(k), k, total)
        out.SetName(f"{name}_Delta{'Dec' if a.decay else 'Sca'}_{a.system}")
        out.SetTitle(f"Delta {'decay' if a.decay else 'scattering'} mapped to pi-{a.system};k* (MeV/c);#DeltaC")
        out.Write()
    fout.Close()


if __name__ == "__main__":
    main()
