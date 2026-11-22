"""Map pi-N correlation functions onto the pi-nucleus relative momentum.

Picture: Delta -> pi N, and the nucleon then coalesces into a nucleus of mass number A,
carrying p_A / A plus an internal momentum q. With velocities,
    k*_piN / mu_piN = k*_piA / mu_piA - q / m_N     (vectors)
so for each k*_piA we average the pi-N curve over q (isotropic Gaussian, sigma per component).
q = 0 gives the sharp mapping k*_piA = (mu_piA / mu_piN) k*_piN (x1.07 for d, x1.10 for A = 3).

Usage:
  python3 map_piN_to_piA.py input.root output.root --A 2 --sigma-q 60
  python3 map_piN_to_piA.py decay.root output.root --A 3 --sigma-q 90 --additive
--additive: map the curve itself (decay term, enters as + lambda*P*C_dec);
default   : map C - 1 and add 1 (scattering correlation).
Jacobians/phase-space weights are neglected: the result is a shape, its amplitude is fitted.
"""
import argparse
import numpy as np
from ROOT import TFile, TGraph

M_PI, M_N = 139.57039, 938.92


def reduced(m1, m2):
    return m1 * m2 / (m1 + m2)


def map_curve(k, y, A, sigma_q, n_samples=4000, seed=1):
    rng = np.random.default_rng(seed)
    q = rng.normal(0.0, sigma_q, size=(n_samples, 3)) if sigma_q > 0 else np.zeros((1, 3))
    mu_piN, mu_piA = reduced(M_PI, M_N), reduced(M_PI, A * M_N)
    out = np.empty_like(k)
    for i, kA in enumerate(k):
        v = np.array([0.0, 0.0, kA / mu_piA])[None, :] - q / M_N
        kN = mu_piN * np.linalg.norm(v, axis=1)
        out[i] = np.mean(np.interp(kN, k, y, left=y[0], right=y[-1]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--A", type=int, required=True, help="mass number of the nucleus")
    parser.add_argument("--sigma-q", type=float, default=0.0, help="internal momentum, MeV/c per component")
    parser.add_argument("--additive", action="store_true", help="map the curve itself, not C - 1")
    args = parser.parse_args()

    fin = TFile(args.input, "READ")
    fout = TFile(args.output, "RECREATE")
    for key in fin.GetListOfKeys():
        obj = key.ReadObj()
        if not obj.InheritsFrom("TGraph"):
            continue
        k = np.array(obj.GetX(), dtype=float)
        y = np.array(obj.GetY(), dtype=float)
        order = np.argsort(k)
        k, y = k[order], y[order]
        if args.additive:
            mapped = map_curve(k, y, args.A, args.sigma_q)
        else:
            mapped = 1.0 + map_curve(k, y - 1.0, args.A, args.sigma_q)
        g = TGraph(len(k), k, mapped)
        g.SetName(f"{obj.GetName()}_A{args.A}_sq{args.sigma_q:g}")
        g.SetTitle(f"{obj.GetTitle()} mapped to pi-A (A={args.A}, sigma_q={args.sigma_q:g} MeV/c)")
        fout.cd()
        g.Write()
    fout.Close()
    fin.Close()


if __name__ == "__main__":
    main()
