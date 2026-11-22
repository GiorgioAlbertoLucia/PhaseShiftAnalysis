"""Combine the Coulomb + s-wave correlation function with the Delta(1232) terms.

    C(k*) = 1 + lambda_fsi * (C_FSI(k*) - 1) + lambda_sca * dC_sca(k*) + lambda_dec * dC_dec(k*)

C_FSI  : graphs g<R>_aRe<..>_aIm<..>_r<d0> of the main ROOT file
dC_sca : gDeltaSca_R<R> of the Delta file (same source size R)
dC_dec : gDeltaDec of the Delta file

Every combination of the lambda values given is written to its own directory of the output file.
lambda values can be listed (0 0.5 1) or given as ranges start:stop:step (0:1:0.25, stop included).

Examples:
  python3 combine_cf_delta.py --cf piminus_d.root --delta piminus_d_Delta_rsca1.00_rdec0.20_sq60.root \\
      -o piminus_d_combined.root --lambda-sca 0.5 --lambda-dec 0.5

  # grid: 3 x 5 combinations, only two radii and the nominal scattering length, also as TH1D
  python3 combine_cf_delta.py --cf piminus_d.root --delta piminus_d_Delta_rsca1.00_rdec0.20_sq60.root \\
      -o grid.root --lambda-sca 0:1:0.5 --lambda-dec 0:1:0.25 --radii 1.2 2.0 --match aRe-0.0353 --hist

  # list what is in the files
  python3 combine_cf_delta.py --cf piminus_d.root --delta piminus_d_Delta_rsca1.00_rdec0.20_sq60.root --list
"""
import argparse
import sys

import numpy as np
from ROOT import TFile, TGraph, TH1D


def parse_values(tokens):
    """['0', '0.5', '0:1:0.25'] -> sorted unique floats (ranges include stop)."""
    values = []
    for tok in tokens:
        if ":" in tok:
            parts = [float(x) for x in tok.split(":")]
            if len(parts) != 3 or parts[2] <= 0:
                raise argparse.ArgumentTypeError(f"bad range '{tok}', expected start:stop:step with step > 0")
            start, stop, step = parts
            n = int(np.floor((stop - start) / step + 1e-9)) + 1
            values += [round(start + i * step, 10) for i in range(n)]
        else:
            values.append(float(tok))
    return sorted(set(values))


def read_graphs(path):
    """name -> (k, y) as sorted numpy arrays, for every TGraph in the file."""
    f = TFile(path, "READ")
    if not f or f.IsZombie():
        sys.exit(f"Error: cannot open {path}")
    graphs = {}
    for key in f.GetListOfKeys():
        obj = key.ReadObj()
        if not obj.InheritsFrom("TGraph"):
            continue
        k = np.array([obj.GetPointX(i) for i in range(obj.GetN())], dtype=float)
        y = np.array([obj.GetPointY(i) for i in range(obj.GetN())], dtype=float)
        order = np.argsort(k)
        graphs[obj.GetName()] = (k[order], y[order])
    f.Close()
    return graphs


def radius_of(cf_name):
    """'g1.20_aRe-0.0353_aIm0.0088_r0.00' -> '1.20' (None if not a CF graph)."""
    head = cf_name.split("_")[0]
    if not head.startswith("g"):
        return None
    try:
        float(head[1:])
    except ValueError:
        return None
    return head[1:]


def to_hist(name, k, y):
    """TH1D with bins centred on the k points (same convention as the other macros)."""
    edges = np.empty(len(k) + 1)
    edges[1:-1] = 0.5 * (k[1:] + k[:-1])
    edges[0] = k[0] - 0.5 * (k[1] - k[0])
    edges[-1] = k[-1] + 0.5 * (k[-1] - k[-2])
    h = TH1D(name, f"{name};k* (MeV/c);C(k*)", len(k), edges)
    h.SetDirectory(0)
    for i, v in enumerate(y):
        h.SetBinContent(i + 1, v)
        h.SetBinError(i + 1, 0.0)
    return h


def fmt(x):
    return f"{x:g}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cf", required=True, help="main ROOT file (Coulomb + s wave), from --output-root")
    p.add_argument("--delta", required=True, help="Delta ROOT file of the same run (--resonance / --mode delta)")
    p.add_argument("-o", "--output", help="output ROOT file (overwritten)")
    p.add_argument("--lambda-fsi", nargs="+", default=["1"], metavar="L",
                   help="strength(s) of C_FSI - 1 (default: 1)")
    p.add_argument("--lambda-sca", nargs="+", default=["1"], metavar="L",
                   help="strength(s) of the Delta scattering term (default: 1)")
    p.add_argument("--lambda-dec", nargs="+", default=["1"], metavar="L",
                   help="strength(s) of the Delta decay term (default: 1)")
    p.add_argument("--radii", nargs="+", type=float, metavar="R",
                   help="only these source sizes (as in the graph names, e.g. 1.2 2.0)")
    p.add_argument("--match", nargs="+", metavar="TEXT",
                   help="only CF graphs whose name contains all these strings (e.g. aRe-0.0353)")
    p.add_argument("--hist", action="store_true", help="also write each curve as a TH1D (h... names)")
    p.add_argument("--components", action="store_true",
                   help="also write the inputs used (C_FSI, dC_sca, dC_dec) to a 'components' directory")
    p.add_argument("--list", action="store_true", help="list the graphs in both files and exit")
    a = p.parse_args()

    try:
        lam_fsi, lam_sca, lam_dec = (parse_values(v) for v in (a.lambda_fsi, a.lambda_sca, a.lambda_dec))
    except argparse.ArgumentTypeError as e:
        p.error(str(e))

    cf = read_graphs(a.cf)
    delta = read_graphs(a.delta)

    # Catch swapped files early
    if "gDeltaDec" not in delta:
        hint = " (the files look swapped: --cf and --delta)" if "gDeltaDec" in cf else ""
        sys.exit(f"Error: no gDeltaDec in {a.delta}{hint}")

    cf_names = sorted(n for n in cf if radius_of(n) is not None)
    if a.list:
        print(f"{a.cf}: {len(cf_names)} CF graphs")
        for n in cf_names:
            print("   ", n)
        print(f"{a.delta}: {len(delta)} graphs")
        for n in sorted(delta):
            print("   ", n)
        return
    if not a.output:
        p.error("-o/--output is required (unless --list)")

    # Select CF graphs and pair them with the Delta scattering graph of the same radius
    selected, missing = [], []
    for n in cf_names:
        R = radius_of(n)
        if a.radii and not any(abs(float(R) - r) < 1e-6 for r in a.radii):
            continue
        if a.match and not all(m in n for m in a.match):
            continue
        sca = f"gDeltaSca_R{R}"
        (selected if sca in delta else missing).append((n, sca))
    if missing:
        print(f"Warning: {len(missing)} CF graphs have no gDeltaSca_R<R> in {a.delta} and are skipped, e.g. {missing[0][0]}")
    if not selected:
        sys.exit("Error: no CF graph selected/matched; check --radii, --match, or run with --list")

    kd, yd = delta["gDeltaDec"]
    combos = [(lf, ls, ld) for lf in lam_fsi for ls in lam_sca for ld in lam_dec]
    print(f"{len(selected)} CF graphs x {len(combos)} lambda combinations -> {a.output}")

    fout = TFile(a.output, "RECREATE")
    if a.components:
        d = fout.mkdir("components")
        d.cd()
        written = set()
        for n, sca in selected:
            for name, (k, y) in ((n, cf[n]), (sca, delta[sca])):
                if name not in written:
                    g = TGraph(len(k), k, y)
                    g.SetName(name)
                    g.Write()
                    written.add(name)
        g = TGraph(len(kd), kd, yd)
        g.SetName("gDeltaDec")
        g.Write()

    n_written = 0
    for lf, ls, ld in combos:
        dirname = f"lfsi{fmt(lf)}_lsca{fmt(ls)}_ldec{fmt(ld)}"
        d = fout.mkdir(dirname)
        d.cd()
        for n, sca in selected:
            k, c = cf[n]
            ks, ys = delta[sca]
            # Delta curves evaluated at the CF k points (linear interpolation)
            total = 1.0 + lf * (c - 1.0) + ls * np.interp(k, ks, ys) + ld * np.interp(k, kd, yd)
            g = TGraph(len(k), k, total)
            g.SetName(n)
            g.SetTitle(f"{n}: #lambda_{{FSI}}={fmt(lf)}, #lambda_{{sca}}={fmt(ls)}, #lambda_{{dec}}={fmt(ld)};"
                       "k* (MeV/c);C(k*)")
            g.Write()
            if a.hist:
                to_hist("h" + n[1:], k, total).Write()
            n_written += 1
    fout.Close()
    print(f"Wrote {n_written} curves to {a.output} (one directory per lambda combination)")


if __name__ == "__main__":
    main()