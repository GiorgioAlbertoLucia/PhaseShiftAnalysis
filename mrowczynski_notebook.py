import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import ROOT

# Physical constants and parameters
alpha = 1/137  # fine-structure constant
mu = 938.3 * (2*938.3 + 939.6 - 7.7) / (938.3 + 2*938.3 + 939.6 - 7.7)  # 3He-p reduced mass in MeV
E0 = 938.3 + (2*938.3 + 939.6 - 7.7) + 4.07  # 4Li resonance mass in MeV
q0 = np.sqrt(2*mu*4.07)  # relative momentum at resonance peak in MeV
G = 6.03  # 4Li resonance width in MeV
a1 = 11.1/197.3  # scattering length of 3He-p in singlet state in inverse MeV
a3 = 9.05/197.3  # scattering length of 3He-p in triplet state in inverse MeV
aB = 1/(2*mu*alpha)  # Bohr radius of 3He-p in inverse MeV

print(f'{aB=}')

Delta_R = 1  # resonance strength

# Gamow factor
def Gamow(q):
    return (2*np.pi) / (aB*q * (np.exp(2*np.pi/(aB*q)) - 1))

# Singlet scattering amplitudes
def Reamp1(q):
    return - a1 / (1 + q**2 * a1**2)

def Imamp1(q):
    return (q*a1**2) / (1 + q**2 * a1**2)

def Mod2amp1(q):
    return a1**2 / (1 + q**2 * a1**2)

# Triplet scattering amplitudes
def Reamp3(q):
    return - a3 / (1 + q**2 * a3**2)

def Imamp3(q):
    return (q*a3**2) / (1 + q**2 * a3**2)

def Mod2amp3(q):
    return a3**2 / (1 + q**2 * a3**2)

# Auxiliary function
def Fun(Rs, r, q):
    return np.exp(-(3*r**2)/(8*Rs**2)) * np.sin(2*q*r)

# Singlet correlation function due to s-wave scattering
def Funcorr1s(q, Rs):
    integrand = lambda r: Fun(Rs, r, q)
    integral, _ = quad(integrand, 0, 0.5)
    
    term1 = 1 + (3/(4 * Rs**2)) * Mod2amp1(q)
    term2 = -3 * (1 - np.exp(-(8/3) * Rs**2 * q**2)) / (4 * Rs**2 * q) * Imamp1(q)
    term3 = (np.sqrt(27)*Reamp1(q)) / (np.sqrt(32 * np.pi) * Rs**3 * q) * integral
    
    return term1 + term2 + term3

# Triplet correlation function due to s-wave scattering
def Funcorr3s(q, Rs):
    integrand = lambda r: Fun(Rs, r, q)
    integral, _ = quad(integrand, 0, 0.5)
    
    term1 = 1 + (3/(4*Rs**2)) * Mod2amp3(q)
    term2 = -3 * (1 - np.exp(-(8/3) * Rs**2 * q**2)) / (4 * Rs**2 * q) * Imamp3(q)
    term3 = (np.sqrt(27)*Reamp3(q)) / (np.sqrt(32*np.pi) * Rs**3 * q) * integral
    
    return term1 + term2 + term3

# Energy in CM frame
def Energy(q):
    return q**2 / (2*mu) + 938.3 + (2*938.3 + 939.6 - 7.7)

# Resonance scattering amplitudes
def Reampri(q):
    return -Delta_R * (3/(2*q0)) * (G*(Energy(q) - E0)) / ((Energy(q) - E0)**2 + 0.25*G**2)

def Imampri(q):
    return Delta_R * (3/(4*q0)) * G**2 / ((Energy(q) - E0)**2 + 0.25*G**2)

def Mod2ampri(q):
    return Delta_R**2 * (9/(4*q0**2)) * G**2 / ((Energy(q) - E0)**2 + 0.25*G**2)

# Resonance integrals
def ReK1(q, Rs):
    def integrand(r):
        return np.exp(-(3*r**2)/(8*Rs**2)) * (np.sin(2*q*r) - (2*(np.sin(q*r))**2)/q*r)
    integral, _ = quad(integrand, 0, 0.5)
    return (2*np.pi/q) * integral

def ImK1(q, Rs):
    def integrand_inner(r):
        return np.exp(-(3*r**2)/(8*Rs**2)) * np.sin(2*q*r) / r
    integral_inner, _ = quad(integrand_inner, 0, 0.5)
    
    def integrand_outer(r):
        return np.exp(-(8*q**2*Rs**2)/3)
    integral_outer, _ = quad(integrand_outer, 0, 0.5)
    
    return (2*np.pi**3*Rs) / (3**(3/2)*q) * (1 + np.exp(-(8*q**2*Rs**2)/3)) - (2*np.pi/q**2) * integral_inner

# Correlation function due to resonance scattering
def Funcorrr(q, Rs):
    term1 = 1 + (1/(4*Rs**2)) * Mod2ampri(q)
    term2 = 2 * ((3/(8*Rs**2))**(3/2)) * (Reampri(q)*ReK1(q, Rs) - Imampri(q)*ImK1(q, Rs))
    return term1 + term2

# Complete correlation function
def Funcorr(q, Rs):
    result = (Gamow(q) * ((1/4)*Funcorr1s(q, Rs) + (3/4)*(Funcorr3s(q, Rs))) ) # + Funcorrr(q, Rs) - 1.0)))
    return result if result > 0 else 0.0

# Generate plot
#if __name__ == "__main__":
#    q_values = np.linspace(0., 150, 100)
#    
#    # Calculate correlation values
#    Rs_values = [1.5, 2., 2.5, 3.]
#    
#    plt.figure(figsize=(10, 6))
#    
#    for Rs in Rs_values:
#        corr_values = [Funcorr(q, Rs/197.3) for q in q_values]
#        plt.plot(q_values, corr_values, linewidth=2, label=f'Rs = {Rs}')
#    
#    plt.xlim(0., 150.0)
#    plt.ylim(0, 2.5)
#    plt.xlabel('k* (MeV/c)', fontsize=12)
#    plt.ylabel('C(k*)', fontsize=12)
#    plt.title('', fontsize=14)
#    plt.legend()
#    plt.grid(True, alpha=0.3)
#    plt.tight_layout()
#    plt.savefig('output/mrowczynski_3He_p_correlation.pdf', dpi=300)


if __name__ == "__main__":
    q_values = np.linspace(0., 150., 100)

    Rs_values = [1.5, 2., 2.5, 3., 6.23]

    c = ROOT.TCanvas("c", "3He-p correlation", 800, 600)
    outfile = ROOT.TFile("output/mrowczynski_3He_p_correlation.root", "RECREATE")

    mg = ROOT.TMultiGraph()
    legend = ROOT.TLegend(0.55, 0.55, 0.82, 0.78)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    colors = [ROOT.kOrange-3, ROOT.kBlue-2, ROOT.kGreen+2, ROOT.kRed+1, ROOT.kViolet+1, ROOT.kCyan+3, ROOT.kMagenta, ROOT.kTeal+2]
    hists, graphs = [], []

    for i, Rs in enumerate(Rs_values):
        corr_values = np.array([Funcorr(q, Rs/197.3) for q in q_values])

        graph = ROOT.TGraph(
            len(q_values),
            q_values.astype(np.float64),
            corr_values.astype(np.float64)
        )

        hist = ROOT.TH1F(f"hist_Rs_{Rs}", f"3He-p Correlation Function Rs={Rs} fm", 100, 0., 150.)
        for j in range(len(q_values)):
            hist.Fill(q_values[j], corr_values[j])

        graph.SetLineWidth(2)
        graph.SetLineColor(colors[i])
        graph.SetMarkerColor(colors[i])

        graphs.append(graph)
        hists.append(hist)
        if Rs == 6.23:
            continue
        mg.Add(graph, "L")
        legend.AddEntry(graph, f"R_{{s}} = {Rs} fm", "l")

    mg.Draw("A")
    mg.GetXaxis().SetTitle("#it{k}* (MeV/#it{c})")
    mg.GetYaxis().SetTitle("C(#it{k}*)")
    mg.GetXaxis().SetLimits(0., 150.)
    mg.SetMinimum(0.)
    mg.SetMaximum(2.5)

    legend.Draw()
    c.Update()
    c.SaveAs("output/mrowczynski_3He_p_correlation.pdf")


    for hist, graph, Rs in zip(hists, graphs, Rs_values):
        outdir = outfile.mkdir(f"r={Rs:.2f}_fm")
        outdir.cd()
        hist.Write()
        graph.Write()

    c.Write()
    outfile.Close()
