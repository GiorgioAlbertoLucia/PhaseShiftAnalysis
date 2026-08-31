#pragma once

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>

#include <TRandom3.h>
#include <TMinuit.h>
#include <TMath.h>
#include <TGraph.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TMultiGraph.h>

#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_coulomb.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_linalg.h>

#include "Constants.h"
#include "Matrix.h"

using namespace std::complex_literals;

class pHe3DoubleGaus {

public:
    pHe3DoubleGaus();
    ~pHe3DoubleGaus() = default;

    void Initialize();

    void LoadExperimentalData(const std::vector<double>& q_exp,
                              const std::vector<std::vector<double>>& delta_exp,
                              const std::vector<std::vector<double>>& delta_exp_error);

    double GetPhaseShift(int channel, double q) const;

    void FitToPhaseShifts(int channel, int n_iterations = 1000, int debug_level = -1);

    void PrintPhaseShifts(const std::string& filename = "") const;
    void PlotPhaseShifts(int channel = -1, const char* outputPdfPath = "pHe3DoubleGaus.pdf") const;

    void GetParameters(int channel, std::vector<double>& sigma_vals,
                       std::vector<double>& V_vals) const;
    void SetParameters(int channel, const std::vector<double>& sigma_vals,
                       const std::vector<double>& V_vals);

private:
    // ── Physical constants ────────────────────────────────────────────────────
    double m_proton, m_He3, mu;
    double q1q2, alpha;
    int    n_channels, l_max;

    // ── Momentum grid ─────────────────────────────────────────────────────────
    double q_max, dq;
    int    n_q_points;

    // ── Experimental data ─────────────────────────────────────────────────────
    std::vector<double>              m_q_exp;
    std::vector<std::vector<double>> m_delta_exp;
    std::vector<std::vector<double>> m_delta_exp_error;

    // ── Channel quantum numbers ───────────────────────────────────────────────
    std::vector<int> L_values, S_values, ell;

    // ── Double-Gaussian parameters per channel: sigma[ch][0,1], V0[ch][0,1] ──
    std::vector<std::vector<double>> sigma;   // widths  (fm)
    std::vector<std::vector<double>> V0;      // depths  (MeV)

    // ── Phase shifts & Coulomb phases ─────────────────────────────────────────
    std::vector<std::vector<double>>              delta;
    std::vector<std::vector<std::complex<double>>> cgs_qwell;  // [iq][l]

    // ── Minuit statics ────────────────────────────────────────────────────────
    static pHe3DoubleGaus* s_fit_instance;
    static int            s_fit_channel;
    static void FCN(int& npar, double* grad, double& fval, double* par, int flag);

    // ── Helpers ───────────────────────────────────────────────────────────────
    void InitializeParameters();
    void InitializeArrays();
    void ComputePhaseShiftsAllChannels();

    // Double-Gaussian potential at radius r for a given channel
    double Potential(int ch, double r) const {
        return V0[ch][0] * std::exp(-r*r / (sigma[ch][0]*sigma[ch][0]))
             + V0[ch][1] * std::exp(-r*r / (sigma[ch][1]*sigma[ch][1]));
    }

    // RK4 step for the reduced radial equation u'' = f(r)*u
    // f(r) = [l(l+1)/r² + 2μ(V(r) - E)/ℏc²]   (E = q²/2μ in natural units)
    // State vector: y = {u, u'}
    void RK4Step(int ch, int l, double q, double r, double dr,
                 double u, double up, double& u_new, double& up_new) const
    {
        auto f = [&](double rr, double /*uu*/) -> double {
            double centrifugal = (l * (l + 1)) / (rr * rr + 1e-30);
            double E_kin       = (q * q) / (Constants::HBARC * Constants::HBARC);
            double V_term      = 2.0 * mu * Potential(ch, rr)
                                 / (Constants::HBARC * Constants::HBARC);
            return centrifugal - E_kin + V_term;
        };

        // k1
        double k1u  = up;
        double k1up = f(r, u) * u;
        // k2
        double k2u  = up  + 0.5*dr*k1up;
        double k2up = f(r + 0.5*dr, u + 0.5*dr*k1u) * (u + 0.5*dr*k1u);
        // k3
        double k3u  = up  + 0.5*dr*k2up;
        double k3up = f(r + 0.5*dr, u + 0.5*dr*k2u) * (u + 0.5*dr*k2u);
        // k4
        double k4u  = up  + dr*k3up;
        double k4up = f(r + dr,     u + dr*k3u)      * (u + dr*k3u);

        u_new  = u  + (dr/6.0)*(k1u  + 2*k2u  + 2*k3u  + k4u);
        up_new = up + (dr/6.0)*(k1up + 2*k2up + 2*k3up + k4up);
    }

    std::complex<double> CGamma(std::complex<double> z);
};

// ── Static member definitions ─────────────────────────────────────────────────
pHe3DoubleGaus* pHe3DoubleGaus::s_fit_instance = nullptr;
int            pHe3DoubleGaus::s_fit_channel   = -1;

// ─────────────────────────────────────────────────────────────────────────────
pHe3DoubleGaus::pHe3DoubleGaus() {
    m_proton = Constants::M_PROTON;
    m_He3    = Constants::M_HELIUM3;
    mu       = (m_proton * m_He3) / (m_proton + m_He3);
    q1q2     = Constants::Z_PROTON * Constants::Z_HELIUM3;
    alpha    = Constants::ALPHA_EM;

    n_channels = 4;
    l_max      = 1;

    L_values = {0, 0, 1, 1};
    S_values = {0, 1, 0, 1};
    ell      = {0, 0, 1, 1};

    q_max      = 100.0;
    dq         = 1.0;
    n_q_points = static_cast<int>(q_max / dq) + 1;

    InitializeArrays();
    InitializeParameters();
}

void pHe3DoubleGaus::InitializeArrays() {
    sigma.resize(n_channels, std::vector<double>(2));
    V0   .resize(n_channels, std::vector<double>(2));
    delta.resize(n_channels, std::vector<double>(n_q_points, 0.0));

    cgs_qwell.resize(n_q_points, std::vector<std::complex<double>>(l_max + 1));
}

void pHe3DoubleGaus::InitializeParameters() {
    // Initial guesses: widths in fm, depths in MeV.
    // Short-range repulsive Gaussian + longer-range attractive Gaussian.

    // L=0, S=0  (1S0 — strong attraction)
    sigma[0] = {0.5, 1.8};   V0[0] = {200.0, -100.0};
    // L=0, S=1  (3S1 — slightly weaker attraction)
    sigma[1] = {0.5, 1.5};   V0[1] = {200.0,  -90.0};
    // L=1, S=0  (3P0 — repulsive at low k*)
    sigma[2] = {0.5, 2.0};   V0[2] = {100.0,   10.0};
    // L=1, S=1  (3P1 — repulsive, stronger)
    sigma[3] = {0.5, 2.5};   V0[3] = {100.0,   20.0};
}

// ── Complex gamma ──────────────────────────────────────────────────────────────
std::complex<double> pHe3DoubleGaus::CGamma(std::complex<double> z) {
    gsl_sf_result lnr, arg;
    gsl_sf_lngamma_complex_e(z.real(), z.imag(), &lnr, &arg);
    return std::exp(std::complex<double>(lnr.val, arg.val));
}

// ── Initialize: Coulomb phases + phase shifts ─────────────────────────────────
void pHe3DoubleGaus::Initialize() {
    // Coulomb phase factors
    for (int iq = 0; iq < n_q_points; iq++) {
        double q = std::max(iq * dq, 1e-6);
        double eta0 = q1q2 * mu * alpha / q;
        for (int l = 0; l <= l_max; l++) {
            std::complex<double> arg = (l + 1.0) + 1i * eta0;
            std::complex<double> g   = CGamma(arg);
            cgs_qwell[iq][l] = std::conj(g / std::abs(g));
        }
    }
    ComputePhaseShiftsAllChannels();
}

void pHe3DoubleGaus::LoadExperimentalData(
        const std::vector<double>& q_exp,
        const std::vector<std::vector<double>>& delta_exp,
        const std::vector<std::vector<double>>& delta_exp_error) {
    m_q_exp          = q_exp;
    m_delta_exp      = delta_exp;
    m_delta_exp_error = delta_exp_error;
}

// ── Core: numerically integrate inside the well, match to Coulomb outside ─────
void pHe3DoubleGaus::ComputePhaseShiftsAllChannels() {

    // Match radius: far enough that V(r_match) ≈ 0 for all channels.
    // Use 6 * max(sigma) as a conservative cutoff, minimum 10 fm.
    double r_match = 10.0;
    for (int ch = 0; ch < n_channels; ch++)
        r_match = std::max(r_match, 6.0 * *std::max_element(sigma[ch].begin(), sigma[ch].end()));

    const double dr   = 0.01;   // fm  — RK4 step size
    const int    n_r  = static_cast<int>(r_match / dr) + 1;
    const double r0   = 1e-4;   // fm  — start away from origin

    for (int ch = 0; ch < n_channels; ch++) {
        int l = ell[ch];

        for (int iq = 1; iq < n_q_points; iq++) {
            double q = iq * dq;

            // ── Integrate u(r) outward from r0 to r_match ──────────────────
            // Regular boundary condition at origin: u ~ r^(l+1)
            double u  = std::pow(r0, l + 1);
            double up = (l + 1) * std::pow(r0, l);

            double r = r0;
            for (int ir = 0; ir < n_r; ir++, r += dr) {
                double u_new, up_new;
                RK4Step(ch, l, q, r, dr, u, up, u_new, up_new);
                u  = u_new;
                up = up_new;
            }
            // r is now ≈ r_match

            // ── Match to Coulomb F and G at r_match ────────────────────────
            double eta0 = q1q2 * mu * alpha / q;
            double x    = r * q / Constants::HBARC;

            gsl_sf_result F_res, G_res, Ferr_res, Gerr_res;
            double eF, eG;
            gsl_sf_coulomb_wave_FG_e(eta0, x, static_cast<double>(l), 0,
                                     &F_res, &Ferr_res, &G_res, &Gerr_res, &eF, &eG);
            double F = F_res.val;
            double G = G_res.val;
            // derivatives w.r.t. x: F' = dF/dx etc.
            // Use GSL's recurrence: F'_l = (l/x)*F_l - F_{l+1} / ... 
            // Simpler: finite difference at r ± dr/2 for the numerical solution,
            // and analytic Wronskian relation for Coulomb:
            //   tan(δ) = (u'·F - u·F') / (u·G' - u'·G)   at r_match
            // where primes are w.r.t. r (absorb q/ℏc from chain rule).

            // Recompute at r - dr for a two-point derivative of u
            // (we kept only the last u, so use the derivative up directly)
            double scale = q / Constants::HBARC;  // dr/dr = (dr/dx)·(dx/dr)

            // Numerical u'/u at r_match (Logarithmic derivative)
            double log_deriv_u = up / (u + 1e-30);  // u'/u  w.r.t. r

            // Coulomb log derivatives w.r.t. r  (= scale * dF/dx / F etc.)
            // We need F and G and their r-derivatives.
            // Use a small step in x to get dF/dx numerically:
            double dx   = 1e-5;
            gsl_sf_result Fp2_res, Gp2_res, Fp2err_res, Gp2err_res;
            double eF2, eG2;
            gsl_sf_coulomb_wave_FG_e(eta0, x + dx, static_cast<double>(l), 0,
                                     &Fp2_res, &Fp2err_res, &Gp2_res, &Gp2err_res, &eF2, &eG2);
            double dFdr = scale * (Fp2_res.val - F) / dx;
            double dGdr = scale * (Gp2_res.val - G) / dx;

            // Phase shift from Wronskian matching:
            //   u = A·F + B·G  =>  tan(δ) = -B/A  (outgoing = F - iG convention)
            //   A = W(u,G)/W(F,G),  B = W(F,u)/W(F,G)
            //   W(F,G) = F·G' - G·F'  (= 1/x² by Wronskian identity, but use numeric)
            double W_FG = F * dGdr - G * dFdr;
            double A    = (u * dGdr - up * G) / W_FG;
            double B    = (up * F  - u * dFdr) / W_FG;

            delta[ch][iq] = std::atan2(-B, A);
        }
    }
}

double pHe3DoubleGaus::GetPhaseShift(int channel, double q) const {
    int iq = static_cast<int>(q / dq);
    if (iq < 0 || iq >= n_q_points - 1) {
        if (iq == n_q_points - 1) return delta[channel][iq];
        return 0.0;
    }
    double frac  = (q - iq * dq) / dq;
    double value = delta[channel][iq] * (1.0 - frac) + delta[channel][iq + 1] * frac;

    if ((channel == 0 || channel == 1) && value > 0)
        value -= Constants::PI;

    return value;
}

// ── Minuit FCN ────────────────────────────────────────────────────────────────
void pHe3DoubleGaus::FCN(int& /*npar*/, double* /*grad*/, double& fval,
                         double* par, int /*flag*/)
{
    pHe3DoubleGaus* sw = s_fit_instance;
    int ch = s_fit_channel;

    sw->sigma[ch] = { std::abs(par[0]), std::abs(par[1]) };
    sw->V0[ch]    = { par[2], par[3] };
    sw->Initialize();

    fval = 0.0;
    for (size_t i = 0; i < sw->m_q_exp.size(); ++i) {
        if (std::isnan(sw->m_delta_exp[ch][i])) continue;
        double calc = sw->GetPhaseShift(ch, sw->m_q_exp[i]) * 180.0 / Constants::PI;
        double diff = calc - sw->m_delta_exp[ch][i];
        double err  = sw->m_delta_exp_error[ch][i];
        fval += (err > 0) ? (diff * diff) / (err * err) : diff * diff;
    }
}

// ── FitToPhaseShifts ──────────────────────────────────────────────────────────
void pHe3DoubleGaus::FitToPhaseShifts(int channel, int n_iterations, int /*debug_level*/) {

    if (static_cast<int>(m_delta_exp.size()) < channel + 1) {
        std::cerr << "Error: No experimental data loaded for channel " << channel << "\n";
        return;
    }

    s_fit_instance = this;
    s_fit_channel  = channel;

    TMinuit minuit(4);
    minuit.SetFCN(FCN);
    minuit.SetPrintLevel(-1);

    const auto& s = sigma[channel];
    const auto& V = V0[channel];

    // Parameters: sigma0, sigma1, V0, V1
    minuit.DefineParameter(0, "sigma0", s[0], 0.05,  0.1,  5.0);
    minuit.DefineParameter(1, "sigma1", s[1], 0.05,  0.1,  5.0);
    minuit.DefineParameter(2, "V0",     V[0], 5.0,  -500., 500.);
    minuit.DefineParameter(3, "V1",     V[1], 5.0,  -500., 500.);

    double arglist[2] = { static_cast<double>(n_iterations), 1e-6 };
    int ierr = 0;
    minuit.mnexcm("MIGRAD",  arglist, 2, ierr);
    minuit.mnexcm("IMPROVE", arglist, 1, ierr);

    double val, err;
    for (int i = 0; i < 4; i++) {
        minuit.GetParameter(i, val, err);
        if (i < 2) sigma[channel][i] = std::abs(val);
        else       V0[channel][i-2]  = val;
    }

    double fmin, fedm, errdef;
    int nvpar, nparx, istat;
    minuit.mnstat(fmin, fedm, errdef, nvpar, nparx, istat);
    std::cout << "  χ²_min = " << fmin << "  (status " << istat << ")\n";
    std::cout << "  σ = {" << sigma[channel][0] << ", " << sigma[channel][1] << "} fm\n";
    std::cout << "  V = {" << V0[channel][0]    << ", " << V0[channel][1]    << "} MeV\n";

    Initialize();
}

// ── I/O ───────────────────────────────────────────────────────────────────────
void pHe3DoubleGaus::PrintPhaseShifts(const std::string& filename) const {
    std::ofstream file;
    std::ostream* out = &std::cout;
    if (!filename.empty()) { file.open(filename); out = &file; }

    *out << "# q (MeV/c) | delta_L0S0 | delta_L0S1 | delta_L1S0 | delta_L1S1\n";
    for (int iq = 0; iq < n_q_points; iq++) {
        *out << iq * dq;
        for (int ch = 0; ch < n_channels; ch++)
            *out << "  " << delta[ch][iq] * 180.0 / Constants::PI;
        *out << "\n";
    }
    if (file.is_open()) file.close();
}

void pHe3DoubleGaus::PlotPhaseShifts(int channel, const char* outputPdfPath) const {
    TCanvas*    c1 = new TCanvas("c1", "p-He3 Gauss Phase Shifts", 800, 600);
    TMultiGraph* mg = new TMultiGraph();

    static const char* names[4] = {
        "L=0, S=0 (^{1}S_{0})", "L=0, S=1 (^{3}S_{1})",
        "L=1, S=0 (^{3}P_{0})", "L=1, S=1 (^{3}P_{1})"
    };

    int start = (channel < 0) ? 0 : channel;
    int end   = (channel < 0) ? n_channels : channel + 1;

    for (int ch = start; ch < end; ch++) {
        TGraph* gr = new TGraph();
        for (int iq = 0; iq < n_q_points; iq++)
            gr->SetPoint(iq, iq * dq, delta[ch][iq] * 180.0 / Constants::PI);
        gr->SetLineColor(ch + 1);
        gr->SetLineWidth(2);
        gr->SetTitle(names[ch]);
        mg->Add(gr, "L");
    }

    mg->Draw("A");
    mg->SetTitle("p-He3 Gauss Phase Shifts;#it{k}* (MeV/#it{c});#delta (degrees)");
    c1->BuildLegend();
    c1->SaveAs(outputPdfPath);
}

void pHe3DoubleGaus::GetParameters(int channel, std::vector<double>& sigma_vals,
                                   std::vector<double>& V_vals) const {
    sigma_vals = sigma[channel];
    V_vals     = V0[channel];
}

void pHe3DoubleGaus::SetParameters(int channel, const std::vector<double>& sigma_vals,
                                   const std::vector<double>& V_vals) {
    sigma[channel] = sigma_vals;
    V0[channel]    = V_vals;
    Initialize();
}