#pragma once

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>

#include <TRandom3.h>
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

class pHe3SquareWell {
    
public:
    pHe3SquareWell();
    ~pHe3SquareWell();
    
    // Initialize phase shifts and wavefunctions
    void Initialize();
    
    void LoadExperimentalData(const std::vector<double>& q_exp, 
                              const std::vector<std::vector<double>>& delta_exp, 
                              const std::vector<std::vector<double>>& delta_exp_error);

    // Get phase shift for specific channel and momentum
    double GetPhaseShift(int channel, double q) const;
    
    // Fit parameters to experimental phase shifts
    void FitToPhaseShifts(int channel, int n_iterations = 1000, int debug_level = -1);
    
    // Print current phase shifts
    void PrintPhaseShifts(const std::string& filename = "") const;
    
    // Plot phase shifts
    void PlotPhaseShifts(int channel = -1, const char * outputPdfPath = "pHe3SquareWell.pdf") const;
    
    // Compute wavefunction squared including Coulomb
    double CalcPsiSquared(double q, double r, double ctheta);
    
    // Get potential parameters
    void GetParameters(int channel, std::vector<double>& a_vals, std::vector<double>& V_vals) const;
    
    // Set potential parameters
    void SetParameters(int channel, const std::vector<double>& a_vals, const std::vector<double>& V_vals);
    
private:
    // Physical constants
    double m_proton;
    double m_He3;
    double mu;        // reduced mass
    double q1q2;      // product of charges (Z1*Z2)
    double alpha;     // fine structure constant
    int n_channels;
    
    // Momentum grid
    double q_max, dq;
    int n_q_points;

    std::vector<double> m_q_exp;
    std::vector<std::vector<double>> m_delta_exp;
    std::vector<std::vector<double>> m_delta_exp_error;
    
    // Channel quantum numbers: L and S
    std::vector<int> L_values;
    std::vector<int> S_values;
    std::vector<int> ell;  // angular momentum for each channel
    int l_max;
    
    // Square well parameters for each channel
    std::vector<int> n_wells_vec;
    std::vector<std::vector<double>> a;   // radii
    std::vector<std::vector<double>> V0;  // depths (MeV)
    
    // Phase shifts: delta[channel][q_index]
    std::vector<std::vector<double>> delta;
    
    // Coefficients for wavefunction construction: A[channel][iq][coefficient_index]
    std::vector<std::vector<std::vector<std::complex<double>>>> A;
    
    // Coulomb phase factors
    std::vector<std::vector<std::complex<double>>> cgs_qwell;  // [iq][l]
    
    // Helper functions
    void InitializeParameters();
    void InitializeArrays();
    void ComputePhaseShiftsAllChannels();
    
    std::complex<double> CGamma(std::complex<double> z);
    
    // Matrix solver for matching conditions
    void SolveLinearSystem(int n, std::complex<double>** M, std::complex<double>* Y, 
                          std::complex<double>* solution);
};

pHe3SquareWell::pHe3SquareWell() {

    m_proton = Constants::M_PROTON;
    //m_He3 = 2.0 * m_proton + Constants::M_PROTON - 7.718; // Binding energy
    m_He3 = Constants::M_HELIUM3;
    mu = (m_proton * m_He3) / (m_proton + m_He3);
    
    q1q2 = Constants::Z_PROTON * Constants::Z_HELIUM3;
    alpha = Constants::ALPHA_EM;
    
    n_channels = 4;
    l_max = 1;
    
    // Initialize quantum numbers
    L_values = {0, 0, 1, 1};
    S_values = {0, 1, 0, 1};
    ell = {0, 0, 1, 1};
    
    // Number of wells per channel
    n_wells_vec = {2, 2, 2, 2};
    
    q_max = 100.0;  // MeV/c
    dq = 1.0;       // MeV/c
    n_q_points = static_cast<int>(q_max / dq) + 1;
    
    InitializeArrays();
    InitializeParameters();
}

pHe3SquareWell::~pHe3SquareWell() {
    // Cleanup is handled by std::vector destructors
}

void pHe3SquareWell::InitializeArrays() {
    // Resize arrays
    a.resize(n_channels);
    V0.resize(n_channels);
    delta.resize(n_channels);
    A.resize(n_channels);
    
    for (int ichannel = 0; ichannel < n_channels; ichannel++) {
        int n_wells = n_wells_vec[ichannel];
        a[ichannel].resize(n_wells);
        V0[ichannel].resize(n_wells);
        delta[ichannel].resize(n_q_points, 0.0);
        
        A[ichannel].resize(n_q_points);
        for (int iq = 0; iq < n_q_points; iq++) {
            A[ichannel][iq].resize(2 * n_wells);
        }
    }
    
    // Initialize Coulomb phase factors
    cgs_qwell.resize(n_q_points);
    for (int iq = 0; iq < n_q_points; iq++) {
        cgs_qwell[iq].resize(l_max + 1);
    }
}

void pHe3SquareWell::InitializeParameters() {
    // From Scott Pratt's fitted values
    
    // L=0, S=0 (singlet)
    a[0] = {1.80628, 2.05664};
    V0[0] = {-2.71804, -114.483};
    
    // L=0, S=1 (triplet)
    a[1] = {0.835825, 1.10527};
    V0[1] = {-6.16757, -155.224};
    
    // L=1, S=0 (singlet)
    a[2] = {1.10123, 6.55019};
    V0[2] = {11.5845, -1.51854};
    
    // L=1, S=1 (triplet)
    a[3] = {0.0956635, 5.23552};
    V0[3] = {31.0429, -4.76129};
}

// Complex gamma function
std::complex<double> pHe3SquareWell::CGamma(std::complex<double> z) {
    gsl_sf_result lnr, arg;
    gsl_sf_lngamma_complex_e(z.real(), z.imag(), &lnr, &arg);
    return std::exp(std::complex<double>(lnr.val, arg.val));
}

// Solve linear system using GSL
void pHe3SquareWell::SolveLinearSystem(int n, std::complex<double>** M, 
                                       std::complex<double>* Y, 
                                       std::complex<double>* solution) {
    // Convert to GSL format
    gsl_matrix_complex* A_gsl = gsl_matrix_complex_alloc(n, n);
    gsl_vector_complex* b_gsl = gsl_vector_complex_alloc(n);
    gsl_vector_complex* x_gsl = gsl_vector_complex_alloc(n);
    gsl_permutation* p = gsl_permutation_alloc(n);
    
    for (int i = 0; i < n; i++) {
        gsl_complex y_val = {{Y[i].real(), Y[i].imag()}};
        gsl_vector_complex_set(b_gsl, i, y_val);
        
        for (int j = 0; j < n; j++) {
            gsl_complex m_val = {{M[i][j].real(), M[i][j].imag()}};
            gsl_matrix_complex_set(A_gsl, i, j, m_val);
        }
    }
    
    int signum;
    gsl_linalg_complex_LU_decomp(A_gsl, p, &signum);
    gsl_linalg_complex_LU_solve(A_gsl, p, b_gsl, x_gsl);
    
    for (int i = 0; i < n; i++) {
        gsl_complex x_val = gsl_vector_complex_get(x_gsl, i);
        solution[i] = std::complex<double>(GSL_REAL(x_val), GSL_IMAG(x_val));
    }
    
    gsl_matrix_complex_free(A_gsl);
    gsl_vector_complex_free(b_gsl);
    gsl_vector_complex_free(x_gsl);
    gsl_permutation_free(p);
}

void pHe3SquareWell::Initialize() {
    // Compute Coulomb phase factors
    for (int iq = 0; iq < n_q_points; iq++) {
        double q = iq * dq;
        if (q < 1e-6) q = 1e-6;
        double eta0 = q1q2 * mu * alpha / q;
        
        for (int l = 0; l <= l_max; l++) {
            std::complex<double> arg = (l + 1.0) + 1i * eta0;
            std::complex<double> gamma_val = CGamma(arg);
            cgs_qwell[iq][l] = std::conj(gamma_val / std::abs(gamma_val));
        }
    }
    
    ComputePhaseShiftsAllChannels();
}

void pHe3SquareWell::LoadExperimentalData(const std::vector<double>& q_exp, 
                                      const std::vector<std::vector<double>>& delta_exp, 
                                      const std::vector<std::vector<double>>& delta_exp_error) {
    m_q_exp = q_exp;
    m_delta_exp = delta_exp;
    m_delta_exp_error = delta_exp_error;
}

void pHe3SquareWell::ComputePhaseShiftsAllChannels() {
    // Main loop over channels and momenta (following Pratt's SquareWell_Init)
    
    for (int ichannel = 0; ichannel < n_channels; ichannel++) {
        int n_wells = n_wells_vec[ichannel];
        int l = ell[ichannel];
        
        if (n_wells == 2) {
            // Two-well case
            auto cmatrix = new CGSLMatrix_Complex(4);
            std::complex<double>** M = new std::complex<double>*[4];
            for (int i = 0; i < 4; i++) {
                M[i] = new std::complex<double>[4];
            }
            std::complex<double>* Y = new std::complex<double>[4];
            std::complex<double>* Avec = new std::complex<double>[4];
            
            for (int iq = 1; iq < n_q_points; iq++) {
                double q = iq * dq;
                if (q < 1e-6) q = 1e-6;
                
                // Wave numbers in each region
                double qsq1 = q*q - 2.0*mu*V0[ichannel][0];
                std::complex<double> q1 = (qsq1 > 0) ? std::sqrt(qsq1) : 1i*std::sqrt(std::abs(qsq1));
                
                double qsq2 = q*q - 2.0*mu*V0[ichannel][1];
                std::complex<double> q2 = (qsq2 > 0) ? std::sqrt(qsq2) : 1i*std::sqrt(std::abs(qsq2));
                
                // Positions times wave numbers
                std::complex<double> x1b = a[ichannel][0] * q1 / Constants::HBARC;
                std::complex<double> x2a = a[ichannel][0] * q2 / Constants::HBARC;
                std::complex<double> x2b = a[ichannel][1] * q2 / Constants::HBARC;
                std::complex<double> x = a[ichannel][1] * q / Constants::HBARC;
                
                // Sommerfeld parameters
                std::complex<double> eta1 = q1q2 * mu * alpha / q1;
                std::complex<double> eta2 = q1q2 * mu * alpha / q2;
                std::complex<double> eta0 = q1q2 * mu * alpha / q;
                
                // Get Coulomb functions at boundaries
                double F1b, G1b, F1bp, G1bp;
                double F2a, G2a, F2ap, G2ap;
                double F2b, G2b, F2bp, G2bp;
                double F, G, Fp, Gp;
                
                math::FGprime_ComplexQ(l, x1b, eta1, &F1b, &G1b, &F1bp, &G1bp);
                math::FGprime_ComplexQ(l, x2a, eta2, &F2a, &G2a, &F2ap, &G2ap);
                math::FGprime_ComplexQ(l, x2b, eta2, &F2b, &G2b, &F2bp, &G2bp);
                math::FGprime_ComplexQ(l, x, eta0, &F, &G, &Fp, &Gp);
                
                // Build matrix (matching boundary conditions)
                M[0][0] = F1b;                  M[0][1] = -F2a;                 M[0][2] = -G2a;                 M[0][3] = 0.0;
                M[1][0] = std::abs(q1)*F1bp;    M[1][1] = -std::abs(q2)*F2ap;   M[1][2] = -std::abs(q2)*G2ap;   M[1][3] = 0.0;
                M[2][0] = 0.0;                  M[2][1] = F2b;                  M[2][2] = G2b;                  M[2][3] = -0.5*(F + 1i*G);
                M[3][0] = 0.0;                  M[3][1] = std::abs(q2)*F2bp;    M[3][2] = std::abs(q2)*G2bp;    M[3][3] = -0.5*q*(Fp + 1i*Gp);
                
                Y[0] = 0.0; Y[1] = 0.0;
                Y[2] = 0.5*(F - 1i*G);
                Y[3] = 0.5*q*(Fp - 1i*Gp);
                
                cmatrix->SolveLinearEqs(Y, M, Avec);
                //SolveLinearSystem(4, M, Y, Avec);
                
                for (int ia = 0; ia < 4; ia++) {
                    A[ichannel][iq][ia] = Avec[ia];
                }
                
                // Extract phase shift
                delta[ichannel][iq] = -0.5 * std::atan2(A[ichannel][iq][3].imag(), 
                                                         A[ichannel][iq][3].real());
                if (delta[ichannel][iq] < 0.0) {
                    delta[ichannel][iq] += Constants::PI;
                }
            }
            
            for (int i = 0; i < 4; i++) delete[] M[i];
            delete[] M;
            delete[] Y;
            delete[] Avec;
        }
        else {
            std::cerr << "Only 2-well potentials currently implemented" << std::endl;
        }
    }
}

double pHe3SquareWell::GetPhaseShift(int channel, double q) const {
    int iq = static_cast<int>(q / dq);
    if (iq < 0 || iq >= n_q_points - 1) {
        if (iq == n_q_points - 1) return delta[channel][iq];
        return 0.0;
    }
    
    // Linear interpolation
    double q_low = iq * dq;
    double frac = (q - q_low) / dq;

    double delta_value = delta[channel][iq] * (1.0 - frac) + delta[channel][iq + 1] * frac;

    if ((channel == 0 | channel == 1) && delta_value > 0)
        delta_value -= Constants::PI;

    return delta_value;
}

void pHe3SquareWell::FitToPhaseShifts(int channel, int n_iterations, int debug_level) {
    
    std::vector<double> q_targets, delta_targets, delta_targets_errors;

    if (m_delta_exp.size() < channel + 1) {
        std::cerr << "Error: No experimental data loaded for channel 1" << std::endl;
        return;
    }
    
    q_targets = m_q_exp;
    delta_targets = m_delta_exp[channel];
    delta_targets_errors = m_delta_exp_error[channel];
        
    double best_error = 1e99;
    std::vector<double> best_a = a[channel];
    std::vector<double> best_V = V0[channel];
    
    double dela = 0.03;
    double delV = 0.5;
    int n_miss = 0;
    
    const int print_step = std::max(1, n_iterations / 100);

    for (int iter = 0; iter < n_iterations; iter++) {
        n_miss++;
        
        if (iter > 0) {
            for (size_t i = 0; i < a[channel].size(); i++) {
                a[channel][i] = std::abs(best_a[i] + dela * gRandom->Gaus());
                V0[channel][i] = best_V[i] + delV * gRandom->Gaus();
            }
        }
        
        Initialize();
        
        double error = 0.0;
        for (size_t i = 0; i < q_targets.size(); i++) {
            double delta_calc = GetPhaseShift(channel, q_targets[i]) * 180.0 / Constants::PI;
            if ((channel == 0 | channel == 1) && delta_calc > 0)
                delta_calc -= 180;
            error += std::pow(delta_calc - delta_targets[i], 2) / std::pow(delta_targets_errors[i], 2);
        }
        
        if (error < best_error) {
            best_error = error;
            best_a = a[channel];
            best_V = V0[channel];
            
            std::cout << "Iteration " << iter << ": error = " << error << std::endl;
            std::cout << "  a = {";
            for (size_t i = 0; i < a[channel].size(); i++) {
                std::cout << a[channel][i];
                if (i < a[channel].size()-1) std::cout << ", ";
            }
            std::cout << "}" << std::endl;
            std::cout << "  V = {";
            for (size_t i = 0; i < V0[channel].size(); i++) {
                std::cout << V0[channel][i];
                if (i < V0[channel].size()-1) std::cout << ", ";
            }
            std::cout << "}" << std::endl;
            
            n_miss = 0;
        }

        if (debug_level > 0) {
            if (iter % print_step == 0) {
                std::cout << "Iteration " << iter << ": error = " << error << std::endl;
                for (size_t i = 0; i < best_a.size(); i++) {
                    std::cout << best_a[i];
                    if (i < best_a.size()-1) std::cout << ", ";
                }
                std::cout << "}" << std::endl;
                std::cout << "  V = {";
                for (size_t i = 0; i < best_V.size(); i++) {
                    std::cout << best_V[i];
                    if (i < best_V.size()-1) std::cout << ", ";
                }
                std::cout << "}" << std::endl;        
            }
        }
            
        
        if (n_miss > 50) {
            dela *= 0.8;
            delV *= 0.8;
            n_miss = 0;
        }
    }
    
    a[channel] = best_a;
    V0[channel] = best_V;
    Initialize();
}

void pHe3SquareWell::PrintPhaseShifts(const std::string& filename) const {

    std::ofstream file;
    std::ostream* out = &std::cout;
    
    if (!filename.empty()) {
        file.open(filename);
        out = &file;
    }
    
    *out << "# q (MeV/c) | delta_L0S0 | delta_L0S1 | delta_L1S0 | delta_L1S1" << std::endl;
    
    for (int iq = 0; iq < n_q_points; iq++) {
        double q = iq * dq;
        *out << q;
        for (int channel = 0; channel < n_channels; channel++) {
            *out << " " << delta[channel][iq] * 180.0 / Constants::PI;
        }
        *out << std::endl;
    }
    
    if (file.is_open()) file.close();
}

void pHe3SquareWell::PlotPhaseShifts(int channel, const char * outputPdfPath) const {
    TCanvas* c1 = new TCanvas("c1", "p-He3 Phase Shifts", 800, 600);
    TMultiGraph* mg = new TMultiGraph();
    
    std::vector<std::string> channel_names = {
        "L=0, S=0 (singlet)", 
        "L=0, S=1 (triplet)", 
        "L=1, S=0 (singlet)", 
        "L=1, S=1 (triplet)"
    };
    
    int start = (channel < 0) ? 0 : channel;
    int end = (channel < 0) ? n_channels : channel + 1;
    
    for (int ch = start; ch < end; ch++) {
        TGraph* gr = new TGraph();
        for (int iq = 0; iq < n_q_points; iq++) {
            double q = iq * dq;
            double phase = delta[ch][iq] * 180.0 / Constants::PI;
            gr->SetPoint(iq, q, phase);
        }
        gr->SetLineColor(ch + 1);
        gr->SetLineWidth(2);
        gr->SetTitle(channel_names[ch].c_str());
        mg->Add(gr, "L");
    }
    
    mg->Draw("A");
    mg->SetTitle("p-He3 Phase Shifts;q (MeV/c);#delta (degrees)");
    
    c1->BuildLegend();
    c1->Draw();

    c1->SaveAs(outputPdfPath);
}

void pHe3SquareWell::GetParameters(int channel, std::vector<double>& a_vals, 
                                   std::vector<double>& V_vals) const {
    a_vals = a[channel];
    V_vals = V0[channel];
}

void pHe3SquareWell::SetParameters(int channel, const std::vector<double>& a_vals, 
                                   const std::vector<double>& V_vals) {
    a[channel] = a_vals;
    V0[channel] = V_vals;
    Initialize();
}