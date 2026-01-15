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

#include <Eigen/Dense>

#include "Constants.h"
#include "Physics.h"

using namespace std::complex_literals;



class pdSquareWell {
    
public:
    pdSquareWell();
    ~pdSquareWell();
    
    // Initialize phase shifts and wavefunctions
    void Initialize();
    
    // Get phase shift for specific channel and momentum
    double GetPhaseShift(int channel, double q) const;
    
    // Get scattering length for specific channel
    double GetScatteringLength(int channel) const;
    
    // Fit parameters to target phase shifts
    void FitToPhaseShifts(int channel, int n_iterations = 10000, int debug_level = -1);
    
    // Print current phase shifts
    void PrintPhaseShifts(const std::string& filename = "") const;
    
    // Plot phase shifts
    void PlotPhaseShifts(int channel = -1, const char* outputPdfPath = "pdSquareWell.pdf") const;
    
    // Get potential parameters
    void GetParameters(int channel, std::vector<double>& a_vals, std::vector<double>& V_vals) const;
    
    // Set potential parameters
    void SetParameters(int channel, const std::vector<double>& a_vals, const std::vector<double>& V_vals);
    
private:
    // Physical constants
    double m_proton;
    double m_deuteron;
    double mu;        // reduced mass
    double q1q2;      // product of charges (Z1*Z2) = 0 for p-d
    int n_channels;
    
    // Momentum grid
    double q_max, dq;
    int n_q_points;
    
    // Channel quantum numbers: L and S
    std::vector<int> ell;  // angular momentum for each channel
    int l_max;
    
    // Square well parameters for each channel
    std::vector<int> n_wells_vec;
    std::vector<std::vector<double>> a;   // radii (fm)
    std::vector<std::vector<double>> V0;  // depths (MeV)
    
    // Phase shifts: delta[channel][q_index]
    std::vector<std::vector<double>> delta;
    
    // Scattering lengths
    std::vector<double> scatt_length;
    
    // Binding energy for s=1/2 channel (triton)
    double BE_s12;
    
    // Helper functions
    void InitializeParameters();
    void InitializeArrays();
    void ComputePhaseShifts(int channel);
    
    // Solve for V0[0] given binding energy
    bool FixV0(double BE, int channel);
    
    // Invert tan(x)/x or tanh(x)/x equation
    bool InvertTanXoverX(double target, double& x);
};

pdSquareWell::pdSquareWell() {
    m_proton = Constants::M_PROTON;
    m_deuteron = Constants::M_DEUTERON; // Deuteron binding energy
    mu = (m_proton * m_deuteron) / (m_proton + m_deuteron);
    
    q1q2 = 0.0;  // No Coulomb for p-d
    
    n_channels = 2;  // s=1/2 and s=3/2
    l_max = 0;       // Only s-wave for now
    
    // Channel 0: s=1/2 (doublet), Channel 1: s=3/2 (quartet)
    ell = {0, 0};
    
    // Number of wells per channel
    // s=1/2 has 3 wells (with bound state), s=3/2 has 1 well
    n_wells_vec = {3, 1};
    
    // Binding energy for triton (s=1/2 channel)
    BE_s12 = 8.481;  // MeV (triton binding energy)
    
    q_max = 160.0;  // MeV/c (matching NQMAX*2*DELQ from original)
    dq = 2.0;       // MeV/c
    n_q_points = static_cast<int>(q_max / dq);
    
    InitializeArrays();
    InitializeParameters();
}

pdSquareWell::~pdSquareWell() {
    // Cleanup handled by std::vector destructors
}

void pdSquareWell::InitializeArrays() {
    a.resize(n_channels);
    V0.resize(n_channels);
    delta.resize(n_channels);
    scatt_length.resize(n_channels, 0.0);
    
    for (int ichannel = 0; ichannel < n_channels; ichannel++) {
        int n_wells = n_wells_vec[ichannel];
        a[ichannel].resize(n_wells);
        V0[ichannel].resize(n_wells);
        delta[ichannel].resize(n_q_points, 0.0);
    }
}

void pdSquareWell::InitializeParameters() {
    // Initial parameters (these will be fitted)
    // Channel 0: s=1/2 (3 wells with bound state)
    a[0] = {1.5, 2.0, 3.0};
    V0[0] = {0.0, 50.0, 5.0};  // V0[0] will be fixed by binding energy
    
    // Channel 1: s=3/2 (1 well, no bound state)
    a[1] = {2.0};
    V0[1] = {50.0};
}

bool pdSquareWell::InvertTanXoverX(double target, double& x) {
    bool success = true;
    double y, dydx, missby = 1.0E10;
    int ntry = 0;
    
    if (target >= 0.0 && target <= 1.0) {
        // tanh(x)/x regime
        x = 0.5;
        y = std::tanh(x) / x;
        while (std::abs(missby) > 1.0E-10 && ntry < 100) {
            ntry++;
            dydx = (1.0/x) * std::pow(std::cosh(x), -2) - y/x;
            double delx = (target - y) / dydx;
            if (std::abs(delx) > 0.5 * std::abs(x))
                delx = (0.5 * std::abs(x)) * std::abs(delx) / delx;
            x = x + delx;
            y = std::tanh(x) / x;
            missby = y - target;
        }
    }
    else if (target >= 1.0) {
        // tan(x)/x regime, 0 < x < pi/2
        x = 0.25 * Constants::PI;
        y = std::tan(x) / x;
        missby = y - target;
        while (std::abs(missby) > 1.0E-10 && ntry < 100) {
            ntry++;
            dydx = (1.0/x) * std::pow(std::cos(x), -2) - y/x;
            double delx = (target - y) / dydx;
            if (x + delx > 0.5 * Constants::PI)
                delx = 0.75 * (0.5 * Constants::PI - x);
            if (x + delx < 0.0)
                delx = -0.25 * x;
            x = x + delx;
            y = std::tan(x) / x;
            missby = y - target;
        }
    }
    else if (target <= 0.0) {
        // tan(x)/x regime, pi/2 < x < pi
        x = 0.75 * Constants::PI;
        y = std::tan(x) / x;
        missby = y - target;
        while (std::abs(missby) > 1.0E-10 && ntry < 100) {
            ntry++;
            dydx = (1.0/x) * std::pow(std::cos(x), -2) - y/x;
            double delx = (target - y) / dydx;
            if (x + delx > Constants::PI)
                delx = 0.75 * (Constants::PI - x);
            if (x + delx < 0.5 * Constants::PI)
                delx = -0.25 * (x - 0.5 * Constants::PI);
            x = x + delx;
            y = std::tan(x) / x;
            missby = y - target;
        }
    }
    
    if (ntry == 100) {
        success = false;
    }
    
    return success;
}

bool pdSquareWell::FixV0(double BE, int channel) {
    int n_wells = n_wells_vec[channel];
    std::vector<double> q(n_wells + 1), KE(n_wells + 1);
    std::vector<double> A(n_wells), B(n_wells);
    std::vector<double> psi(n_wells), psiprime(n_wells);
    
    // Calculate wave numbers in each region
    for (int iwell = 0; iwell < n_wells; iwell++) {
        KE[iwell] = -BE - V0[channel][iwell];
        q[iwell] = std::sqrt(2.0 * mu * std::abs(KE[iwell])) / Constants::HBARC;
    }
    
    // Boundary condition at infinity
    psi[n_wells - 1] = 1.0;
    psiprime[n_wells - 1] = -std::sqrt(2.0 * mu * BE) / Constants::HBARC;
    
    // Work backwards from outer boundary
    for (int iwell = n_wells - 1; iwell > 0; iwell--) {
        double c, s;
        if (KE[iwell] > 0.0) {
            // Oscillatory region
            c = std::cos(q[iwell] * a[channel][iwell]);
            s = std::sin(q[iwell] * a[channel][iwell]);
            A[iwell] = psi[iwell] * c - (psiprime[iwell] / q[iwell]) * s;
            B[iwell] = (psiprime[iwell] / q[iwell]) * c + psi[iwell] * s;
            
            c = std::cos(q[iwell] * a[channel][iwell - 1]);
            s = std::sin(q[iwell] * a[channel][iwell - 1]);
            psi[iwell - 1] = A[iwell] * c + B[iwell] * s;
            psiprime[iwell - 1] = q[iwell] * (-A[iwell] * s + B[iwell] * c);
        }
        else {
            // Exponential region
            c = std::cosh(q[iwell] * a[channel][iwell]);
            s = std::sinh(q[iwell] * a[channel][iwell]);
            A[iwell] = psi[iwell] * c - (psiprime[iwell] / q[iwell]) * s;
            B[iwell] = (psiprime[iwell] / q[iwell]) * c - psi[iwell] * s;
            
            c = std::cosh(q[iwell] * a[channel][iwell - 1]);
            s = std::sinh(q[iwell] * a[channel][iwell - 1]);
            psi[iwell - 1] = A[iwell] * c + B[iwell] * s;
            psiprime[iwell - 1] = q[iwell] * (A[iwell] * s + B[iwell] * c);
        }
    }
    
    // Solve for innermost potential
    A[0] = 0.0;
    double target = psi[0] / (psiprime[0] * a[channel][0]);
    double x;
    bool success = InvertTanXoverX(target, x);
    
    if (success) {
        q[0] = x / a[channel][0];
        if (target > 0.0 && target < 1.0) {
            KE[0] = -0.5 * q[0] * q[0] * Constants::HBARC * Constants::HBARC / mu;
            B[0] = psi[0] / std::sinh(q[0] * a[channel][0]);
        }
        else {
            KE[0] = 0.5 * q[0] * q[0] * Constants::HBARC * Constants::HBARC / mu;
            B[0] = psi[0] / std::sin(q[0] * a[channel][0]);
        }
        V0[channel][0] = -BE - KE[0];
        
        // Check for nodes
        int nnodes = 0;
        double delr = 0.1;
        double oldphi = 0.0;
        for (double r = 0.5 * delr; r < a[channel][n_wells - 1]; r += delr) {
            int iwell = 0;
            while (r > a[channel][iwell])
                iwell++;
            
            double phi;
            if (KE[iwell] < 0.0) {
                phi = A[iwell] * std::cosh(q[iwell] * r) + B[iwell] * std::sinh(q[iwell] * r);
            }
            else {
                phi = A[iwell] * std::cos(q[iwell] * r) + B[iwell] * std::sin(q[iwell] * r);
            }
            
            if (r > delr && oldphi * phi < 0)
                nnodes++;
            oldphi = phi;
        }
        
        if (nnodes > 0 || std::abs(V0[channel][0]) > 1000.0) {
            success = false;
        }
    }
    
    return success;
}

void pdSquareWell::ComputePhaseShifts(int channel) {
    int n_wells = n_wells_vec[channel];
    int l = ell[channel];
    
    if (n_wells == 1) {
        // Single well case (for s=3/2)
        for (int iq = 0; iq < n_q_points; iq++) {
            double q = (iq + 0.5) * dq;
            
            double qsquared = q * q - 2.0 * mu * V0[channel][0];
            std::complex<double> q1;
            if (qsquared > 0)
                q1 = std::sqrt(qsquared);
            else
                q1 = 1i * std::sqrt(std::abs(qsquared));
            
            std::complex<double> x1 = q1 * a[channel][0] / Constants::HBARC;
            std::complex<double> x2 = q * a[channel][0] / Constants::HBARC;
            std::complex<double> eta1 = 0.0;  // No Coulomb
            std::complex<double> eta0 = 0.0;
            
            double F1, G1, F1prime, G1prime;
            double F2, G2, F2prime, G2prime;
            
            math::FGprime_ComplexQ(l, x1, eta1, &F1, &G1, &F1prime, &G1prime);
            math::FGprime_ComplexQ(l, x2, eta0, &F2, &G2, &F2prime, &G2prime);
            
            double beta = (std::abs(q1) / q) * F1prime / F1;
            delta[channel][iq] = -std::atan2(beta * F2 - F2prime, beta * G2 - G2prime);
        }
    }
    else if (n_wells == 2) {
        // Two-well case
        Eigen::VectorXcd Y(4), Avec(4);
        Eigen::MatrixXcd M(4, 4);
        
        for (int iq = 0; iq < n_q_points; iq++) {
            double q = (iq + 0.5) * dq;
            
            std::complex<double> q1 = std::sqrt(std::abs(q*q - 2.0*mu*V0[channel][0]));
            if (q*q - 2.0*mu*V0[channel][0] < 0.0)
                q1 = 1i * q1;
            
            std::complex<double> q2 = std::sqrt(std::abs(q*q - 2.0*mu*V0[channel][1]));
            if (q*q - 2.0*mu*V0[channel][1] < 0.0)
                q2 = 1i * q2;
            
            std::complex<double> x1b = a[channel][0] * q1 / Constants::HBARC;
            std::complex<double> x2a = a[channel][0] * q2 / Constants::HBARC;
            std::complex<double> x2b = a[channel][1] * q2 / Constants::HBARC;
            std::complex<double> x = a[channel][1] * q / Constants::HBARC;
            
            std::complex<double> eta1 = 0.0, eta2 = 0.0, eta0 = 0.0;
            
            double F1b, G1b, F1bp, G1bp;
            double F2a, G2a, F2ap, G2ap;
            double F2b, G2b, F2bp, G2bp;
            double F, G, Fp, Gp;
            
            math::FGprime_ComplexQ(l, x1b, eta1, &F1b, &G1b, &F1bp, &G1bp);
            math::FGprime_ComplexQ(l, x2a, eta2, &F2a, &G2a, &F2ap, &G2ap);
            math::FGprime_ComplexQ(l, x2b, eta2, &F2b, &G2b, &F2bp, &G2bp);
            math::FGprime_ComplexQ(l, x, eta0, &F, &G, &Fp, &Gp);
            
            M(0,0) = F1b;                     M(0,1) = -F2a;                    M(0,2) = -G2a;                    M(0,3) = 0.0;
            M(1,0) = std::abs(q1)*F1bp;       M(1,1) = -std::abs(q2)*F2ap;      M(1,2) = -std::abs(q2)*G2ap;      M(1,3) = 0.0;
            M(2,0) = 0.0;                     M(2,1) = F2b;                     M(2,2) = G2b;                     M(2,3) = -0.5*(F + 1i*G);
            M(3,0) = 0.0;                     M(3,1) = std::abs(q2)*F2bp;       M(3,2) = std::abs(q2)*G2bp;       M(3,3) = -0.5*q*(Fp + 1i*Gp);
            
            Y[0] = 0.0; Y[1] = 0.0;
            Y[2] = 0.5*(F - 1i*G);
            Y[3] = 0.5*q*(Fp - 1i*Gp);
            
            Avec = M.colPivHouseholderQr().solve(Y);
            
            delta[channel][iq] = -0.5 * std::atan2(Avec(3).imag(), Avec(3).real());
            if (delta[channel][iq] < -Constants::PI)
                delta[channel][iq] += Constants::PI;
        }
    }
    else if (n_wells == 3) {
        // Three-well case (for s=1/2 with bound state)
        Eigen::VectorXcd Y(6), Avec(6);
        Eigen::MatrixXcd M(6, 6);
        
        for (int iq = 0; iq < n_q_points; iq++) {
            double q = (iq + 0.5) * dq;
            
            std::complex<double> q1 = std::sqrt(std::abs(q*q - 2.0*mu*V0[channel][0]));
            if (q*q - 2.0*mu*V0[channel][0] < 0.0) q1 = 1i * q1;
            
            std::complex<double> q2 = std::sqrt(std::abs(q*q - 2.0*mu*V0[channel][1]));
            if (q*q - 2.0*mu*V0[channel][1] < 0.0) q2 = 1i * q2;
            
            std::complex<double> q3 = std::sqrt(std::abs(q*q - 2.0*mu*V0[channel][2]));
            if (q*q - 2.0*mu*V0[channel][2] < 0.0) q3 = 1i * q3;
            
            std::complex<double> x1b = a[channel][0] * q1 / Constants::HBARC;
            std::complex<double> x2a = a[channel][0] * q2 / Constants::HBARC;
            std::complex<double> x2b = a[channel][1] * q2 / Constants::HBARC;
            std::complex<double> x3a = a[channel][1] * q3 / Constants::HBARC;
            std::complex<double> x3b = a[channel][2] * q3 / Constants::HBARC;
            std::complex<double> x = a[channel][2] * q / Constants::HBARC;
            
            std::complex<double> eta1 = 0.0, eta2 = 0.0, eta3 = 0.0, eta0 = 0.0;
            
            double F1b, G1b, F1bp, G1bp;
            double F2a, G2a, F2ap, G2ap;
            double F2b, G2b, F2bp, G2bp;
            double F3a, G3a, F3ap, G3ap;
            double F3b, G3b, F3bp, G3bp;
            double F, G, Fp, Gp;
            
            math::FGprime_ComplexQ(l, x1b, eta1, &F1b, &G1b, &F1bp, &G1bp);
            math::FGprime_ComplexQ(l, x2a, eta2, &F2a, &G2a, &F2ap, &G2ap);
            math::FGprime_ComplexQ(l, x2b, eta2, &F2b, &G2b, &F2bp, &G2bp);
            math::FGprime_ComplexQ(l, x3a, eta3, &F3a, &G3a, &F3ap, &G3ap);
            math::FGprime_ComplexQ(l, x3b, eta3, &F3b, &G3b, &F3bp, &G3bp);
            math::FGprime_ComplexQ(l, x, eta0, &F, &G, &Fp, &Gp);
            
            M(0,0) = F1b;                     M(0,1) = -F2a;                    M(0,2) = -G2a;                    M(0,3) = 0.0;             M(0,4) = 0.0;             M(0,5) = 0.0;
            M(1,0) = std::abs(q1)*F1bp;       M(1,1) = -std::abs(q2)*F2ap;      M(1,2) = -std::abs(q2)*G2ap;      M(1,3) = 0.0;             M(1,4) = 0.0;             M(1,5) = 0.0;
            M(2,0) = 0.0;                     M(2,1) = F2b;                     M(2,2) = G2b;                     M(2,3) = -F3a;            M(2,4) = -G3a;            M(2,5) = 0.0;
            M(3,0) = 0.0;                     M(3,1) = std::abs(q2)*F2bp;       M(3,2) = std::abs(q2)*G2bp;       M(3,3) = -std::abs(q3)*F3ap; M(3,4) = -std::abs(q3)*G3ap; M(3,5) = 0.0;
            M(4,0) = 0.0;                     M(4,1) = 0.0;                     M(4,2) = 0.0;                     M(4,3) = F3b;             M(4,4) = G3b;             M(4,5) = -0.5*(F + 1i*G);
            M(5,0) = 0.0;                     M(5,1) = 0.0;                     M(5,2) = 0.0;                     M(5,3) = std::abs(q3)*F3bp; M(5,4) = std::abs(q3)*G3bp; M(5,5) = -0.5*q*(Fp + 1i*Gp);
            
            Y[0] = 0.0; Y[1] = 0.0; Y[2] = 0.0; Y[3] = 0.0;
            Y[4] = 0.5*(F - 1i*G);
            Y[5] = 0.5*q*(Fp - 1i*Gp);
            
            Avec = M.colPivHouseholderQr().solve(Y);
            
            delta[channel][iq] = -0.5 * std::atan2(Avec(5).imag(), Avec(5).real());
            if (delta[channel][iq] < -Constants::PI)
                delta[channel][iq] += Constants::PI;
        }
    }
    
    // Calculate scattering length
    scatt_length[channel] = -(delta[channel][0] / (0.5 * dq)) * Constants::HBARC;
}

void pdSquareWell::Initialize() {
    for (int channel = 0; channel < n_channels; channel++) {
        ComputePhaseShifts(channel);
    }
}

double pdSquareWell::GetPhaseShift(int channel, double q) const {
    int iq = static_cast<int>(q / dq);
    if (iq < 0 || iq >= n_q_points - 1) {
        if (iq == n_q_points - 1) return delta[channel][iq];
        return 0.0;
    }
    
    double q_low = iq * dq;
    double frac = (q - q_low) / dq;
    return delta[channel][iq] * (1.0 - frac) + delta[channel][iq + 1] * frac;
}

double pdSquareWell::GetScatteringLength(int channel) const {
    return scatt_length[channel];
}

void pdSquareWell::FitToPhaseShifts(int channel, int n_iterations, int debug_level) {
    // Target phase shifts and scattering lengths
    std::vector<double> q_targets(3), Ep_targets;
    std::vector<double> delta_targets;
    double scatt_target;
    
    if (channel == 0) {
        // s=1/2 channel targets
        Ep_targets = {1.0, 2.0, 3.0};
        for (int i = 0; i < Ep_targets.size(); i++) {
            double Ep = Ep_targets[i];
            double q = computeKstarFromEprojectile(Ep, m_proton, m_deuteron);
            q_targets[i] = q;
        }
        delta_targets = {-4.5, -20.0, -27.5};  // degrees
        scatt_target = -0.13;  // fm
    }
    else {
        // s=3/2 channel targets
        Ep_targets = {1.0, 2.0, 3.0};
        for (int i = 0; i < Ep_targets.size(); i++) {
            double Ep = Ep_targets[i];
            double q = computeKstarFromEprojectile(Ep, m_proton, m_deuteron);
            q_targets[i] = q;
        }
        delta_targets = {-37.5, -52.5, -64.0};  // degrees
        scatt_target = 0.0;  // Not used in tune_pd_s32.cc
    }
    
    double best_chi = 1.0E10;
    std::vector<double> best_a = a[channel];
    std::vector<double> best_V = V0[channel];
    
    TRandom3 randy;
    randy.SetSeed(12345);
    
    int n_wells = n_wells_vec[channel];
    
    for (int itry = 0; itry < n_iterations; itry++) {
        bool success = true;
        
        // Generate random parameters
        if (itry < 100000) {
            if (channel == 0) {
                // s=1/2 channel (3 wells)
                V0[channel][0] = -100.0;  // Will be fixed by binding energy
                a[channel][0] = 0.2 + 3.5 * randy.Rndm();
                for (int iwell = 1; iwell < n_wells; iwell++) {
                    a[channel][iwell] = a[channel][iwell - 1] + 3.5 * randy.Rndm();
                }
                V0[channel][1] = 40 + 100.0 * (0.5 - randy.Rndm());
                V0[channel][2] = -5. + 10.0 * (1.0 - 2.0 * randy.Rndm());
            }
            else {
                // s=3/2 channel (1 well)
                V0[channel][0] = 100.0 + 50. * (1. - 2. * randy.Rndm());
                a[channel][0] = 5. + 3.5 * (1. - 2. * randy.Rndm());
            }
        }
        else {
            // Refine around best parameters
            do {
                a[channel][0] = std::abs(best_a[0] + 0.2 * randy.Rndm());
            } while (a[channel][0] < 0.2);
            
            if (channel == 0) {
                V0[channel][0] = 0.0;
                for (int iwell = 1; iwell < n_wells; iwell++) {
                    do {
                        a[channel][iwell] = best_a[iwell] + 0.25 * randy.Rndm();
                    } while (a[channel][iwell] < a[channel][iwell - 1]);
                    V0[channel][iwell] = best_V[iwell] + randy.Rndm();
                }
            }
            else {
                V0[channel][0] = best_V[0] + randy.Rndm();
            }
        }
        
        // Fix V0[0] for bound state channel
        if (channel == 0) {
            success = FixV0(BE_s12, channel);
        }
        
        if (success) {
            ComputePhaseShifts(channel);
            
            // Calculate chi-squared
            double chi = 0.0;
            if (channel == 0) {
                // s=1/2: include scattering length in chi-squared
                
                //const double q_14 = 2. * 14;  // MeV/c
                //const double delta_14 = GetPhaseShift(channel, q_14);
                //const double _20 = 2. * 20;  // MeV/c
                //const double delta_20 = GetPhaseShift(channel, _20);
                //const double _24 = 2. * 24;  // MeV/c
                //const double delta_24 = GetPhaseShift(channel, _24);
                const double delta_14 = GetPhaseShift(channel, q_targets[0]);
                const double delta_20 = GetPhaseShift(channel, q_targets[1]);
                const double delta_24 = GetPhaseShift(channel, q_targets[2]);
                const double scatt_length = GetScatteringLength(channel);

                chi = 8 * pow(scatt_length - scatt_target, 2) + pow(delta_14 * 180.0/Constants::PI - delta_targets[0], 2) + 
                        pow(delta_20 * 180.0/Constants::PI - delta_targets[1], 2) + pow(delta_24 * 180.0/Constants::PI - delta_targets[2], 2);
            }
            
            for (size_t i = 0; i < q_targets.size(); i++) {
                //const double q_14 = 2. * 14;  // MeV/c
                //const double delta_14 = GetPhaseShift(channel, q_14);
                //const double _20 = 2. * 20;  // MeV/c
                //const double delta_20 = GetPhaseShift(channel, _20);
                //const double _24 = 2. * 24;  // MeV/c
                //const double delta_24 = GetPhaseShift(channel, _24);

                const double delta_14 = GetPhaseShift(channel, q_targets[0]);
                const double delta_20 = GetPhaseShift(channel, q_targets[1]);
                const double delta_24 = GetPhaseShift(channel, q_targets[2]);
                const double scatt_length = GetScatteringLength(channel);

                chi = + pow(delta_14 * 180.0/Constants::PI - delta_targets[0], 2) + 
                        pow(delta_20 * 180.0/Constants::PI - delta_targets[1], 2) + pow(delta_24 * 180.0/Constants::PI - delta_targets[2], 2);
            }
            
            if (chi < best_chi) {
                best_chi = chi;
                best_a = a[channel];
                best_V = V0[channel];
                
                if (debug_level >= 0) {
                    std::cout << "---------------------------- " << itry << std::endl;
                    std::cout << "chi = " << chi << std::endl;
                    std::cout << "# a (fm)    V0 (MeV)" << std::endl;
                    for (int iwell = 0; iwell < n_wells; iwell++) {
                        std::cout << std::fixed << std::setprecision(5) << a[channel][iwell] 
                                  << "  " << std::setprecision(3) << V0[channel][iwell] << std::endl;
                    }
                    if (channel == 0) {
                        std::cout << "scatt_length = " << scatt_length[channel] << " fm" << std::endl;
                    }
                }
            }
        }
    }
    
    // Set best parameters
    a[channel] = best_a;
    V0[channel] = best_V;
    if (channel == 0) {
        FixV0(BE_s12, channel);
    }
    ComputePhaseShifts(channel);
}

void pdSquareWell::PrintPhaseShifts(const std::string& filename) const {

    std::ofstream file;
    std::ostream* out = &std::cout;
    
    if (!filename.empty()) {
        file.open(filename);
        out = &file;
    }
    
    *out << "# q (MeV/c) | Ep (MeV) | delta_s12 (deg) | delta_s32 (deg)" << std::endl;
    *out << "# Scattering lengths: s=1/2: " << scatt_length[0] 
         << " fm, s=3/2: " << scatt_length[1] << " fm" << std::endl;
    
    for (int iq = 0; iq < n_q_points; iq++) {
        double q = (iq + 0.5) * dq;
        double Ep = 0.5 * m_proton * q * q / (mu * mu);
        *out << std::fixed << std::setprecision(2) << q << " " 
             << std::setprecision(4) << Ep;
        for (int channel = 0; channel < n_channels; channel++) {
            *out << " " << std::setprecision(3) << delta[channel][iq] * 180.0 / Constants::PI;
        }
        *out << std::endl;
    }
    
    if (file.is_open()) file.close();
}

void pdSquareWell::PlotPhaseShifts(int channel, const char* outputPdfPath) const {
    TCanvas* c1 = new TCanvas("c1", "p-d Phase Shifts", 1000, 600);
    
    if (channel < 0) {
        c1->Divide(2, 1);
    }
    
    std::vector<std::string> channel_names = {
        "s=1/2 (doublet)", 
        "s=3/2 (quartet)"
    };
    
    int start = (channel < 0) ? 0 : channel;
    int end = (channel < 0) ? n_channels : channel + 1;
    
    for (int ch = start; ch < end; ch++) {
        if (channel < 0) c1->cd(ch + 1);
        
        TGraph* gr = new TGraph();
        for (int iq = 0; iq < n_q_points; iq++) {
            double q = (iq + 0.5) * dq;
            double phase = delta[ch][iq] * 180.0 / Constants::PI;
            gr->SetPoint(iq, q, phase);
        }
        gr->SetLineColor(ch + 1);
        gr->SetLineWidth(2);
        gr->SetTitle((channel_names[ch] + ";q (MeV/c);#delta (degrees)").c_str());
        gr->Draw("AL");
    }
    
    c1->SaveAs(outputPdfPath);
}

void pdSquareWell::GetParameters(int channel, std::vector<double>& a_vals, 
                                  std::vector<double>& V_vals) const {
    a_vals = a[channel];
    V_vals = V0[channel];
}

void pdSquareWell::SetParameters(int channel, const std::vector<double>& a_vals, 
                                  const std::vector<double>& V_vals) {
    a[channel] = a_vals;
    V0[channel] = V_vals;
    if (channel == 0) {
        FixV0(BE_s12, channel);
    }
    ComputePhaseShifts(channel);
}
