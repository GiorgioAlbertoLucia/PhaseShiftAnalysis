#pragma once

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_coulomb.h>
#include <gsl/gsl_integration.h>

#include <TMath.h>


// Constants class
namespace Constants {
    const double M_PROTON = 938.272;    // GeV/c^2
    const double M_DEUTERON = 1875.613;  // GeV/c^2
    const double Z_PROTON = 1.0;
    const double Z_DEUTERON = 1.0;
    const double HBARC = 197.3;          // MeV·fm
}

class LednickyCoulombWavefunction {
private:
    double m1, m2;
    double charge1, charge2;
    double f0s, d0s, f0t, d0t;
    double cg_singlet, cg_triplet;
    double mu;
    
    static const double alpha;
    static const double hbarc;
    
public:
    LednickyCoulombWavefunction(double m1, double m2, 
                                double charge1, double charge2,
                                double f0s, double d0s, 
                                double f0t, double d0t,
                                double cg_singlet = 0.25, 
                                double cg_triplet = 0.75);
    
    double bohr_radius() const;
    double eta(double k) const;
    double gamow_factor(double eta) const;
    std::complex<double> coulomb_function_F(double eta, double xi) const;
    double h_function(double eta) const;
    std::complex<double> scattering_amplitude_f_c(double k, double eta, 
                                                   const char* state) const;
    std::complex<double> G_function(double rho, double eta) const;
    
    void wavefunction_at_point(const double k_vec[3], const double r_vec[3],
                              std::complex<double>& psi_s, 
                              std::complex<double>& psi_t) const;
    
    double correlation_function_optimized(double k, 
                                         double R_source,
                                         const std::vector<double>& r_points,
                                         int n_theta = 20, 
                                         int n_phi = 20) const;
};

// Static member initialization
const double LednickyCoulombWavefunction::alpha = 1.0/137.036;
const double LednickyCoulombWavefunction::hbarc = 197.3;

// Constructor
LednickyCoulombWavefunction::LednickyCoulombWavefunction(
    double m1, double m2, 
    double charge1, double charge2,
    double f0s, double d0s, 
    double f0t, double d0t,
    double cg_singlet, 
    double cg_triplet)
    : m1(m1), m2(m2), charge1(charge1), charge2(charge2),
      f0s(f0s), d0s(d0s), f0t(f0t), d0t(d0t),
      cg_singlet(cg_singlet), cg_triplet(cg_triplet) {
    mu = (m1 * m2) / (m1 + m2);
}

double LednickyCoulombWavefunction::bohr_radius() const {
    double charge_product = TMath::Abs(charge1 * charge2);
    if (charge_product < 1e-10) {
        return 1e10;  // Very large number instead of infinity
    }
    return hbarc / (alpha * charge_product * mu);
}

double LednickyCoulombWavefunction::eta(double k) const {
    double a_C = bohr_radius();
    return 1.0 / (k * a_C);
}

double LednickyCoulombWavefunction::gamow_factor(double eta) const {
    if (TMath::Abs(eta) < 1e-10) {
        return 1.0;
    }
    return 2.0 * TMath::Pi() * eta / (TMath::Exp(2.0 * TMath::Pi() * eta) - 1.0);
}

std::complex<double> LednickyCoulombWavefunction::coulomb_function_F(
    double eta, double xi) const {
    
    gsl_sf_result F, Fp, G, Gp;
    double exp_F, exp_G;
    
    int status = gsl_sf_coulomb_wave_FG_e(eta, xi, 0.0, 0, 
                                          &F, &Fp, &G, &Gp, 
                                          &exp_F, &exp_G);
    
    return std::complex<double>(F.val, 0.0);
}

double LednickyCoulombWavefunction::h_function(double eta) const {
    if (TMath::Abs(eta) < 1e-10) {
        return 0.0;
    }
    
    // Approximation for h(η) = Re[ψ(iη)]
    if (TMath::Abs(eta) > 0.1) {
        return TMath::Log(TMath::Abs(eta)) - 1.0/(2.0*eta);
    }
    
    return 0.0;
}

std::complex<double> LednickyCoulombWavefunction::scattering_amplitude_f_c(
    double k, double eta, const char* state) const {
    
    double f0, d0;
    if (strcmp(state, "singlet") == 0) {
        f0 = f0s;
        d0 = d0s;
    } else if (strcmp(state, "triplet") == 0) {
        f0 = f0t;
        d0 = d0t;
    } else {
        return std::complex<double>(0.0, 0.0);
    }
    
    double a_C = bohr_radius();
    
    double term1 = -1.0 / f0;
    double term2 = (d0 * k * k) / 2.0;
    double gamow = gamow_factor(eta);
    double term4 = -2.0 * h_function(eta) / a_C;
    
    std::complex<double> denominator(term1 + term2 + term4, -k * gamow);
    
    return 1.0 / denominator;
}

std::complex<double> LednickyCoulombWavefunction::G_function(
    double rho, double eta) const {
    
    gsl_sf_result F0, Fp, G0, Gp;
    double exp_F, exp_G;
    
    int status = gsl_sf_coulomb_wave_FG_e(eta, rho, 0.0, 0,
                                          &F0, &Fp, &G0, &Gp,
                                          &exp_F, &exp_G);
    
    double A_C = gamow_factor(eta);
    std::complex<double> result = TMath::Sqrt(A_C) * 
                                  std::complex<double>(G0.val, F0.val);
    
    return result;
}

void LednickyCoulombWavefunction::wavefunction_at_point(
    const double k_vec[3], const double r_vec[3],
    std::complex<double>& psi_s, 
    std::complex<double>& psi_t) const {
    
    double k = TMath::Sqrt(k_vec[0]*k_vec[0] + k_vec[1]*k_vec[1] + k_vec[2]*k_vec[2]);
    double r = TMath::Sqrt(r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2]);
    
    double eta_val = eta(k);
    double A_C = gamow_factor(eta_val);
    
    double cos_theta = (k * r > 1e-10) ? 
        (k_vec[0]*r_vec[0] + k_vec[1]*r_vec[1] + k_vec[2]*r_vec[2]) / (k * r) : 1.0;
    
    double rho = k * r;
    double xi = rho * (1.0 + cos_theta);
    
    // Scattering amplitudes
    std::complex<double> f_c_s = scattering_amplitude_f_c(k, eta_val, "singlet");
    std::complex<double> f_c_t = scattering_amplitude_f_c(k, eta_val, "triplet");
    
    // Coulomb functions
    std::complex<double> F = coulomb_function_F(eta_val, xi);
    std::complex<double> G = G_function(rho, eta_val);
    
    // Phase factor
    gsl_sf_result lnr, arg;
    gsl_sf_lngamma_complex_e(1.0, eta_val, &lnr, &arg);
    std::complex<double> phase(TMath::Cos(arg.val), TMath::Sin(arg.val));
    
    // Wavefunctions
    std::complex<double> common = phase * TMath::Sqrt(A_C) * 
                                  std::exp(std::complex<double>(0.0, -rho)) * F;
    
    psi_s = common + f_c_s * G / r;
    psi_t = common + f_c_t * G / r;
}

double LednickyCoulombWavefunction::correlation_function_optimized(
    double k, 
    double R_source,
    const std::vector<double>& r_points,
    int n_theta, 
    int n_phi) const {
    
    // Pre-compute angular grid
    std::vector<double> theta(n_theta), phi(n_phi);
    for (int i = 0; i < n_theta; ++i) {
        theta[i] = i * TMath::Pi() / (n_theta - 1);
    }
    for (int i = 0; i < n_phi; ++i) {
        phi[i] = i * 2.0 * TMath::Pi() / (n_phi - 1);
    }
    
    double dtheta = theta[1] - theta[0];
    double dphi = phi[1] - phi[0];
    
    double k_vec[3] = {0.0, 0.0, k};
    
    std::vector<double> integrand_s(r_points.size());
    std::vector<double> integrand_t(r_points.size());
    
    // Gaussian source function
    double source_norm = 1.0 / TMath::Power(2.0 * R_source * TMath::Sqrt(TMath::Pi()), 3);
    
    for (size_t j = 0; j < r_points.size(); ++j) {
        double r = r_points[j];
        double source_val = source_norm * TMath::Exp(-r*r / (2.0 * R_source * R_source));
        
        double psi_s_squared_sum = 0.0;
        double psi_t_squared_sum = 0.0;
        
        for (int ti = 0; ti < n_theta; ++ti) {
            double th = theta[ti];
            double sin_th = TMath::Sin(th);
            
            for (int pi = 0; pi < n_phi; ++pi) {
                double ph = phi[pi];
                
                double r_vec[3] = {
                    r * sin_th * TMath::Cos(ph),
                    r * sin_th * TMath::Sin(ph),
                    r * TMath::Cos(th)
                };
                
                std::complex<double> psi_s, psi_t;
                wavefunction_at_point(k_vec, r_vec, psi_s, psi_t);
                
                double weight = sin_th * dtheta * dphi;
                psi_s_squared_sum += std::norm(psi_s) * weight;
                psi_t_squared_sum += std::norm(psi_t) * weight;
            }
        }
        
        double psi_s_squared_avg = psi_s_squared_sum / (4.0 * TMath::Pi());
        double psi_t_squared_avg = psi_t_squared_sum / (4.0 * TMath::Pi());
        
        integrand_s[j] = source_val * psi_s_squared_avg * r * r;
        integrand_t[j] = source_val * psi_t_squared_avg * r * r;
    }
    
    // Trapezoidal integration
    double C_s = 0.0, C_t = 0.0;
    for (size_t i = 1; i < r_points.size(); ++i) {
        double dr = r_points[i] - r_points[i-1];
        C_s += 0.5 * (integrand_s[i] + integrand_s[i-1]) * dr;
        C_t += 0.5 * (integrand_t[i] + integrand_t[i-1]) * dr;
    }
    
    C_s *= 4.0 * TMath::Pi();
    C_t *= 4.0 * TMath::Pi();
    
    return cg_singlet * C_s + cg_triplet * C_t;
}