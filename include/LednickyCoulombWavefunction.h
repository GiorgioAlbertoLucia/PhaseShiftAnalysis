#pragma once

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numeric>

#include <TMath.h>
#include <TRandom3.h>

#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_coulomb.h>
#include <gsl/gsl_integration.h>

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>

#include "Constants.h"

class LednickyCoulombWavefunction {
    
public:
    LednickyCoulombWavefunction(double m1, double m2, 
                                double charge1, double charge2,
                                double f0s, double d0s, 
                                double f0t, double d0t,
                                double cg_singlet = 0.25, 
                                double cg_triplet = 0.75);

    double _eta(const double k) const;
    double GamowFactor(const double eta) const;
    std::complex<double> h_function(const double eta) const;
    void compute_scattering_amplitude(const double k);
    std::complex<double> _G_tilde(const double rho, const double eta) const;
    
    void compute_wavefunction(const double k_vec[3], const double r_vec[3],
                              std::complex<double>& psi_s, std::complex<double>& psi_t);
    
    double correlation_function_optimized(double k, 
                                         double R_source,
                                         const int n_iterations = 10000);

private:
    double m1, m2;
    double charge1, charge2;
    double f0s, d0s, f0t, d0t;
    std::complex<double> f_c_s, f_c_t;
    double cg_singlet, cg_triplet;
    double mu, a_C;
};

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
    
    f_c_s = std::complex<double>{0., 0.};
    f_c_t = std::complex<double>{0., 0.};
    mu = (m1 * m2) / (m1 + m2);
    a_C = Constants::HBARC / (charge1 * charge2 * mu * Constants::ALPHA_EM); // fm
}

double LednickyCoulombWavefunction::_eta(const double k) const
{
    return 1. / (k * a_C);
}

double LednickyCoulombWavefunction::GamowFactor(const double eta) const
{
    return 2. * eta * Constants::PI/ (std::exp(2. * Constants::PI * eta) - 1.);
}

std::complex<double> LednickyCoulombWavefunction::h_function(double eta) const {
    
    double h = 0., h_previous = 0;
    for (int n = 1; n < 30; n++)
    {
        h += eta*eta / (n*(n*n +  eta*eta));
        if ((h - h_previous)/h < 1e-7)
            break;
    
        h_previous = h;
    }
    
    h = h - std::log(eta) - Constants::EULER;
    
    return std::complex<double>{h, 0.};

    //std::complex<double> z = {0., eta};
    //return 0.5 * (math::digamma_complex(z) + math::digamma_complex(-z) - std::log(eta*eta));
}

void LednickyCoulombWavefunction::compute_scattering_amplitude(const double k)
{
    
    double term1_s = -1.0 / f0s;
    double term2_s = (d0s * k * k) / 2.0;

    double term1_t = -1.0 / f0t;
    double term2_t = (d0t * k * k) / 2.0;
    
    const double eta = _eta(k);
    double gamow = GamowFactor(eta);
    
    std::complex<double> term4 = -2.0 * h_function(eta) / a_C;
    
    std::complex<double> denominator_s(term1_s + term2_s, -k * gamow);
    denominator_s += term4;
    f_c_s = 1. / denominator_s;

    std::complex<double> denominator_t(term1_t + term2_t, -k * gamow);
    denominator_t += term4;
    f_c_t = 1. / denominator_t;
}

std::complex<double> LednickyCoulombWavefunction::_G_tilde(const double rho, const double eta) const
{
    // Debug output
    //std::cout << "DEBUG _G_tilde: rho = " << rho << ", eta = " << eta << std::endl;
    
    if (rho <= 0.0) {
        std::cerr << "ERROR: rho must be positive! rho = " << rho << std::endl;
    }
    if (std::isnan(rho) || std::isnan(eta)) {
        std::cerr << "ERROR: NaN detected! rho = " << rho << ", eta = " << eta << std::endl;
    }
    if (std::isinf(rho) || std::isinf(eta)) {
        std::cerr << "ERROR: Inf detected! rho = " << rho << ", eta = " << eta << std::endl;
    }
    
    gsl_sf_result F0, G0, F0_prime, G0_prime;
    double exp_F, exp_G;
    double l = 0;
    gsl_sf_coulomb_wave_FG_e(eta, rho, l, 0, &F0, &G0, &F0_prime, &G0_prime, &exp_F, &exp_G);
    return std::complex<double>{G0.val, F0.val};
}

void LednickyCoulombWavefunction::compute_wavefunction(const double k_vec[3], const double r_vec[3],
                                                       std::complex<double>& psi_s, std::complex<double>& psi_t) 
{

    const double k_dot_r = std::inner_product(k_vec, k_vec + 3, r_vec, 0.);
    const double k = std::hypot(k_vec[0], k_vec[1], k_vec[2]);
    const double r = std::hypot(r_vec[0], r_vec[1], r_vec[2]);
    const double rho = k*r;
    //std::cout << "DEBUG compute_wf: k = " << k << ", r = " << r << ", k_dot_r = " << k_dot_r << std::endl;

    const double xi = rho + k_dot_r;

    const double eta = _eta(k);
    const double A_c = GamowFactor(eta);
    //std::cout << "DEBUG: eta = " << eta << ", a_C = " << a_C << std::endl;

    gsl_sf_result lnr, arg;
    gsl_sf_lngamma_complex_e(1., eta, &lnr, &arg);
    //const std::complex<double> e_i_sigma_c = std::exp(std::complex<double>(lnr.val, arg.val));
    const std::complex<double> e_i_sigma_c = std::exp(std::complex<double>(0., arg.val));
    
    //const std::complex<double> e_i_sigma_c{std::cos(sigma_c), std::sin(sigma_c)};

    //const std::complex<double> F = boost::math::hypergeometric_1F1(std::complex<double>{0., -eta}, 1., std::complex<double>{0., xi});
    //std::cout << "DEBUG: eta = " << eta << ", xi = " << xi << std::endl;
    const std::complex<double> F = math::hypergeometric_1F1_complex(eta, xi, 1.e-12, 400);

    LednickyCoulombWavefunction::compute_scattering_amplitude(k);

    const std::complex<double> G_tilde = _G_tilde(rho, eta);

    const std::complex<double> common_term = e_i_sigma_c * std::sqrt(A_c) * std::exp(std::complex<double>{0., -k_dot_r}) * F;

    psi_s = common_term + e_i_sigma_c * std::sqrt(A_c) * f_c_s * G_tilde / r;
    psi_t = common_term + e_i_sigma_c * std::sqrt(A_c) * f_c_t * G_tilde / r;
}




double LednickyCoulombWavefunction::correlation_function_optimized(double k, double R_source,
                                                                   const int n_iterations) {
    
    double k_vec[3] = {0.0, 0.0, k};
    double r_vec[3] = {0., 0., 0.};
    const double sqrt2 = std::sqrt(2);
    double r, costheta, sintheta, phi;
    
    std::complex<double> psi_s, psi_t;
    double C_s = 0.0, C_t = 0.0;
        
    
    std::vector<double> integrand_s(n_iterations);
    std::vector<double> integrand_t(n_iterations);
    
    for (size_t iter = 0; static_cast<int>(iter) < n_iterations; ++iter) {
        
        r_vec[0] = gRandom->Gaus(0., R_source * sqrt2);
        r_vec[1] = gRandom->Gaus(0., R_source * sqrt2);
        r_vec[2] = gRandom->Gaus(0., R_source * sqrt2);

        //r = gRandom->Gaus(0., R_source * sqrt2);
        //costheta = gRandom->Uniform(-1., 1.);
        //phi = gRandom->Uniform(0., 2. * Constants::PI);
        //sintheta = std::sqrt(1. - costheta*costheta);
        //
        //r_vec[0] = r * sintheta* std::cos(phi);
        //r_vec[1] = r * sintheta* std::sin(phi);
        //r_vec[2] = r * costheta;
        
        compute_wavefunction(k_vec, r_vec, psi_s, psi_t);
                
        C_s += std::norm(psi_s);
        C_t += std::norm(psi_t);
    }
    
    C_s = C_s / n_iterations;
    C_t = C_t / n_iterations;
    
    return cg_singlet * C_s + cg_triplet * C_t;
}