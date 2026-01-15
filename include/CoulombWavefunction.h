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
#include <gsl/gsl_sf_legendre.h>

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>

#include "Constants.h"

class CoulombWavefunction {
    
public:
    CoulombWavefunction(double m1, double m2, 
                        double charge1, double charge2);

    double _eta(const double k) const;
    double GamowFactor(const double eta) const;
    std::complex<double> h_function(const double eta) const;
    void compute_scattering_amplitude(const double k);
    std::complex<double> _G_tilde(const double rho, const double eta) const;
    
    void compute_wavefunction(const double k_vec[3], const double r_vec[3],
                              std::complex<double>& psi_c);
    void compute_wavefunction_series(const double k_vec[3], const double r_vec[3],
                              std::complex<double>& psi_c, const int lmax = 30);
    
    double correlation_function_optimized(double k, 
                                         double R_source,
                                         const int n_iterations = 10000);

private:
    double m1, m2;
    double charge1, charge2;
    double mu, a_C;
};

// Constructor
CoulombWavefunction::CoulombWavefunction(
    double m1, double m2, 
    double charge1, double charge2)
    : m1(m1), m2(m2), charge1(charge1), charge2(charge2)
{
    mu = (m1 * m2) / (m1 + m2);
    a_C = Constants::HBARC / (charge1 * charge2 * mu * Constants::ALPHA_EM); // fm
}

double CoulombWavefunction::_eta(const double k) const
{
    return 1. / (k * a_C);
}

double CoulombWavefunction::GamowFactor(const double eta) const
{
    return 2. * eta * Constants::PI/ (std::exp(2. * Constants::PI * eta) - 1.);
}


std::complex<double> CoulombWavefunction::_G_tilde(const double rho, const double eta) const
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

void CoulombWavefunction::compute_wavefunction(const double k_vec[3], const double r_vec[3],
                                                       std::complex<double>& psi_c) 
{
    const double k_dot_r = std::inner_product(k_vec, k_vec + 3, r_vec, 0.);
    const double k = std::hypot(k_vec[0], k_vec[1], k_vec[2]);
    const double r = std::hypot(r_vec[0], r_vec[1], r_vec[2]);
    const double rho = k*r;

    const double xi = rho - k_dot_r;

    const double eta = _eta(k);
    const double A_c = GamowFactor(eta);

    // e_i_sigma_c = Gamma(1 + i eta)
    gsl_sf_result lnr, arg;
    gsl_sf_lngamma_complex_e(1., eta, &lnr, &arg);
    const std::complex<double> Gamma = std::exp(std::complex<double>(lnr.val, arg.val));

    const std::complex<double> F = math::hypergeometric_1F1_complex(eta, xi, 1.e-12, 400);
    psi_c = std::exp(- Constants::PI * eta / 2.) * Gamma * std::exp(std::complex<double>(0., k_dot_r)) * F;
}

void CoulombWavefunction::compute_wavefunction_series(const double k_vec[3], const double r_vec[3],
                                                       std::complex<double>& psi_c, const int lmax) 
{
    using namespace std::complex_literals;

    const double k_dot_r = std::inner_product(k_vec, k_vec + 3, r_vec, 0.);
    const double k = std::hypot(k_vec[0], k_vec[1], k_vec[2]);
    const double r = std::hypot(r_vec[0], r_vec[1], r_vec[2]);
    const double rho = k*r;

    const double xi = rho - k_dot_r;
    const double costheta = k_dot_r / rho;

    const double eta = _eta(k);
    psi_c = 0;

    for (int l = 0; l < lmax; l++)
    {
        gsl_sf_result lnr, arg, Pl;
        const std::complex<double> e_i_sigma_l = std::exp(std::complex<double>(0., arg.val));
        gsl_sf_lngamma_complex_e(1. + l, eta, &lnr, &arg);

        gsl_sf_legendre_Pl_e(l, costheta, &Pl);
        
        gsl_sf_result F, Fp, G, Gp;
        double Fexp, Gexp;
        gsl_sf_coulomb_wave_FG_e(eta, rho, l, 0., &F, &Fp, &G, &Gp, &Fexp, &Gexp);

        psi_c += (2.*l + 1.) * std::pow(1i, l) * e_i_sigma_l * F.val * Pl.val;
    
    }

    psi_c = psi_c / rho;
}



double CoulombWavefunction::correlation_function_optimized(double k, double R_source,
                                                                   const int n_iterations) {
    
    double k_vec[3] = {0.0, 0.0, k};
    double r_vec[3] = {0., 0., 0.};
    const double sqrt2 = std::sqrt(2);
    double r, costheta, sintheta, phi;
    
    std::complex<double> psi_c;
    double C = 0.0;
        
    
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
        
        compute_wavefunction(k_vec, r_vec, psi_c);
        //compute_wavefunction_series(k_vec, r_vec, psi_c);
                
        C += std::norm(psi_c);
    }
    
    C = C / n_iterations;
    
    return C;
}