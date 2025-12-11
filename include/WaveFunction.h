#pragma once

#include <complex>
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           #include <boost/math/special_functions/gamma.hpp>
#include <gsl/gsl_sf_gamma.h>
#include <boost/math/constants/constants.hpp>

#include "Physics.h"
 

std::complex<double> complex_gamma(const std::complex<double>& z)
{
    gsl_sf_result lnr, arg;
    gsl_sf_lngamma_complex_e(z.real(), z.imag(), &lnr, &arg);
    
    return std::exp(std::complex<double>(lnr.val, arg.val));
}

/**
 * Coulomb radial solution
 * @param r radius, in natural units (MeV)
 * @param k momentum (MeV)
 * @param eta Sommerfeld parameter
*/
std::complex<double> u0(const double r, const double k, const double eta)
{
    const std::complex<double> one_plus_ieta = 1. + 1i*eta;
    return std::exp(-boost::math::constants::pi<double>()*eta/2.) * complex_gamma(one_plus_ieta) * std::exp(1i*k*r) *
        k*r * boost::math::hypergeometric_1F1(one_plus_ieta, 2., -2.*1i*k*r);
}

/**
 * Coulomb solution
 * @param r radius, in natural units (MeV^{-1})
 * @param k momentum (MeV)
 * @param eta Sommerfeld parameter
*/
std::complex<double> psi0(const double r, const double k, const double eta)
{
    return u0(r, k, eta) / (k*r);
}

double Coulomb_penetration_factor(const double eta)
{
    return 2.*boost::math::constants::pi<double>()*eta / (std::exp(2.*boost::math::constants::pi<double>()*eta) - 1.);
}

double h_lambda(const double eta, const double lambda)
{
    double h = 0, h_previous = 0;
    for (int n=0; n<15; n++) {
        h += eta*eta / (n*n + eta*eta);
        if (h > 0 && (h - h_previous)/h < 1e-7)
            break;  // breakpoint for convergence
        h_previous = h;
    }

    h = h - std::log(lambda*eta) - boost::math::constants::euler<double>();
    
    return h;
}

/**
 * @param eta Sommerfeld parameter
 * @param lambda can be ±1
*/
std::complex<double> H_lambda(const double eta, const double lambda)
{
    return h_lambda(eta, lambda) + 1i*Coulomb_penetration_factor(eta)/(2.*eta);
}

/**
 * Scattering amplitude expressed with the effective range approximation
 * @param k momentum (MeV)
 * @param eta Sommerfeld parameter
 * @param mu reduced mass of the system (MeV)
 * @param a0 scattering length in natural units (MeV^{-1})
 * @param r0 effective range in natural units (MeV^{-1})
 * @param lambda can be ±1
*/
std::complex<double> scattering_amplitude(const double k, const double eta, const double mu, const double a0, const double r0, const double lambda=1.)
{
    const std::complex<double> H = H_lambda(eta, lambda);
    const std::complex<double> inverse_f = -1./a0 + 0.5*r0*k*k - 2*lambda*constant::ALPHA_EM*mu*H;
    return 1. / inverse_f;
}

std::complex<double> T_matrix_element(const double k, const double eta, const double mu, const double a0, const double r0, const double lambda=1.)
{
    const std::complex<double> f_SC = scattering_amplitude(k, eta, mu, a0, r0, lambda);
    const double Ac = Coulomb_penetration_factor(eta);

    const std::complex<double> one_plus_ieta = 1. + 1i*eta;
    const std::complex<double> one_minus_ieta = 1. - 1i*eta;
    const std::complex<double> e_2isigma0 = boost::math::tgamma(one_plus_ieta) / boost::math::tgamma(one_minus_ieta);

    return - 2.*boost::math::constants::pi<double>()/mu * Ac * e_2isigma0 * f_SC;
}

/**
 * G_C
*/
std::complex<double> Coulomb_propagator(const double r, const double k, const double eta, const double mu)
{
    const std::complex<double> one_plus_ieta = 1. + 1i*eta;
    return 1i*mu*k/boost::math::constants::pi<double>() * std::exp(1i*k*r) * boost::math::tgamma(one_plus_ieta) 
        * Tricomi_hypergeometric_U(one_plus_ieta, static_cast<std::complex<double>>(2.), -2.*1i*k*r);
}

std::complex<double> phi0(const double r, const double k, const double eta, const double mu, const double a0, const double r0, const double lambda=1.)
{
    std::complex<double> psi0_value = psi0(r, k, eta);
    std::complex<double> T_SC = T_matrix_element(k, eta, mu, a0, r0, lambda);
    std::complex<double> G_C = Coulomb_propagator(r, k, eta, mu);
    std::complex<double> e_pi_eta = std::exp(-boost::math::constants::pi<double>()*eta/2.);
    const std::complex<double> one_plus_ieta = 1. + 1i*eta;
    std::complex<double> gamma_one_plus_ieta = boost::math::tgamma(one_plus_ieta);
    
    return psi0_value + T_SC*G_C/(e_pi_eta*gamma_one_plus_ieta);
}
