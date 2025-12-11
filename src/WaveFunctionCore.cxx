#include "../include/WaveFunctionCore.h"
#include "../include/Physics.h"

#include <cmath>
#include <complex>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <boost/math/constants/constants.hpp>

template<typename T>
T Tricomi_U(const T a, const T b, const T z)
{
    return boost::math::tgamma(1. - b)/boost::math::tgamma(a + 1. - b)*boost::math::hypergeometric_1F1(a, b, z) + 
            boost::math::tgamma(b - 1.)/boost::math::tgamma(a)*std::pow(z, 1.-b)*boost::math::hypergeometric_1F1(a, b, z);
}

std::complex<double> u0(const double r, const double k, const double eta)
{
    return std::exp(-boost::math::constants::pi<double>()*eta/2.) * boost::math::tgamma(one_plus_ieta) * std::exp(1i*k*r) 
        * k*r * boost::math::hypergeometric_1F1(one_plus_ieta, 2., -2.*1i*k*r);
}

std::complex<double> psi0(const double r, const double k, const double eta)
{
    return u0(r, k, eta) / (k*r);
}

double Coulomb_penetration_factor(const double eta)
{
    return 2.0 * boost::math::constants::pi<double>() * eta / (std::exp(2.0 * boost::math::constants::pi<double>() * eta) - 1.0);
}

double h_lambda(double eta, double lambda)
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

std::complex<double> H_lambda(double eta, double lambda)
{
    return h_lambda(eta, lambda) + 1i*Coulomb_penetration_factor(eta)/(2.*eta);
}

// scattering amplitude
std::complex<double> scattering_amplitude(double k, double eta,
                                          double mu, double a0, double r0,
                                          double lambda)
{
    const std::complex<double> H = H_lambda(eta, lambda);
    const std::complex<double> inverse_f = -1./a0 + 0.5*r0*k*k - 2*lambda*constant::ALPHA_EM*mu*H;
    return 1. / inverse_f;
}

std::complex<double> T_matrix_element(double k, double eta,
                                      double mu, double a0, double r0,
                                      double lambda)
{
    const std::complex<double> f_SC = scattering_amplitude(k, eta, mu, a0, r0, lambda);
    const double Ac = Coulomb_penetration_factor(eta);

    const std::complex<double> one_plus_ieta = 1. + 1i*eta;
    const std::complex<double> one_minus_ieta = 1. - 1i*eta;
    const std::complex<double> e_2isigma0 = boost::math::tgamma(one_plus_ieta) / boost::math::tgamma(one_minus_ieta);

    return - 2.*boost::math::constants::pi<double>()/mu * Ac * e_2isigma0 * f_SC;
}

// Coulomb propagator
std::complex<double> Coulomb_propagator(double r, double k, double eta, double mu)
{
    const std::complex<double> one_plus_ieta = 1. + 1i*eta;
    const std::complex<double> b = 2.;
    const std::complex<double> z = -2. * 1i * k*r;

    return 1i*mu*k/boost::math::constants::pi<double>() * std::exp(1i*k*r) * boost::math::tgamma(one_plus_ieta) 
        * Tricomi_U(one_plus_ieta, b, z);
}

// phi0(r)
std::complex<double> phi0(double r, double k, double eta, double mu,
                          double a0, double r0, double lambda)
{
    std::complex<double> psi0_value = psi0(r, k, eta);
    std::complex<double> T_SC = T_matrix_element(k, eta, mu, a0, r0, lambda);
    std::complex<double> G_C = Coulomb_propagator(r, k, eta, mu);
    std::complex<double> e_pi_eta = std::exp(-boost::math::constants::pi<double>()*eta/2.);
    const std::complex<double> one_plus_ieta = 1. + 1i*eta;
    std::complex<double> gamma_one_plus_ieta = boost::math::tgamma(one_plus_ieta);
    
    return psi0_value + T_SC * G_C / (e_pi_eta * gamma_one_plus_ieta);
}
