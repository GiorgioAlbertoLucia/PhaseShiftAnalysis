#pragma once
#include <complex>

/**
 * Tricomi's confluent hypergeometric function U
*/
template<typename T>
T Tricomi_U(const T a, const T b, const T z);

/**
 * Coulomb radial solution
 * @param r radius, in natural units (MeV)
 * @param k momentum (MeV)
 * @param eta Sommerfeld parameter
*/
std::complex<double> u0(const double r, const double k, const double eta);
/**
 * Coulomb solution
 * @param r radius, in natural units (MeV^{-1})
 * @param k momentum (MeV)
 * @param eta Sommerfeld parameter
*/
std::complex<double> psi0(const double r, const double k, const double eta);

double Coulomb_penetration_factor(const double eta);
double h_lambda(const double eta, const double lambda);
/**
 * @param eta Sommerfeld parameter
 * @param lambda can be ±1
*/
std::complex<double> H_lambda(const double eta, const double lambda);

/**
 * Scattering amplitude expressed with the effective range approximation
 * @param k momentum (MeV)
 * @param eta Sommerfeld parameter
 * @param mu reduced mass of the system (MeV)
 * @param a0 scattering length in natural units (MeV^{-1})
 * @param r0 effective range in natural units (MeV^{-1})
 * @param lambda can be ±1
*/
std::complex<double> scattering_amplitude(const double k, const double eta,
                                          const double mu, const double a0, const double r0,
                                          const double lambda = 1.0);

std::complex<double> T_matrix_element(const double k, const double eta,
                                      const double mu, const double a0, const double r0,
                                      const double lambda = 1.0);

/**
 * G_C
*/
std::complex<double> Coulomb_propagator(const double r, const double k, const double eta, const double mu);

/**
 * Full wavefunction (Coulomb + s-wave strong interaction)
*/
std::complex<double> phi0(const double r, const double k, const double eta, const double mu,
                          const double a0, const double r0, const double lambda = 1.0);
