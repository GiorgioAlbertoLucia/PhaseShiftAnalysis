#pragma once
#include <complex>
#include "Config.h"

class CoulombWaveFunction {
public:
    CoulombWaveFunction(double chargeRadius, double effectiveRange);
    
    // Calculate wave function at given parameters
    std::complex<double> Psi(double k, double r, double t, 
                             std::complex<double> scatteringLength) const;
    
    // Calculate correlation function integrand
    double GetIntegrand(double k, double t, double r, 
                       std::complex<double> scatteringLength) const;
    
    // Calculate dC(k,y) - the correlation function
    double CalculateDCky(double k, double r, 
                        std::complex<double> scatteringLength,
                        unsigned int integrationSteps = 64) const;

private:
    double chargeRadius_;
    double effectiveRange_;
    
    // Helper functions
    double CalculateAc(double eta) const;
    double CalculateH(double eta) const;
    std::complex<double> CalculateScatteringAmplitude(
        double k, std::complex<double> f0, double eta) const;
    std::complex<double> CalculateTildeG(double rho, double eta) const;
    std::complex<double> CalculateHypergeometric1F1(double eta, double zeta) const;
};
