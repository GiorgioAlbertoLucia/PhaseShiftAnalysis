#pragma once

#include "Config.h"

#include <complex>
#include <functional>

class CoulombWaveFunction {
public:
    CoulombWaveFunction(double chargeRadius) : chargeRadius_(chargeRadius) {};

    struct EtaContext {
        double eta;
        double sqrtAc;
        std::complex<double> fc;
        std::complex<double> gammaPhase;
    };
    
    // Calculate wave function at given parameters
    std::complex<double> Psi(double k, double r, double t, const EtaContext& ctx) const;

    double GetIntegrand(double k, double t, double r, const EtaContext& ctx) const;

    EtaContext PrecomputeEtaContext(double k, std::complex<double> scatteringLength,
                                    double effectiveRange) const;
    
    // Thread-safe composite Simpson's rule, equivalent to DLM_INT_SimpsonWiki but
    // taking the integrand directly instead of going through DLM_Integration's
    // global function-pointer state (which isn't safe to share across threads).
    static double ThreadSafeSimpsonWiki(const std::function<double(double)>& f,
                                double a, double b, unsigned int N);

    // Calculate dC(k,y) - the correlation function
    double CalculateDCky(double k, double r, 
                        std::complex<double> scatteringLength,
                        double effectiveRange,
                        unsigned int integrationSteps = 64) const;

    double CalculateDCky(double k, double r, const EtaContext& ctx,
                    unsigned int integrationSteps = 64) const;

private:
    double chargeRadius_;
    
    // Helper functions
    double CalculateAc(double eta) const;
    double CalculateH(double eta) const;
    std::complex<double> CalculateScatteringAmplitude(
        double k, std::complex<double> f0, double d0, double eta) const;
    std::complex<double> CalculateTildeG(double rho, double eta) const;
    std::complex<double> CalculateHypergeometric1F1(double eta, double zeta) const;
};
