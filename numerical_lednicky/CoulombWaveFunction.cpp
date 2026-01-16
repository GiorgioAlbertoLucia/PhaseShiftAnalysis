#include "CoulombWaveFunction.h"
#include "gsl/gsl_sf_coulomb.h"
#include "gsl/gsl_sf_gamma.h"
#include "flint/acb.h"
#include "flint/acb_hypgeom.h"
#include "flint/arb.h"
#include "flint/flint.h"
#include "DLM_Integration.h"
#include <cmath>

using namespace std;

CoulombWaveFunction::CoulombWaveFunction(double chargeRadius, double effectiveRange)
    : chargeRadius_(chargeRadius), effectiveRange_(effectiveRange) {}

double CoulombWaveFunction::CalculateAc(double eta) const {
    return 2.0 * PhysicalConstants::PI * eta / (exp(2.0 * PhysicalConstants::PI * eta) - 1.0);
}

double CoulombWaveFunction::CalculateH(double eta) const {
    if (fabs(eta) < 0.3) {
        return 1.2 * eta * eta - log(fabs(eta)) - PhysicalConstants::GAMMA;
    } else {
        double sum = 0.0;
        for (int n = 1; n <= 100000; ++n) {
            sum += 1.0 / (n * (n * n + eta * eta));
        }
        return eta * eta * sum - PhysicalConstants::GAMMA - log(fabs(eta));
    }
}

complex<double> CoulombWaveFunction::CalculateScatteringAmplitude(
    double k, complex<double> f0, double eta) const {
    
    const complex<double> i(0, 1);
    double ac = chargeRadius_ * PhysicalConstants::FM_TO_NU;
    double d0 = effectiveRange_ * PhysicalConstants::FM_TO_NU;
    
    return 1.0 / (1.0 / f0 + 0.5 * d0 * k * k - 
                  2.0 / ac * CalculateH(eta) - 
                  i * k * CalculateAc(eta));
}

complex<double> CoulombWaveFunction::CalculateTildeG(double rho, double eta) const {
    int kmax = 0;
    double fc_array, gc_array;
    double L_min = 0.0;
    double OverflowF = 0, OverflowG = 0;
    
    gsl_sf_coulomb_wave_FG_array(L_min, kmax, eta, fabs(rho), 
                                 &fc_array, &gc_array, 
                                 &OverflowF, &OverflowG);
    
    const complex<double> i(0, 1);
    return sqrt(CalculateAc(eta)) * (i * fc_array + gc_array);
}

complex<double> CoulombWaveFunction::CalculateHypergeometric1F1(
    double eta, double zeta) const {
    
    acb_t eta_acb, zeta_acb, b_value, result_acb;
    acb_init(eta_acb);
    acb_init(zeta_acb);
    acb_init(b_value);
    acb_init(result_acb);
    
    acb_set_d_d(eta_acb, 0.0, eta);
    acb_set_d_d(zeta_acb, 0.0, zeta);
    acb_set_d(b_value, 1.0);
    
    int regularized = 0;
    acb_hypgeom_1f1(result_acb, eta_acb, b_value, zeta_acb, regularized, 64);
    
    double real_part = arf_get_d(arb_midref(acb_realref(result_acb)), ARF_RND_NEAR);
    double imag_part = arf_get_d(arb_midref(acb_imagref(result_acb)), ARF_RND_NEAR);
    
    acb_clear(eta_acb);
    acb_clear(zeta_acb);
    acb_clear(b_value);
    acb_clear(result_acb);
    
    complex<double> result(real_part, imag_part);
    flint_cleanup();
    return result;
}

complex<double> CoulombWaveFunction::Psi(double k, double r, double t,
                                         complex<double> scatteringLength) const {
    const complex<double> i(0, 1);
    
    double eta = 1.0 / (k * chargeRadius_) / PhysicalConstants::FM_TO_NU;
    double rhoval = k * r * PhysicalConstants::FM_TO_NU;
    double zeta = rhoval * (1.0 + t);
    double rval = r * PhysicalConstants::FM_TO_NU;
    
    complex<double> f0 = scatteringLength * PhysicalConstants::FM_TO_NU;
    complex<double> fc = CalculateScatteringAmplitude(k, f0, eta);
    
    gsl_sf_result lnr, arg;
    gsl_sf_lngamma_complex_e(1.0, eta, &lnr, &arg);
    
    return pow(CalculateAc(eta), 0.5) * exp(i * arg.val) *
           (exp(-i * k * rval * t) * CalculateHypergeometric1F1(-eta, zeta) +
            fc * CalculateTildeG(rhoval, eta) / rval);
}

double CoulombWaveFunction::GetIntegrand(double k, double t, double r,
                                        complex<double> scatteringLength) const {
    complex<double> psi = Psi(k, r, t, scatteringLength);
    double integrand = abs(conj(psi) * psi) * r * r * 
                      pow(PhysicalConstants::FM_TO_NU, 3) * 
                      2.0 * PhysicalConstants::PI;
    return integrand;
}

// Static wrapper for integration
static double integrand_wrapper(double *params) {
    double &k = params[0];
    double &t = params[1];
    double &r = params[2];
    double &aRe = params[3];
    double &aIm = params[4];
    double &effRange = params[5];
    double &chargeRad = params[6];
    
    complex<double> scatLen(aRe, aIm);
    CoulombWaveFunction wf(chargeRad, effRange);
    return wf.GetIntegrand(k, t, r, scatLen);
}

double CoulombWaveFunction::CalculateDCky(double k, double r,
                                          complex<double> scatteringLength,
                                          unsigned int integrationSteps) const {
    double params[7] = {k, 0.1, r, scatteringLength.real(), 
                       scatteringLength.imag(), effectiveRange_, 
                       chargeRadius_};
    
    DLM_INT_SetFunction(integrand_wrapper, params, 1);
    return DLM_INT_SimpsonWiki(-1.0, 1.0, integrationSteps);
}