#include "CoulombWaveFunction.h"

#include "DLM_Integration.h"

#include "gsl/gsl_sf_coulomb.h"
#include "gsl/gsl_sf_gamma.h"

#include "acb.h"
#include "acb_hypgeom.h"
#include "arb.h"
#include "flint.h"
/// #include "flint/acb.h"
/// #include "flint/acb_hypgeom.h"
/// #include "flint/arb.h"
/// #include "flint/flint.h"

#include <cmath>

using namespace std;

double CoulombWaveFunction::CalculateAc(double eta) const {
    return 2.0 * PhysicalConstants::PI * eta / (exp(2.0 * PhysicalConstants::PI * eta) - 1.0);
}

double CoulombWaveFunction::CalculateH(double eta) const {
    if (fabs(eta) < 0.3) {
        return 1.2 * eta * eta - log(fabs(eta)) - PhysicalConstants::GAMMA;
    } else {
        double sum = 0.0;
        for (int n = 1; n <= 100000; ++n) {
            double term = 1.0 / (n * (n * n + eta * eta));
            sum += term;
            if (term < 1e-15 * sum) break;   // converged
        }
        return eta * eta * sum - PhysicalConstants::GAMMA - log(fabs(eta));
    }
}

complex<double> CoulombWaveFunction::CalculateScatteringAmplitude(
    double k, complex<double> f0, double d0, double eta) const {
    
    const complex<double> i(0, 1);
    double ac = chargeRadius_ * PhysicalConstants::FM_TO_NU;
    d0 = d0 * PhysicalConstants::FM_TO_NU;
    
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
    //flint_cleanup();
    return result;
}

complex<double> CoulombWaveFunction::Psi(double k, double r, double t,
                                         const EtaContext& ctx) const {
    const complex<double> i(0, 1);

    double rhoval = k * r * PhysicalConstants::FM_TO_NU;
    double zeta = rhoval * (1.0 + t);
    double rval = r * PhysicalConstants::FM_TO_NU;

    return ctx.sqrtAc * ctx.gammaPhase *
           (exp(-i * k * rval * t) * CalculateHypergeometric1F1(-ctx.eta, zeta) +
            ctx.fc * CalculateTildeG(rhoval, ctx.eta) / rval);
}

CoulombWaveFunction::EtaContext CoulombWaveFunction::PrecomputeEtaContext(
    double k, complex<double> scatteringLength, double effectiveRange) const {

    EtaContext ctx;
    ctx.eta = 1.0 / (k * chargeRadius_) / PhysicalConstants::FM_TO_NU;
    ctx.sqrtAc = pow(CalculateAc(ctx.eta), 0.5);

    complex<double> f0 = scatteringLength * PhysicalConstants::FM_TO_NU;
    double d0 = effectiveRange * PhysicalConstants::FM_TO_NU;
    ctx.fc = CalculateScatteringAmplitude(k, f0, d0, ctx.eta);

    gsl_sf_result lnr, arg;
    gsl_sf_lngamma_complex_e(1.0, ctx.eta, &lnr, &arg);
    const complex<double> i(0, 1);
    ctx.gammaPhase = exp(i * arg.val);

    return ctx;
}

double CoulombWaveFunction::GetIntegrand(double k, double t, double r,
                                        const EtaContext& ctx) const {
    complex<double> psi = Psi(k, r, t, ctx);
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
    double &eta = params[3];
    double &sqrtAc = params[4];
    double &fcRe = params[5];
    double &fcIm = params[6];
    double &gammaRe = params[7];
    double &gammaIm = params[8];
    double &chargeRad = params[9];

    CoulombWaveFunction::EtaContext ctx{eta, sqrtAc, {fcRe, fcIm}, {gammaRe, gammaIm}};
    CoulombWaveFunction wf(chargeRad);
    return wf.GetIntegrand(k, t, r, ctx);
}

// Same composite Simpson's rule formula as DLM_INT_SimpsonWiki in DLM_Integration.cpp,
// reimplemented here so it doesn't touch that file's thread-unsafe global state.
double CoulombWaveFunction::ThreadSafeSimpsonWiki(const std::function<double(double)>& f,
                             double a, double b, unsigned int N) {
    if (N == 0) return 0.0;
    double h = (b - a) / double(N);
    double result = 0.0;
    for (unsigned i = 0; i < N; i++) {
        double x0 = a + i * h;
        double xm = x0 + 0.5 * h;
        double x1 = a + (i + 1) * h;
        result += h * (f(x0) + 4.0 * f(xm) + f(x1)) / 6.0;
    }
    return result;
}

double CoulombWaveFunction::CalculateDCky(double k, double r, const EtaContext& ctx,
                                          unsigned int integrationSteps) const {
    auto integrand = [&](double t) { return GetIntegrand(k, t, r, ctx); };
    return CoulombWaveFunction::ThreadSafeSimpsonWiki(integrand, -1.0, 1.0, integrationSteps);
}

double CoulombWaveFunction::CalculateDCky(double k, double r,
                                          complex<double> scatteringLength,
                                          double effectiveRange,
                                          unsigned int integrationSteps) const {
    EtaContext ctx = PrecomputeEtaContext(k, scatteringLength, effectiveRange);
    return CalculateDCky(k, r, ctx, integrationSteps);
}
