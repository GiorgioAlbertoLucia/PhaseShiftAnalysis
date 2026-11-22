#include "CoulombWaveFunction.h"

#include "DLM_Integration.h"

#include "gsl/gsl_sf_coulomb.h"
#include "gsl/gsl_sf_gamma.h"
#include "gsl/gsl_errno.h"

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
    if (eta == 0.0) return 1.0; // neutral pair (e.g. pi n): no Coulomb
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
    
    const double coulombTerm = std::isfinite(ac) ? 2.0 / ac * CalculateH(eta) : 0.0; // 0 for neutral pairs
    return 1.0 / (1.0 / f0 + 0.5 * d0 * k * k - 
                  coulombTerm - 
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
    // f0 = 0 means no strong interaction (1/f0 would give NaN with std::complex)
    ctx.fc = (f0 == 0.0) ? complex<double>(0.0, 0.0)
                         : CalculateScatteringAmplitude(k, f0, d0, ctx.eta);

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
    if (resonance_.enabled && resonance_.decayOnly) return ResonanceTerm(k, r, ctx.eta);
    auto integrand = [&](double t) { return GetIntegrand(k, t, r, ctx); };
    double value = CoulombWaveFunction::ThreadSafeSimpsonWiki(integrand, -1.0, 1.0, integrationSteps);
    if (resonance_.enabled) value += ResonanceTerm(k, r, ctx.eta);
    return value;
}

void CoulombWaveFunction::SetResonance(const ResonanceConfig& res, double mass1, double mass2) {
    resonance_ = res;
    mass1_ = mass1;
    mass2_ = mass2;
    if (resonance_.enabled) gsl_set_error_handler_off(); // check GSL status codes ourselves
}

double CoulombWaveFunction::ResonanceTerm(double k, double r, double eta) const {
    const complex<double> i(0, 1);
    const int L = resonance_.L;
    const double rho = k * r * PhysicalConstants::FM_TO_NU;
    if (rho <= 0.0) return 0.0;

    // Regular / irregular Coulomb functions F_L, G_L
    gsl_sf_result F, Fp, G, Gp;
    double expF = 0.0, expG = 0.0;
    int status = gsl_sf_coulomb_wave_FG_e(eta, rho, double(L), 0, &F, &Fp, &G, &Gp, &expF, &expG);
    if (status != GSL_SUCCESS && status != GSL_EOVRFLW) return 0.0;
    double FL = F.val * exp(expF);
    double GL = G.val * exp(expG);
    // Asymptotic form is not valid inside the interaction region: regulate G_L at small r
    GL *= pow(1.0 - exp(-r / resonance_.regulatorRadius), 2 * L + 1);

    // Breit-Wigner with energy-dependent width, Gamma(k) ~ k^(2L+1) with Blatt-Weisskopf barrier (L = 1 form)
    auto barrier = [&](double q) {
        double z = pow(q * resonance_.barrierRadius * PhysicalConstants::FM_TO_NU, 2);
        return (L == 0) ? 1.0 : z / (1.0 + z);
    };
    auto momentumAt = [&](double sqrtS) {
        double s = sqrtS * sqrtS;
        double a = s - pow(mass1_ + mass2_, 2), b = s - pow(mass1_ - mass2_, 2);
        return (a > 0) ? sqrt(a * b) / (2.0 * sqrtS) : 0.0;
    };
    const double kR = momentumAt(resonance_.mass);
    const double sqrtS = sqrt(mass1_ * mass1_ + k * k) + sqrt(mass2_ * mass2_ + k * k);
    double gammaK = resonance_.width;
    if (kR > 0.0) gammaK *= (k / kR) * barrier(k) / barrier(kR);

    complex<double> T = resonance_.elasticity * 0.5 * gammaK /
                        (resonance_.mass - sqrtS - 0.5 * i * gammaK);

    complex<double> scattered = T * (GL + i * FL);
    double delta = resonance_.decayOnly ? norm(scattered) / (rho * rho)
                                        : (norm(FL + scattered) - FL * FL) / (rho * rho);

    // Same normalisation as integrating GetIntegrand over t in [-1, 1]:
    // 2 pi * int dt |psi|^2 r^2 -> 4 pi r^2 sum_L (2L+1) |psi_L|^2 / rho^2
    return 4.0 * PhysicalConstants::PI * r * r * pow(PhysicalConstants::FM_TO_NU, 3) *
           (2 * L + 1) * resonance_.spinWeight * delta;
}

double CoulombWaveFunction::CalculateDCky(double k, double r,
                                          complex<double> scatteringLength,
                                          double effectiveRange,
                                          unsigned int integrationSteps) const {
    EtaContext ctx = PrecomputeEtaContext(k, scatteringLength, effectiveRange);
    return CalculateDCky(k, r, ctx, integrationSteps);
}
