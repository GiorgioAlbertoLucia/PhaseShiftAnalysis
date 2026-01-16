#include <iostream>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include "gsl/gsl_sf_coulomb.h"
#include "gsl/gsl_sf_gamma.h"
#include "gsl/gsl_sf_hyperg.h"
#include "gsl/gsl_integration.h"
#include "flint/flint.h"
#include "flint/fmpz.h"
#include "flint/fmpz_poly.h"
#include "flint/fmpq.h"
#include "flint/fmpq_poly.h"
#include "flint/arith.h"
#include "flint/nmod_poly.h"
#include "flint/nmod_poly_mat.h"
#include "flint/ulong_extras.h"
#include "flint/hypgeom.h"
#include "flint/acb.h"
#include "flint/acb_hypgeom.h"
#include "flint/arb.h"
#include "flint/double_interval.h"
#include "flint/arb_fpwrap.h"

#include "gmp.h"

#include "CATS.h"
#include "DLM_Integration.h"
#include "TString.h"

using namespace std;

const double Pig = 3.141592653589793;
const double hbarc = 197.3;
const double FmToNu = 1.0 / hbarc;
const double gamma = 0.5772;
const double mpi = 139.57039;
const double mXim = 1321.71;
const double a0 = 52917.724900001;
const std::complex<double> i(0, 1);

double chargeRag(double m1, double m2, double charge)
{
    return a0 * 0.510 / (m1 * m2 / (m1 + m2) * charge);
}

double Ac(double eta)
{
    return 2.0 * Pig * eta * 1.0 / (exp(2.0 * Pig * eta) - 1);
}

double h(double eta)
{
    if (fabs(eta) < 0.3)
        return (1.2 * eta * eta - log(fabs(eta)) - gamma);
    else
    {
        double sum = 0.0;
        for (int n = 1; n <= 100000; ++n)
            sum += 1 / (n * (n * n + eta * eta));
        return (eta * eta * sum - gamma - log(fabs(eta)));
    }
}

std::complex<double> fc(double k, std::complex<double> f0, double d0, double ac, double eta)
{
    return 1.0 / (1. / f0 + 0.5 * d0 * k * k - 2.0 / ac * h(eta) - i * k * Ac(eta));
}

std::complex<double> TildeG(double rho, double eta)
{

    // cout << "4) Inside TildeG" << endl;

    int kmax = 0;    // Number of functions to compute
    double fc_array; // Array to store F values WHAT IS THE CORRECT SIZE
    double gc_array; // Array to store G values
    double L_min = 0.;
    double OverflowF = 0;
    double OverflowG = 0;

    gsl_sf_coulomb_wave_FG_array(L_min, kmax, eta, fabs(rho), &fc_array, &gc_array, &OverflowF, &OverflowG);

    std::complex<double> result = std::sqrt(Ac(eta)) * (i * fc_array + gc_array);

    return result;
}

// std::complex<double> Hypergeometric1F1(std::complex<double> a, std::complex<double> b, std::complex<double> z)
// {
//     std::complex<double> result = 0.0;
//     std::complex<double> term = 1.0;

//     for (int k = 0; k < 100; ++k)
//     { // Summing up to 100 terms
//         result += term;
//         term *= (a + double(k)) / (b + double(k)) * z / (double(k) + 1);
//     }

//     return result;
// }

// std::complex<double> Fun(double eta, double zeta)
// {
//     // Convert real numbers to complex numbers
//     std::complex<double> eta_complex(0, eta);
//     std::complex<double> zeta_complex(0, zeta);

//     // Call the Hypergeometric1F1 function with the appropriate arguments
//     return Hypergeometric1F1(i*eta, 1.0, i*zeta);
// }

std::complex<double> Fun(double eta, double zeta)
{
    std::complex<double> eta_complex(0, eta);
    std::complex<double> zeta_complex(0, zeta);

    // Initialize FLINT's acb_t objects for the complex arguments
    acb_t eta_acb, zeta_acb;
    acb_init(eta_acb);
    acb_init(zeta_acb);

    // Set the real and imaginary parts of the acb_t objects
    acb_set_d_d(eta_acb, 0.0, eta);
    acb_set_d_d(zeta_acb, 0.0, zeta);

    // Compute the hypergeometric function using FLINT
    acb_t result_acb;
    acb_init(result_acb);
    int regularized = 0;

    acb_t b_value;
    acb_init(b_value);
    acb_set_d(b_value, 1.);
    acb_hypgeom_1f1(result_acb, eta_acb, b_value, zeta_acb, regularized, 64);


    double real_part = arf_get_d(arb_midref(acb_realref(result_acb)), ARF_RND_NEAR);
    double imag_part = arf_get_d(arb_midref(acb_imagref(result_acb)), ARF_RND_NEAR);

    std::complex<double> result(real_part, imag_part);
    flint_cleanup();
    return result;
}

std::complex<double> Psi(double k, double r, double t, std::complex<double> ScatLen, double EffecRange, double ChargeRad)
{
    // cout << "3) Inside Psi" << endl;

    double eta = 1.0 / (k * ChargeRad) / FmToNu;
    double rhoval = k * r * FmToNu;
    double zeta = rhoval * (1 + t);
    std::complex<double> f0 = ScatLen * FmToNu;
    double d0 = EffecRange * FmToNu;
    double ac = ChargeRad * FmToNu;
    double rval = r * FmToNu;
    // Compute the argument of the gamma function using gsl_sf_lngamma_complex_e
    gsl_sf_result lnr, arg;
    gsl_sf_lngamma_complex_e(1.0, eta, &lnr, &arg);
    double arg_value = arg.val;
    // cout << "Ac(eta) = " << Ac(eta) << endl;
    // cout << "arg_value = " << arg_value << endl;
    // cout << "k = " << k << "-"
    //      << "rval = " << rval << "-"
    //      << "t=" << t << endl;
    // cout << "eta = " << eta << "-"
    //      << "zeta =" << zeta << endl;
    // cout << "Fun(-eta, zeta) = " << Fun(-eta, zeta) << endl;
    // cout << "f0 = " << f0 / FmToNu << "-"
    //      << "d0 = " << d0 << "-"
    //      << "ac=" << ac << "-"
    //      << "rhoval=" << rhoval << endl;
    // cout << "fc(k, f0, d0, ac, eta) = " << fc(k, f0, d0, ac, eta) << endl;
    // cout << "TildeG(rhoval, eta) = " << TildeG(rhoval, eta) << endl;

    // cout << "----Piece by Piece-----------" << endl;
    // cout << "std::pow(Ac(eta), 0.5) = " << std::pow(Ac(eta), 0.5) << endl;
    // cout << "exp(i * arg_value) = " << exp(i * arg_value) << endl;
    // cout << "exp(-i * k * rval * t) = " << exp(-i * k * rval * t) << endl;
    // cout << "Fun(-eta, zeta) = " << Fun(-eta, zeta) << endl;
    // cout << "fc(k, f0, d0, ac, eta) = " << fc(k, f0, d0, ac, eta) << endl;
    // cout << "TildeG(rhoval, eta) = " << TildeG(rhoval, eta) << endl;
    // cout << "(exp(-i * k * rval * t) * Fun(-eta, zeta) + fc(k, f0, d0, ac, eta) * TildeG(rhoval, eta) / rval) = " << (exp(-i * k * rval * t) * Fun(-eta, zeta) + fc(k, f0, d0, ac, eta) * TildeG(rhoval, eta) / rval) << endl;

    std::complex<double> WF = std::pow(Ac(eta), 0.5) * exp(i * arg_value) * (exp(-i * k * rval * t) * Fun(-eta, zeta) + fc(k, f0, d0, ac, eta) * TildeG(rhoval, eta) / rval);
    // cout << "WF = " << WF << endl;
    // cout << "----------------------------------" << endl;
    // cout << "----------------------------------" << endl;
    // cout << "----------------------------------" << endl;
    // cout << "----------------------------------" << endl;

    return WF;
}

double integrand_function(double *params)
{
    // cout << "3) Inside integrand_function" << endl;

    double &k = params[0];
    double &t = params[1];
    double &rval = params[2];
    double &aRe = params[3];
    double &aIm = params[4];
    double &EffecRange = params[5];
    double &ChargeRad = params[6];

    std::complex<double> ScatLen = aRe + i * aIm;

    // Compute Psi at given arguments
    std::complex<double> psi_conjugate = std::conj(Psi(k, rval, t, ScatLen, EffecRange, ChargeRad));
    std::complex<double> psi = Psi(k, rval, t, ScatLen, EffecRange, ChargeRad);

    // Compute the integrand
    double integrand = std::abs(psi_conjugate * psi) * rval * rval * FmToNu * FmToNu * FmToNu * 2.0 * Pig;
    // cout << "integrand = " << integrand << endl;

    return integrand;
}

double dCky(double *params)
{
    // cout << "2) Inside dCky" << endl;
    double &k = params[0];
    double &t = params[1];
    double &rval = params[2];
    double &aRe = params[3];
    double &aIm = params[4];
    double &EffecRange = params[5];
    double &ChargeRad = params[6];

    DLM_INT_SetFunction(integrand_function, params, 1);

    unsigned NSteps = 64;
    double result = DLM_INT_SimpsonWiki(-1., +1., NSteps);

    return result;
}

int exec(int argc, char *argv[])
{
    printf("\033[0;33m Bah speremo \033[0m\n");

    double c1 = 1.0, c2 = -1.0;
    double chargeRagXiPi = chargeRag(mpi, mXim, c1 * c2);
    std::vector<double> aRe = {0.05, 0.1, 0.15, 0.2};
    std::vector<double> aIm = {0., 0.1, 0.2};

    // std::vector<double> aRe = {0.1, 0.2};
    // std::vector<double> aIm = {0., 0.2};

    double EffRange = 0.;
    std::complex<double> ScatLen(aRe[0], aIm[0]);

    double mom = 0.01;
    double t = 0.1;
    double rval = 0.01;
    double params[7] = {mom, t, rval, aRe[0], aIm[0], EffRange, chargeRagXiPi};
    
    // return 0;

    //TString OutputFolder = "/Users/sartozza/cernbox/Analysis/XiPi_Oton/LednickyCoulombCodes/OutputMiracle/";
    TString OutputFolder = "/Users/glucia/Projects/PhaseShiftAnalysis/ValentinaGhetto/outputPiXi/";

    double kmin = 5.;
    double kmax = 600; // 80.
    double kbinWdith = 5.;
    int nkbins = int((kmax - kmin) / kbinWdith);

    double rmin = 0.01;
    double rmax = 80.01; // 80.
    double rbinWdith = 0.2;
    int nrbins = int((rmax - rmin) / rbinWdith);

    for (unsigned Kbin = 0; Kbin < nkbins; Kbin++)
    {
        mom = kmin + (Kbin * kbinWdith);
        params[0] = mom;
        // cout << "mom = " << mom << "---" << endl;
        for (unsigned aRebin = 0; aRebin < aRe.size(); aRebin++)
        {
            params[3] = aRe[aRebin];
            //cout << "aRe = " << params[3] << "---" << endl;
            for (unsigned aImbin = 0; aImbin < aIm.size(); aImbin++)
            {
                params[4] = aIm[aImbin];
                //cout << "aIm = " << params[4] << "---" << endl;
                std::stringstream filenameStream;
                filenameStream << OutputFolder << "Cky_k" << std::fixed << std::setprecision(0) << mom << "_aRe" << std::fixed << std::setprecision(1) << aRe[aRebin] << "_aIm" << std::fixed << std::setprecision(1) << aIm[aImbin] << ".dat";
                std::string filename = filenameStream.str();
                std::ofstream outfile(filename);
                //cout << "....as a function of r...." << endl;
                for (unsigned Rbin = 0; Rbin < nrbins; Rbin++)
                {
                    rval = rmin + (Rbin * rbinWdith);
                    params[2] = rval;
                    outfile << std::fixed << std::setprecision(3) << rval << "\t" << std::scientific << std::setprecision(4) << dCky(params) << std::endl;
                }
                outfile.close();
                //cout << "------------------------" << endl;
            }
            //cout << "------------------------" << endl;
        }
        //cout << "------------------------" << endl;
    }

    // Close the file

    return 0;
}
