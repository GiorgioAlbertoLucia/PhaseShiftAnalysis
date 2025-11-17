#include <complex>
#include <cmath>
#include <Math/SpecFunc.h>

/**
 * \brief Cross sectio defined from the partial waves expansion
 * \param xs [0] - theta (in radians)
 * \param pars [0] - k [fm^-1] (reduced mass * relative velocity) / hbar;
 *             [1] - eta (Sommerfeld parameter);
 *             [2] - nL (number of quantum numbers l to consider);
 *             [3] and beyond - phase shift for l=0,1,...
*/
double CrossSectionPartialWave(double* xs, double* pars)
{
    using namespace std::complex_literals;

    const double k = pars[0];                     // fm^-1
    const double eta = pars[1];
    const double nL = pars[2] >= 0 ? pars[2] : 0; // number of quantum number l to consider in the partial wave expansion

    const double theta = xs[0];
    std::vector<double> deltaLs(static_cast<int>(nL), 0.); // phase shift parameters

    std::vector<std::complex<double>> Uls(static_cast<int>(nL), 0.);
    std::vector<double> alphaLs(static_cast<int>(nL), 0.);

    const std::complex<double> csc2 = 1. / (std::sin(0.5 * theta)*std::sin(0.5 * theta));
    std::complex<double> complexTerm = 1i * eta * csc2 * std::exp(1i * eta * std::log(csc2));

    for (int iL = 0; iL < static_cast<int>(nL); iL++) {
        
        deltaLs[iL] = pars[3 + iL];
        
        alphaLs[iL] = 0;
        for (int is = 1; is < static_cast<int>(iL+1); is++) {
            alphaLs[iL] += std::atan(eta / is);
        }

        Uls[iL] = std::exp(2.*1i*alphaLs[iL]) * (1. - std::exp(2.*1i*deltaLs[iL]));
        complexTerm += ROOT::Math::legendre(iL, std::cos(theta)) * (2.*iL + 1.) * Uls[iL];
    }

    const double crossSection = 1. / (4. * k * k) * std::abs(complexTerm * std::conj(complexTerm)); // fm^2
    return 0.01 * crossSection;  // b

}
