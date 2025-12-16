#pragma once

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numeric>

#include <TMath.h>
#include <TRandom3.h>

#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_coulomb.h>
#include <gsl/gsl_integration.h>

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>

// Constants class
namespace Constants {
    const double M_PROTON = 938.272;    // GeV/c^2
    const double M_DEUTERON = 1875.613;  // GeV/c^2
    const double M_HELIUM3 = 2808.391;  // GeV/c^2
    const double Z_PROTON = 1.0;
    const double Z_DEUTERON = 1.0;
    const double Z_HELIUM3 = 2.0;
    const double HBARC = 197.3;          // MeV·fm
    const double ALPHA_EM = 1.0/137.036;
    const double PI = 3.141592653589793238462643383279502884;	/* pi */
    const double EULER = 5.772156649015328606065120900824024310e-01;
}

namespace math {

    /**
     * Computes the hypergeometric 1F1(- i eta, 1, i xi)
    */
    std::complex<double> hypergeometric_1F1_complex(double eta, double xi, double tol = 1e-12, int max_terms = 200) {
        const std::complex<double> a(0., -eta);
        const std::complex<double> b(1., 0.);
        const std::complex<double> z(0., xi);
        
        std::complex<double> term(1., 0.);
        std::complex<double> sum = term;
        
        for (int n = 1; n < max_terms; ++n) {
            term *= (a + double(n-1)) * z / double(n*n);  // b=1 simplifies to just /n, then you have the factorial as well
            sum += term;
            
            if (std::abs(term) < tol * std::abs(sum)) {
                return sum;
            }
        }
        
        // Convergence warning
        std::cerr << "Warning: hypergeometric_1F1 series did not converge after " 
                << max_terms << " terms" << std::endl;
        return sum;
    }


    const double PI = 3.141592653589793238462643383279502884;

    // Cotangent of pi*x
    inline double cotpi(double x) {
        return 1.0 / std::tan(PI * x);
    }

    // Coefficients from mpmath
    const std::vector<double> _psi_coeff = {
        0.083333333333333333333,
    -0.0083333333333333333333,
        0.003968253968253968254,
    -0.0041666666666666666667,
        0.0075757575757575757576,
    -0.021092796092796092796,
        0.083333333333333333333,
    -0.44325980392156862745,
        3.0539543302701197438,
    -26.456212121212121212
    };

    // Real digamma
    double digamma_real(double x) {
        int intx = static_cast<int>(x);
        if (x == intx && intx <= 0) {
            throw std::runtime_error("digamma pole at non-positive integer");
        }

        double s = 0.0;

        if (x < 0.5) {
            x = 1.0 - x;
            s = PI * cotpi(x);
        }

        while (x < 10.0) {
            s -= 1.0 / x;
            x += 1.0;
        }

        double x2 = 1.0 / (x * x);
        double t = x2;

        for (double c : _psi_coeff) {
            s -= c * t;
            if (t < 1e-20) break;
            t *= x2;
        }

        return s + std::log(x) - 0.5 / x;
    }

    // Complex digamma
    std::complex<double> digamma_complex(std::complex<double> x) {
        // If purely real, use real version
        if (x.imag() == 0.0) return std::complex<double>(digamma_real(x.real()), 0.0);

        std::complex<double> s = 0.0;

        if (x.real() < 0.5) {
            x = 1.0 - x;
            s = PI / std::tan(PI * x);  // cot(pi x) = 1/tan(pi x)
        }

        while (std::abs(x) < 10.0) {
            s -= 1.0 / x;
            x += 1.0;
        }

        std::complex<double> x2 = 1.0 / (x * x);
        std::complex<double> t = x2;

        for (double c : _psi_coeff) {
            s -= c * t;
            if (std::abs(t) < 1e-20) break;
            t *= x2;
        }

        return s + std::log(x) - 0.5 / x;
    }

}
