#pragma once

#include <complex>
#include <cmath>
#include <Math/SpecFunc.h>

// sph_bessel(0, x) + i * sph_neumann(0, x)
std::complex<double> HankelPlus(const int l, const double x) // note: x > 0
{
    std::complex<double> v(0.0);
    if (x <= 0.0) {
        return v;
    }

    v = std::complex<double>( ROOT::Math::sph_bessel(l, x), ROOT::Math::sph_neumann(l, x) );
    return v;
}

std::complex<double> HankelMinus(const int l, const double x) // note: x > 0
{
    std::complex<double> v(0.0);
    if (x <= 0.0) {
        return v;
    }

    v = std::complex<double>( ROOT::Math::sph_bessel(l, x), -ROOT::Math::sph_neumann(l, x) );
    return v;
}


std::complex<double> DerivativeHankelPlus(const int l, const double x) // note: x > 0
{
    if (x <= 0) {
        return std::complex<double>(0.0);
    }
    return 1 / x * HankelPlus(l, x) - HankelPlus(l+1, x);
}

std::complex<double> DerivativeHankelMinus(const int l, const double x) // note: x > 0
{
    if (x <= 0) {
        return std::complex<double>(0.0);
    }
    return 1 / x * HankelMinus(l, x) - HankelMinus(l+1, x);
}
