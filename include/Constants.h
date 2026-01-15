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

    std::complex<double> complex_gamma(const std::complex<double>& z) {
        gsl_sf_result lnr, arg;
        gsl_sf_lngamma_complex_e(z.real(), z.imag(), &lnr, &arg);
        return std::exp(std::complex<double>(lnr.val, arg.val));
    }

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

    std::complex<double> hypergeometric_1F1_complex(std::complex<double> a, std::complex<double> b, std::complex<double> z, double tol = 1e-12, int max_terms = 200) {
        
        std::complex<double> term(1., 0.);
        std::complex<double> sum = term;
        
        for (int n = 1; n < max_terms; ++n) {
            term *= (a + double(n-1)) * z / ((b + double(n-1)) * std::complex<double>(n, 0.));  // b=1 simplifies to just /n, then you have the factorial as well
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
    
    // ====================================================================================
    // from https://github.com/scottedwardpratt/commonutils/blob/a08fbfcbf646b5b5f6fa1efa610da174e52cc152/software/src/SpecialFunctions/Bessel/bess.cc

    void Bessel_JN_real(int l, double x, double &jl, double &nl, double &jlprime, double &nlprime){
        int il = 0;
        double j[4], n[4], jprime[4], nprime[4];
        double s = sin(x);
        double c = cos(x);
        if(l > 3){
            printf("increase array size in CalcJN_real\n");
            exit(1);
        }
        j[0] = s;
        n[0] = c;
        jprime[0] = c;
        nprime[0] = -s;
        while(il < l){
            il += 1;
            j[il] = (double(il)/x) * j[il-1] - jprime[il-1];
            n[il] = (double(il)/x) * n[il-1] - nprime[il-1];
            jprime[il] = j[il-1] - (double(il)/x) * j[il];
            nprime[il] = n[il-1] - (double(il)/x) * n[il];
        };
        jl = j[l];
        nl = n[l];
        jlprime = jprime[l];
        nlprime = nprime[l];
    }

    void Bessel_JN_imag(int l, double x, double &jl, double &nl, double &jlprime, double &nlprime){
        int il = 0;
        double j[4], n[4], jprime[4], nprime[4];
        double s = sinh(x);
        double c = cosh(x);
        if(l>3){
            printf("increase array size in CalcJN_imag\n");
            exit(1);
        }
        j[0] = s;
        n[0] = c;
        jprime[0] = c;
        nprime[0] = s;
        while(il < l){
            il += 1;
            j[il] = (double(il)/x) * j[il-1] - jprime[il-1];
            n[il] = (double(il)/x) * n[il-1] - nprime[il-1];
            jprime[il] = - j[il-1] - (double(il)/x) * j[il];
            nprime[il] = - n[il-1] - (double(il)/x) * n[il];
        };
        jl = j[l];
        nl = n[l];
        jlprime = jprime[l];
        nlprime = nprime[l];
    }


    // ====================================================================================
    // from https://github.com/scottedwardpratt/commonutils/blob/a08fbfcbf646b5b5f6fa1efa610da174e52cc152/software/src/SpecialFunctions/CoulWave/coulwave.cc
    void FGprime(int L, double x, double eta, double *FL, double *GL, double *FLprime, double *GLprime){
        
        double expF, expG;
        int k=0;
        double * fc = new double[k+1];
        double * gc = new double[k+1];
        double * fcp = new double[k+1];
        double * gcp = new double[k+1];
        // This calculates fc and gc arrays for indices L to L+k  
        gsl_sf_coulomb_wave_FGp_array(L, k, eta, x, fc, fcp, gc, gcp, 
        &expF, &expG);
        *FL = fc[0] * exp(expF);
        *GL = gc[0] * exp(expG);
        *FLprime = fcp[0] * exp(expF);
        *GLprime = gcp[0] * exp(expG);
        delete [] fc;
        delete [] gc;
        delete [] fcp;
        delete [] gcp;
    }

    void FGprime_ImagQ(int lmax, double x, double eta, double *FL, double *GL, double *FLprime, double *GLprime){
        
        double * F = new double[lmax+1]; 
        double * G = new double[lmax+1]; 
        double * Fprime = new double[lmax+1]; 
        double * Gprime = new double[lmax+1];
        
        double ff, root, sign;
        int n, l, nmax;
        
        nmax = 24 + lrint(fabs(x));
        double * A = new double[nmax+1];
        double * B = new double[nmax+1];
        
        // Calc F and Fprime
        A[0] = 0.0; A[1] = 1.0;
        for (n = 0; n <= nmax; n++) 
            B[n] = 0.0;
        for (n = 0; n <= nmax - 2; n++) 
            A[n+2] = (2.0*eta*A[n+1]+A[n]) / double((n+1)*(n+2));
        F[0] = Fprime[0] = 0.0;	
        for (n = 1; n <= nmax; n++){
            F[0] += A[n] * std::pow(x, n);
            Fprime[0] += double(n) * A[n] * std::pow(x,n-1);
        }
        for (l = 0; l < lmax; l++){
            root = (l + 1.0)*(l + 1.0) -eta*eta;
            if (root >= 0){
                root = std::sqrt(root);
                sign = 1;
            }
            else {
                root=sqrt(-root);
                sign=-1;
            }
            //printf("root=%g\n",root);
            ff = ((l+1.0)*(l+1.0) / x) + eta;
            F[l+1] = (ff*F[l] - (l+1.0)*Fprime[l]) / root;
            Fprime[l+1] =- (sign*root*F[l] + ff*F[l+1]) / (l+1.0);
        }
        // CALC G and Gprime
        A[0] = 1.0; A[1] = 0.0; B[1] = 2.0*eta*A[0]; B[0] = 0.0;
        for(n = 0;n<= nmax-2;n++){
            B[n+2] = (2.0*eta*B[n+1] + B[n]) / ((n+1.0)*(n+2.0));
            A[n+2] = (2.0*eta*A[n+1] + A[n] + (1.0-2.0*(n+2.0))*B[n+2]) / ((n+1.0)*(n+2.0));
        }
        G[0] = 1; Gprime[0] = 0.0;
        for(n = 1;n <= nmax;n++){
            G[0] += (A[n]+B[n]*std::log(fabs(x))) * std::pow(x,n);
            Gprime[0] += double(n)*(A[n]+B[n]*std::log(fabs(x)))*std::pow(x,n-1)+B[n]*std::pow(x,n-1);
        }
        for(l = 0; l < lmax; l++){
            root = (l+1.0)*(l+1.0) - eta*eta;
            if(root >= 0){
                root = sqrt(root);
                sign = 1;
            }
            else{
                root = sqrt(-root);
                sign = -1;
            }
            ff = ((l+1.0)*(l+1.0) / x) + eta;
            G[l+1] = (ff*G[l] - (l+1.0)*Gprime[l]) / root;
            Gprime[l+1] = -(sign*root*G[l] + ff*G[l+1]) / (l+1.0);
        }
        
        *FL = F[lmax];
        *GL = G[lmax];
        *FLprime = Fprime[lmax];
        *GLprime = Gprime[lmax];

        delete[] A;
        delete[] B;
        delete[] F;
        delete [] G;
        delete [] Fprime;
        delete [] Gprime;
    }

    // Coulomb wave functions with complex arguments
    void FGprime_ComplexQ(int l, std::complex<double> x, std::complex<double> eta,
                                            double* F, double* G, double* Fprime, double* Gprime) {
        
        if(fabs(std::imag(x)) > fabs(std::real(x))){
            // Using the recurrence relation to get the derivatives of F and G
            // (L+1)du/dx = ((L+1)^2/x + eta)u - sqrt((L+1)^2+eta^2)uL+1
            if(abs(eta)>1.0E-8)
                math::FGprime_ImagQ(l, std::imag(x), -std::imag(eta), F, G, Fprime, Gprime);
            else
                Bessel_JN_imag(l, std::imag(x), *F, *G, *Fprime, *Gprime);
        }
        else{
            if(std::abs(eta)>1.0E-8)
                math::FGprime(l, std::real(x), std::real(eta), F, G, Fprime, Gprime);
            else
                Bessel_JN_real(l, std::real(x), *F, *G, *Fprime, *Gprime);
        }
        return;
        
    }
}
