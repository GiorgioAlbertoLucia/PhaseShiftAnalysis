/*
    Implementation of the Newton-Raphson algorithm to find the Phase Shift by matching the analytical solution to 
    the numerical solution
*/
#pragma once

#include <cmath>
#include <stdexcept>
#include <functional>

/**
 * The phase shift delta is found as the root of the following equation. Given
 * n_l(r): numerical solution to the Schroedinger's equation
 * a_l(r): analytical solution to the Schroedinger's equation in the asymptotic regime
 * N n_l(r) = a_l(r + delta/k)
 *  with N generic normalisation constant, delta the phase shift, k the wavenumber
 * 
 * Given an arbitrary point r + Dr, the equation still holds. The ratio of the two equation is
 *   a_l(r + Dr + delta/k)/a_l(r + delta/k) = n_l(r + Dr)/n_l(r)
 * 
 * Therefore, delta is the root to the equation
 * f(delta) = a_l(r + Dr + delta/k)/a_l(r + delta/k) - n_l(r + Dr)/n_l(r) = 0
*/
class NewtonRaphsonPhaseShift
{
    public:
        NewtonRaphsonPhaseShift() = default;
        NewtonRaphsonPhaseShift(const std::vector<double>& rs, const std::vector<double>& nl,
                                const std::function<double(double)> al, const double r, const double Dr, const double k):
            mRs(rs), mNl(nl), mAl(al), mR(r), mDr(Dr), mK(k) {}
        ~NewtonRaphsonPhaseShift() = default;

        inline void setRadii(const std::vector<double>& rs) { mRs = rs; }
        inline void setNumericalSolution(const std::vector<double>& nl) { mNl = nl; }
        inline void setWavenumber(const double k) { mK = k; }
        double computePhaseShift(const double initialGuess = 0., const double tolerance = 1e-10, const int maxIter = 100) const;

    protected:
        double interpolateNl(const double r) const;
        double computeEquation(const double delta) const;
        double computeDerivative(const double delta, const double h = 1e-5) const;

    private:
        std::vector<double> mRs;            // considered radius values 
        std::vector<double> mNl;            // numerical solution
        std::function<double(double)> mAl;  // asymptotic solution
        double mK;                          // k: wavenumber at which the phase shift is computed
        double mR;                          // R value at which the matching is performed
        double mDr;                         // Delta R value at which the matching is performed
};

/**
 * Return n_l(r) with an interpolation
 * The method assumes a uniform grid
*/
double NewtonRaphsonPhaseShift::interpolateNl(const double r) const 
{   
    const size_t sizeR = mRs.size();
    if (r <= mRs[0]) return mNl[0];
    if (r >= mRs[sizeR-1]) return mNl[sizeR-1];
    
    // Direct calculation assuming uniform grid
    double dr = mRs[1] - mRs[0];
    double idx = (r - mRs[0]) / dr;
    size_t ir = static_cast<size_t>(idx);
    
    double t = idx - ir;  // fractional part
    return mNl[ir] + t * (mNl[ir+1] - mNl[ir]);
}

/**
 * Compute the equation to find the root for
*/
double NewtonRaphsonPhaseShift::computeEquation(const double delta) const {
    
    const double nr = interpolateNl(mR);
    const double nr_dr = interpolateNl(mR + mDr);

    const double ar = mAl(mR + delta/mK);
    const double ar_dr = mAl(mR + mDr + delta/mK);

    return ar_dr/ar - nr_dr/nr;
}

/**
 * Compute numerical derivative of the equation for the root (used in the Newton-Raphson iterations)
*/
double NewtonRaphsonPhaseShift::computeDerivative(const double delta, const double h) const {
    double f_plus = computeEquation(delta + h);
    double f_minus = computeEquation(delta - h);
    return (f_plus - f_minus) / (2.0 * h);
}

/**
 * Compute the phase shift delta as the root of the equation
 * rho = delta / k, where delta is the phase shift, k the wavenumber
*/
double NewtonRaphsonPhaseShift::computePhaseShift(const double initialGuess, const double tolerance, const int maxIter) const
{
    double rho = initialGuess;
    double hDeriv = 1e-5;
    const double minStep = 1e-12;

    for (int iter = 0; iter < maxIter; ++iter) {
        double f = computeEquation(rho);

        if (std::abs(f) < tolerance) {
            //std::cout << "convergence found: f = " << f << std::endl;
            return rho * mK;
        }

        double df = computeDerivative(rho, hDeriv);
        if (std::abs(df) < 1e-14 || !std::isfinite(df))
            throw std::runtime_error("Derivative too small or invalid in Newton-Raphson");

        double step = -f / df;
        double newRho = rho + step;

        // --- Domain-safe evaluation with damping ---
        double fNew = std::numeric_limits<double>::quiet_NaN();
        int guardCount = 0;

        // First, ensure fNew is finite and not worse than f
        while (guardCount < 10) {
            fNew = computeEquation(newRho);

            if (!std::isfinite(fNew) || std::abs(fNew) > 1e15) {
                // Invalid or exploded value → backtrack
                step *= 0.5;
                newRho = rho + step;
                ++guardCount;
                continue;
            }

            // If f got worse, damp the step
            if (std::abs(fNew) > std::abs(f) && std::abs(step) > minStep) {
                step *= 0.5;
                newRho = rho + step;
                ++guardCount;
                continue;
            }

            break; // accepted
        }

        if (!std::isfinite(fNew))
            throw std::runtime_error("NewtonRaphson: f(x) remained invalid after backtracking");

        rho = newRho;

        // --- Secondary convergence check ---
        if (std::abs(fNew) < tolerance || std::abs(step) < 1e-10) {
            //std::cout << "convergence found: f = " << fNew << std::endl;
            return rho * mK;
        }

        // --- Adaptive derivative step: enlarge h if f hardly changes ---
        if (std::abs(fNew - f) < 1e-12)
            hDeriv *= 10.0;
    }

    throw std::runtime_error("Newton-Raphson did not converge");
}


//double NewtonRaphsonPhaseShift::computePhaseShift(const double initialGuess, const double tolerance, const int maxIter) const {
//    double delta = initialGuess;
//    
//    for (int iter = 0; iter < maxIter; iter++) {
//        double f = computeEquation(delta);
//        std::cout << "NewtonRaphson: f = " << f << std::endl;
//        
//        if (std::abs(f) < tolerance) {
//            std::cout << "\tfound delta, f = " << f << std::endl;
//            return delta;  // Converged
//        }
//        
//        double df = computeDerivative(delta);
//        
//        if (std::abs(df) < 1e-15)
//            throw std::runtime_error("Derivative too small, Newton-Raphson failed");
//        
//        double step = -f / df;
//        if (std::abs(step) > 0.1)
//            step = 0.1 * step / std::abs(step);
//        delta += step;
//
//        //delta = delta - f / df;
//    }
//    
//    throw std::runtime_error("Newton-Raphson did not converge");
//}
