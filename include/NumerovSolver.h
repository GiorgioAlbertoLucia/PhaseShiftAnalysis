#pragma once

#include <vector>
#include "Potentials.h"
#include "Physics.h"
#include "NewtonRaphsonPhaseShift.h"
#include "PartialWaves.h"
#include "Potentials.h"

// Numerov algorithm for radial Schrödinger equation
class NumerovSolver {

    public:

        NumerovSolver(const double dr = 0.01, const double rMaxVal = 20.0, 
                      const double rAsymptotic = 18.0, const double drAsymptotic = 1.0) 
            : mDr(dr), mRmax(rMaxVal), mNpoints(static_cast<int>(mRmax / dr)),
            mRasymptotic(rAsymptotic), mDrAsymptotic(drAsymptotic) {}

        double solveAndExtractPhase(int l, double k, const Potential& pot, double Z1, double Z2) const;
        std::vector<double> solveEquation(const int l, const double k, const Potential& pot, const double Z1, const double Z2) const;

    protected:
        
    
    private:
        double mDr;
        double mRmax;
        int mNpoints;

        double mRasymptotic, mDrAsymptotic;
};

double NumerovSolver::solveAndExtractPhase(int l, double k, const Potential& pot,
                            double Z1, double Z2) const {
    
    std::vector<double> yl(mNpoints);   // radial solution to the Schroedinger's equation
    std::vector<double> r(mNpoints);
    std::vector<double> V(mNpoints);

    for (int i = 0; i < mNpoints; ++i) {
        r[i] = (i + 1) * mDr;
        V[i] = pot.eval(r[i]) + coulombPotential(r[i], Z1, Z2);
    }

    double mu = reducedMass(constant::M_PROTON, constant::M_HE3);
    std::vector<double> kSq(mNpoints);
    for (int i = 0; i < mNpoints; ++i) {
        const double E = (k*k * constant::HBARC*constant::HBARC) / (2.0 * mu);
        kSq[i] = - 2.* mu/(constant::HBARC*constant::HBARC) * V[i] - l*(l+1)/(r[i]*r[i]) + 2.*mu*E/(constant::HBARC*constant::HBARC);
    }

    // initial conditions
    yl[0] = TMath::Power(r[0], l + 1);
    yl[1] = TMath::Power(r[1], l + 1);

    double h2_12 = mDr * mDr / 12.0;
    for (int i = 1; i < mNpoints - 1; ++i) {
        double f0 = 1.0 + h2_12 * kSq[i - 1];
        double f1 = 1.0 - 5.0 * h2_12 * kSq[i];
        double f2 = 1.0 + h2_12 * kSq[i + 1];

        yl[i + 1] = (2.0 * yl[i] * f1 - yl[i - 1] * f0) / f2;
    }

    const int Z_PROTON = 1, Z_HE3 = 2;
    std::function<double (double)> coulombPartialWave = generateCoulombPartialWave(l, k, Z_PROTON, Z_HE3, constant::M_PROTON, constant::M_HE3);

    NewtonRaphsonPhaseShift rootFinder(r, yl, coulombPartialWave, mRasymptotic, mDrAsymptotic, k);
    rootFinder.setRadii(r);
    rootFinder.setWavenumber(k);
    rootFinder.setNumericalSolution(yl);

    double delta = rootFinder.computePhaseShift();
    while (delta > TMath::Pi() / 2.)
        delta -= TMath::Pi();
    while (delta < -TMath::Pi() / 2.)
        delta += TMath::Pi();
    return delta;

}

std::vector<double> NumerovSolver::solveEquation(const int l, const double k, const Potential& pot,
                            const double Z1, const double Z2) const {
    
    std::vector<double> yl(mNpoints);   // radial solution to the Schroedinger's equation
    std::vector<double> r(mNpoints);
    std::vector<double> V(mNpoints);

    for (int i = 0; i < mNpoints; ++i) {
        r[i] = (i + 1) * mDr;
        V[i] = pot.eval(r[i]) + coulombPotential(r[i], Z1, Z2);
    }

    double mu = reducedMass(constant::M_PROTON, constant::M_HE3);
    std::vector<double> kSq(mNpoints);
    for (int i = 0; i < mNpoints; ++i) {
        const double E = (k*k * constant::HBARC*constant::HBARC) / (2.0 * mu);
        kSq[i] = - 2.* mu/(constant::HBARC*constant::HBARC) * V[i] - l*(l+1)/(r[i]*r[i]) + 2.*mu*E/(constant::HBARC*constant::HBARC);
    }

    // initial conditions
    yl[0] = TMath::Power(r[0], l + 1);
    yl[1] = TMath::Power(r[1], l + 1);

    double h2_12 = mDr * mDr / 12.0;
    for (int i = 1; i < mNpoints - 1; ++i) {
        double f0 = 1.0 + h2_12 * kSq[i - 1];
        double f1 = 1.0 - 5.0 * h2_12 * kSq[i];
        double f2 = 1.0 + h2_12 * kSq[i + 1];

        yl[i + 1] = (2.0 * yl[i] * f1 - yl[i - 1] * f0) / f2;
    }

    return yl;
}
