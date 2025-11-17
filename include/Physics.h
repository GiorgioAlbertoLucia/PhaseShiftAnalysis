
#pragma once

#include "TMath.h"
#include "Math/Vector4D.h"
#include "Math/Boost.h"

namespace constant {
    // Physical constants
    constexpr double HBARC = 197.3269804; // MeV·fm
    constexpr double M_PROTON = 938.272;   // MeV/c²
    constexpr double M_HE3 = 2808.391;     // MeV/c²
    constexpr double E_CHARGE_SQ = 1.44;   // e² in MeV·fm
    constexpr double Z_PROTON = 1.0;
    constexpr double Z_HE3 = 2.0;
    constexpr double ALPHA_EM = 0.0072973525643;
} // namespace constant


// Reduced mass calculation
inline double reducedMass(double m1, double m2) {
    return m1 * m2 / (m1 + m2);
}

// Wave number calculation
inline double waveNumber(double E_lab, double mu) {
    const double v = TMath::Sqrt(2. * E_lab / constant::M_PROTON); // incident proton beam
    return v * mu / constant::HBARC;
}

inline double waveNumber(const double momentum) { return momentum / constant::HBARC; }

double getKstar(const double incidentMomentum) {

    ROOT::Math::PxPyPzMVector incidentPmu(0., 0., incidentMomentum, constant::M_PROTON);
    ROOT::Math::PxPyPzM4D targetPmu(0., 0., 0., constant::M_HE3);

    auto pBetaVector = (incidentPmu + targetPmu).BoostToCM();
    double beta_px = 0., beta_py = 0., beta_pz = 0.;
    pBetaVector.GetCoordinates(beta_px, beta_py, beta_pz);
    ROOT::Math::Boost pBoost(beta_px, beta_py, beta_pz);

    ROOT::Math::PxPyPzMVector incidentPmuToBoost(0., 0., incidentMomentum, constant::M_PROTON);
    ROOT::Math::PxPyPzM4D targetPmuToBoost(0., 0., 0., constant::M_HE3);
    auto p1muStar = pBoost(incidentPmuToBoost);
    auto p2muStar = pBoost(targetPmuToBoost);

    auto kmuStar = (p1muStar - p2muStar);
    return 0.5 * kmuStar.P();
}

// Coulomb potential
inline double coulombPotential(double r, double Z1, double Z2) {
    return constant::E_CHARGE_SQ * Z1 * Z2 / r;
}
