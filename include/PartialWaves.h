#pragma once

#include <iostream>
#include "Physics.h"

#include <TSystem.h>

#include "gsl/gsl_sf_coulomb.h"
#include "gsl/gsl_errno.h"


/**
 * Coulomb partial wave as hypergeometric function
*/
double coulombPartialWave(const double /*r*/)
{
    std::cout << "Not implemented yet!\n";
    return 0;
}

/**
 * Returns the function for the Coulomb partial wave depending on r only
 * @param l quantum number l
 * @param k wavenumber
 * @param Z1 charge of particle 1
 * @param Z2 charge of particle 2
 * @param mass1 mass of particle 1 (in MeV)
 * @param mass2 mass of particle 2 (in MeV)
*/
std::function<double (double)> generateCoulombPartialWave(const int l, const double k, 
                                                          const int Z1, const int Z2, const float mass1, const float mass2)
{
    return [k, l, mass1, mass2, Z1, Z2](double r) -> double {
        const double mu = reducedMass(mass1, mass2);
        double partialWave = 0, overflowExponent = 0;
        const double kstar = k * constant::HBARC;
        const double eta = constant::ALPHA_EM * Z1 * Z2 * mu / kstar;   // this should not be the correct calculation of eta
        
        if (r == 0) return 0.0;
        double rho = k * r;

        if (std::isnan(eta) || std::isnan(rho) || std::isinf(eta) || std::isinf(rho)) {
            std::cerr << "Invalid GSL input: eta=" << eta << ", rho=" << rho 
                      << ", r=" << r << ", k=" << k << ", lmin=" << l << std::endl;
            return 0.0;
        }

        // GSL requires rho > 0 and eta must be reasonable
        if (std::abs(rho) < 1e-10) return 0.0;

        int status = gsl_sf_coulomb_wave_F_array(l, 0, eta, std::abs(rho), &partialWave, &overflowExponent);
        //int status = gsl_sf_coulomb_wave_sphF_array(l, 0, eta, std::abs(rho), &partialWave, &overflowExponent);

        if (status != GSL_SUCCESS) {
            std::cerr << "GSL error " << status << ": lmin=" << l 
                      << ", eta=" << eta << ", rho=" << rho << std::endl;
            return 0.0;
        }

        double result = partialWave / k;

        if (rho < 0 && l % 2 == 1) {
            result = -result;
        }

        return result;
    };

    //return [k, l, mass1, mass2, Z1, Z2](double r) -> double {
    //    
    //    const double mu = reducedMass(mass1, mass2);
    //    double partialWave = 0, overflowExponent = 0;
    //    const double kstar = k * constant::HBARC;
    //    const double eta = constant::ALPHA_EM * Z1 * Z2 * mu / kstar;
    //    if (r == 0) 
    //        return 0.;
    //    gsl_sf_coulomb_wave_F_array(l, 0, eta, std::abs(kstar*r), &partialWave, &overflowExponent);
    //    return partialWave;
    //};
}
