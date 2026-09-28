#include "Config.h"
#include <cmath>

double CalculationConfig::GetChargeRadius() const {
    double reducedMass = (particle1.mass * particle2.mass) / (particle1.mass + particle2.mass);
    double chargeProduct = particle1.charge * particle2.charge;
    return PhysicalConstants::BOHR_RADIUS * 0.510 / (reducedMass * chargeProduct);
}

namespace DefaultConfigs {
    CalculationConfig GetXiPiConfig() {
        CalculationConfig config;
        
        // Xi-Pi masses and charges
        config.particle1 = ParticleProperties(139.57039, 1.0);  // pi-
        config.particle2 = ParticleProperties(1321.71, -1.0);     // Xi-
        
        // Source sizes
        //config.sourceSizes = {1.19, 1.15, 1.24, 3.16, 3.12, 3.21};
        config.sourceSizes = {1.19};
        
        // Scattering lengths to explore
        //config.realScatteringLengths = {0.1, 0.2, 0.3, 0.4, 0.5};
        //config.imagScatteringLengths = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
        config.realScatteringLengths = {0.2};
        config.imagScatteringLengths = {0.0};
        
        // Output settings
        config.outputFolder = "/Users/glucia/Projects/PhaseShiftAnalysis/numerical_lednicky/piXi/dat/";
        config.outputRootFolder = "/Users/glucia/Projects/PhaseShiftAnalysis/numerical_lednicky/piXi/output/";
        config.outputRootFile = "TheoCF_XiPiFree2G.root";
        
        return config;
    }

    CalculationConfig GetPHeConfig() {
        CalculationConfig config;

        config.kBinWidth = 0.5;
        
        // Xi-Pi masses and charges
        config.particle1 = ParticleProperties(938.272, 1.0);  // p
        config.particle2 = ParticleProperties(2808.391, 2.0); // He3
        
        // Source sizes
        // Values from parameterisation (mT scaling)
        //config.sourceSizes = {6.12, 6.12 - 0.10, 6.12 + 0.11,
        //                     4.90, 4.90 - 0.05, 4.90 + 0.04,};
        // Values from Mrowczysnki
        config.sourceSizes = {5.34, 5.34 - 0.11, 5.34 + 0.10,
                             4.24, 4.24 - 0.06, 4.24 + 0.06,};
        // Source size scan
        //config.sourceSizes = {};
        //for (double r = 3.5; r <= 8.0; r += 0.1) {
        //  config.sourceSizes.push_back(r);
        //}

        /* The Gaussian source in the code is defined as exp(-r^2 / (4 * R^2)), where R is the source size parameter.
         * The usual parameterisation considers the Gaussian to be exp(-r^2 / (2 * R^2)), so we need to convert the source sizes accordingly. The conversion is R_new = R_old / sqrt(2).
        */
        for (auto& size : config.sourceSizes) {
            size = size / std::sqrt(2.);
        }
        
        // Scattering lengths to explore 
        // SIGN CONVENTION: ( f_c = 1 / (1/f0 + 0.5*d0*k^2 - 2/ac*H(eta) - i*k*A_c(eta)) )
        // Values taken from the NLO calculations of https://arxiv.org/pdf/2507.16250
        // variations are obtained as follows: nominal = n, nominal + err = n + e, nominal - err = n - e
        // a = {n0, n0-e0, n0+e0, n0, n0, n1, n1-e1, n1+e1, n1, n1}
        // r = {n0, n0, n0, n0-e0, n0+e0, n1, n1, n1, n1-e1, n1+e1}
        // config.realScatteringLengths = {-11.26, -11.30, -11.22, -11.26, -11.26, -9.06, -9.10, -9.14, -9.06, -9.06}; // fm
        // config.imagScatteringLengths = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // fm
        // config.effectiveRanges = {1.65, 1.65, 1.65, 1.36, 1.94, 1.36, 1.36, 1.36, 1.11, 1.61};   // fm
        
        // without variations
        config.realScatteringLengths = {-11.26, -9.06}; // fm
        config.imagScatteringLengths = {0.0, 0.0}; // fm
        config.effectiveRanges = {1.65, 1.36};   // fm
        
        // Output settings
        config.outputFolder = "/home/galucia/PhaseShiftAnalysis/numerical_lednicky/pHe/dat/";
        config.outputRootFolder = "/home/galucia/PhaseShiftAnalysis/numerical_lednicky/pHe/output/";
        config.outputRootFile = "TheoCF_PHe.root";
        
        return config;
    }
}