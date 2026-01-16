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
        
        // Xi-Pi masses and charges
        config.particle1 = ParticleProperties(938.272, 1.0);  // p
        config.particle2 = ParticleProperties(2808.391, 2.0); // He3
        
        // Source sizes
        config.sourceSizes = {2., 3., 4., 5., 6., 7.};
        
        // Scattering lengths to explore 
        // SIGN CONVENTION: ( f_c = 1 / (1/f0 + 0.5*d0*k^2 - 2/ac*H(eta) - i*k*A_c(eta)) )
        // Values taken from the NLO calculations of https://arxiv.org/pdf/2507.16250
        config.realScatteringLengths = {-11.26, -9.06}; // fm
        config.imagScatteringLengths = {0.0};
        config.effectiveRange = 1.65;   // fm
        
        // Output settings
        config.outputFolder = "/Users/glucia/Projects/PhaseShiftAnalysis/numerical_lednicky/pHe/dat/";
        config.outputRootFolder = "/Users/glucia/Projects/PhaseShiftAnalysis/numerical_lednicky/pHe/output/";
        config.outputRootFile = "TheoCF_PHe.root";
        
        return config;
    }
}