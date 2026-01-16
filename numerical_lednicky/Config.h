#pragma once 

#include <string>
#include <vector>

// Physical constants
struct PhysicalConstants {
    static constexpr double PI = 3.141592653589793;
    static constexpr double HBARC = 197.3;
    static constexpr double FM_TO_NU = 1.0 / HBARC;
    static constexpr double GAMMA = 0.5772;
    static constexpr double BOHR_RADIUS = 52917.724900001; // a0 in atomic units
};

// Particle properties
struct ParticleProperties {
    double mass;
    double charge;
    
    ParticleProperties(double m = 0.0, double q = 0.0) : mass(m), charge(q) {}
};

// Calculation parameters
struct CalculationConfig {
    // Momentum grid
    double kMin = 5.0;
    double kMax = 550.0;
    double kBinWidth = 5.0;
    
    // Radial grid
    double rMin = 0.01;
    double rMax = 80.01;
    double rBinWidth = 0.2;
    
    // Integration parameters
    unsigned int integrationSteps = 64;
    
    // Source sizes for correlation function
    std::vector<double> sourceSizes;
    
    // Scattering parameters
    std::vector<double> realScatteringLengths;
    std::vector<double> imagScatteringLengths;
    double effectiveRange = 0.0;
    
    // Particle properties
    ParticleProperties particle1;
    ParticleProperties particle2;
    
    // Output settings
    std::string outputFolder;
    std::string outputRootFolder;
    std::string outputRootFile;
    
    int GetNumKBins() const { return static_cast<int>((kMax - kMin) / kBinWidth); }
    int GetNumRBins() const { return static_cast<int>((rMax - rMin) / rBinWidth); }
    
    // Calculate charge radius for the particle pair
    double GetChargeRadius() const;
};

// Default configurations
namespace DefaultConfigs {
    // Xi-Pi system configuration
    CalculationConfig GetXiPiConfig();
    CalculationConfig GetPHeConfig();
}
