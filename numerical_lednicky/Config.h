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

// Optional resonant partial wave (L >= 1), added on top of the s-wave Lednicky term.
// The resonant wave replaces the Coulomb partial wave F_L outside the interaction region:
//   psi_L = F_L + T(k) * (G_L + i F_L),   T = x * (Gamma(k)/2) / (M - sqrt(s) - i Gamma(k)/2)
// so the interference with the incident (Coulomb) wave is included, as in arXiv:2607.04351.
struct ResonanceConfig {
    bool enabled = false;
    int L = 1;                    // orbital angular momentum of the resonant wave
    double mass = 0.0;            // MeV, pole mass
    double width = 0.0;           // MeV, total width at the pole
    double elasticity = 1.0;      // x = Gamma_elastic / Gamma_total (x < 1 -> absorption)
    double spinWeight = 1.0;      // (2J+1) / ((2 s1 + 1)(2 s2 + 1)(2L+1)), fraction of the L wave that resonates
    double barrierRadius = 1.0;   // fm, Blatt-Weisskopf radius in the energy-dependent width
    double regulatorRadius = 1.0; // fm, G_L -> G_L * (1 - exp(-r/r_reg))^(2L+1) to tame r -> 0
    bool decayOnly = false;       // keep only |scattered wave|^2 (resonance-decay term, no Coulomb/plane wave)
};

// Delta(1232) contribution to pi-nucleus correlations (Delta -> pi N, N coalesces into the nucleus).
// Computed from pi-p and pi-n P33 waves, mapped onto k*_piA and weighted (see DeltaFeedDown.h).
struct DeltaFeedDownConfig {
    bool enabled = false;
    int massNumber = 0;                // A of the nucleus (0 = not a pi-nucleus system)
    int protonNumber = 0;              // Z of the nucleus
    double mass = 1230.55;             // MeV, Breit-Wigner mass
    double width = 112.2;              // MeV, Breit-Wigner width
    double barrierRadius = 1.0;        // fm, Blatt-Weisskopf radius
    double scatteringRegulator = 1.0;  // fm, small-r regulator for the scattering term
    double decayRegulator = 0.2;       // fm, small-r regulator for the decay term
    double decaySourceSize = 0.42;     // fm, Gaussian R (exp(-r^2/4R^2)) of the decay term
    double sigmaQ = 0.0;               // MeV/c per component, internal nucleon momentum
    int nSamplesQ = 2000;              // Monte Carlo samples for the q smearing
    double weightProton = -1.0;        // < 0: default weights
    double weightNeutron = -1.0;
    double kMaxPiN = 700.0;            // MeV/c, pi-N grid used before mapping
    double kStepPiN = 1.0;             // MeV/c
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
    std::vector<double> effectiveRanges;
    
    // Optional resonance (off by default)
    ResonanceConfig resonance;

    // Optional Delta(1232) contribution for pi-nucleus systems (off by default)
    DeltaFeedDownConfig delta;

    // Particle properties
    ParticleProperties particle1;
    ParticleProperties particle2;
    
    // Output settings
    std::string outputFolder;
    std::string outputRootFolder;
    std::string outputRootFile;

    bool useDatFiles = false;
    std::string intermediateRootFile = "Cky_intermediate.root";
    
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

    // Pion-light-nucleus systems; pionCharge = +1 or -1
    CalculationConfig GetPiDConfig(double pionCharge);
    CalculationConfig GetPiTConfig(double pionCharge);
    CalculationConfig GetPiHe3Config(double pionCharge);
    CalculationConfig GetPiNConfig(double pionCharge, double nucleonCharge); // nucleonCharge 1 = p, 0 = n
}
