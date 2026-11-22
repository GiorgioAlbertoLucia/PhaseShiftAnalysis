#include "Config.h"
#include <cmath>
#include <limits>

double CalculationConfig::GetChargeRadius() const {
    double reducedMass = (particle1.mass * particle2.mass) / (particle1.mass + particle2.mass);
    double chargeProduct = particle1.charge * particle2.charge;
    if (chargeProduct == 0.0) return std::numeric_limits<double>::infinity(); // neutral: eta = 0
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
        //config.sourceSizes = {5.34, 5.34 - 0.11, 5.34 + 0.10,
        //                     4.24, 4.24 - 0.06, 4.24 + 0.06,};
        // Values from fit (free radius)
        //config.sourceSizes = {6.93, 6.93 - 0.88, 6.93 + 0.88,
        //                     6.35, 6.35 - 0.72, 6.35 + 0.72,};
        // Values from fit (free radius, with 26 data)
        config.sourceSizes = {6.62, 6.62 - 0.79, 6.62 + 0.79,
                             5.10, 5.10 - 0.54, 5.10 + 0.54,};
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
        //config.realScatteringLengths = {-11.26, -11.30, -11.22, -11.26, -11.26, -9.06, -9.10, -9.02, -9.06, -9.06}; // fm
        //config.imagScatteringLengths = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // fm
        //config.effectiveRanges = {1.65, 1.65, 1.65, 1.36, 1.94, 1.36, 1.36, 1.36, 1.11, 1.61};   // fm

        // all combinations of {n, n-e, n+e} for a and r, separately for each spin state (nominal first)
        const std::vector<std::vector<double>> aReVariations = {{-11.26, -11.30, -11.22},  // 1S0, fm
                                                                {-9.06, -9.10, -9.02}};    // 3S1, fm
        const std::vector<std::vector<double>> rVariations = {{1.65, 1.36, 1.94},          // 1S0, fm
                                                              {1.36, 1.11, 1.61}};         // 3S1, fm
        config.realScatteringLengths = {};
        config.imagScatteringLengths = {};
        config.effectiveRanges = {};
        for (size_t istate = 0; istate < aReVariations.size(); istate++) {
            for (double aRe : aReVariations[istate]) {
                for (double r : rVariations[istate]) {
                    config.realScatteringLengths.push_back(aRe);
                    config.imagScatteringLengths.push_back(0.0);
                    config.effectiveRanges.push_back(r);
                }
            }
        }

        // without variations
        //config.realScatteringLengths = {-11.26, -9.06}; // fm
        //config.imagScatteringLengths = {0.0, 0.0}; // fm
        //config.effectiveRanges = {1.65, 1.36};   // fm
        
        // Output settings
        config.outputFolder = "/home/galucia/PhaseShiftAnalysis/numerical_lednicky/pHe/dat/";
        config.outputRootFolder = "/home/galucia/PhaseShiftAnalysis/numerical_lednicky/pHe/output/";
        config.outputRootFile = "TheoCF_PHe.root";
        
        return config;
    }

    // ------------------------------------------------------------------
    // Pion - light nucleus systems
    // SIGN CONVENTION: f0 > 0 attractive, as in the pHe config
    //   ( f_c = 1 / (1/f0 + 0.5*d0*k^2 - 2/ac*H(eta) - i*k*A_c(eta)) )
    // Pionic-atom scattering lengths use the same sign (a < 0 = repulsive).
    // Conversion: 1/m_pi = hbar*c / 139.570 MeV = 1.4138 fm
    // Effective ranges are unknown -> zero-range approximation (d0 = 0).
    // Entry 0 of every scattering-length list is f0 = 0 (Coulomb only).
    // ------------------------------------------------------------------
    namespace {
        constexpr double kPionMass     = 139.57039;  // MeV
        constexpr double kDeuteronMass = 1875.61294; // MeV
        constexpr double kTritonMass   = 2808.92113; // MeV
        constexpr double kHelion3Mass  = 2808.39161; // MeV

        // Source radii scan: 0.50-1.20 fm in 0.05 steps, then 1.3-4.0 fm in 0.1 steps
        std::vector<double> PionNucleusSourceScan() {
            std::vector<double> sizes;
            for (int i = 0; i <= 14; i++) sizes.push_back(0.50 + 0.05 * i); // 0.50 ... 1.20
            for (int i = 1; i <= 28; i++) sizes.push_back(1.20 + 0.10 * i); // 1.30 ... 4.00
            for (auto& size : sizes) size = std::round(size * 100.) / 100.;  // remove float noise
            // The code's Gaussian is exp(-r^2 / (4 R^2)). If your radii refer to
            // exp(-r^2 / (2 R^2)), uncomment the conversion (as in GetPHeConfig):
            //for (auto& size : sizes) size = size / std::sqrt(2.);
            return sizes;
        }

        CalculationConfig PionNucleusBase(double pionCharge, double nucleusMass, double nucleusCharge,
                                          const std::string& tag) {
            CalculationConfig config;
            config.kMin = 1.0;
            config.kMax = 551.0;
            config.kBinWidth = 1.0;

            config.particle1 = ParticleProperties(kPionMass, pionCharge);
            config.particle2 = ParticleProperties(nucleusMass, nucleusCharge);
            config.sourceSizes = PionNucleusSourceScan();

            const std::string pion = (pionCharge > 0) ? "Pip" : "Pim";
            config.outputFolder     = "./" + tag + "/dat_" + pion + "/";
            config.outputRootFolder = "./" + tag + "/output/";
            config.outputRootFile   = "TheoCF_" + pion + tag + ".root";
            return config;
        }
    }

    CalculationConfig GetPiDConfig(double pionCharge) {
        CalculationConfig config = PionNucleusBase(pionCharge, kDeuteronMass, 1.0, "D");
        config.delta.massNumber = 2;   // Delta(1232) term, enabled with --resonance
        config.delta.protonNumber = 1;
        config.delta.sigmaQ = 60.0;    // MeV/c, illustrative internal momentum

        // [0] Coulomb only (ALICE treats pi+ d this way, PRC 112 (2025) 064003)
        // [1] PSI pionic deuterium, Strauch et al. EPJA 47 (2011) 88:
        //     eps_1s = -2356(31) meV, Gamma_1s = 1171(+23,-49) meV
        //     -> a = -0.0250(3) + i 0.0062(1) m_pi^-1 = -0.0353(5) + i 0.0088(2) fm
        // [2,3] Re f0 -/+ 1 sigma
        // [4] value used by ALICE pi-d femtoscopy: -0.0382(7) + i 0.0092(10) fm
        // Strong part is isospin symmetric (isoscalar target): same values for pi+ d and pi- d.
        config.realScatteringLengths = {0.0, -0.0353, -0.0358, -0.0348, -0.0382}; // fm
        config.imagScatteringLengths = {0.0,  0.0088,  0.0088,  0.0088,  0.0092}; // fm
        config.effectiveRanges       = {0.0,  0.0,     0.0,     0.0,     0.0};    // fm
        return config;
    }

    // pi-N with the Delta(1232) in P33. The pi-nucleus Delta peak comes from Delta -> pi N with the
    // nucleon then coalescing into the nucleus: compute pi-N here, then map onto the pi-A relative
    // momentum with macros/map_piN_to_piA.py.
    //   --resonance                   : Coulomb + Delta scattering (incident/scattered interference)
    //   --resonance --res-decay-only  : Delta decay term, |scattered wave|^2 only (use a small source)
    // Breit-Wigner M = 1230.55 MeV, Gamma = 112.2 MeV (values quoted in arXiv:2607.04351).
    // "elasticity" carries the I = 3/2 isospin weight of the elastic channel:
    //   pi+ p, pi- n : 1     pi- p, pi+ n : 1/3
    CalculationConfig GetPiNConfig(double pionCharge, double nucleonCharge) {
        const double nucleonMass = (nucleonCharge != 0.0) ? 938.27209 : 939.56542; // MeV
        const std::string tag = (nucleonCharge != 0.0) ? "P" : "N";
        CalculationConfig config = PionNucleusBase(pionCharge, nucleonMass, nucleonCharge, tag);

        config.realScatteringLengths = {0.0}; // s wave left out (enable here if needed), fm
        config.imagScatteringLengths = {0.0};
        config.effectiveRanges       = {0.0};

        const bool stretched = (pionCharge > 0) == (nucleonCharge != 0.0); // pi+ p or pi- n
        config.resonance.L = 1;
        config.resonance.mass = 1230.55;
        config.resonance.width = 112.2;
        config.resonance.elasticity = stretched ? 1.0 : 1.0 / 3.0;
        config.resonance.spinWeight = 4.0 / 6.0; // J = 3/2 out of J = 1/2, 3/2 for L = 1
        config.resonance.regulatorRadius = 1.0;  // fm, vary 0.2-1.0: moves the scattering peak
        return config;
    }

    CalculationConfig GetPiTConfig(double pionCharge) {
        CalculationConfig config = PionNucleusBase(pionCharge, kTritonMass, 1.0, "T");
        config.delta.massNumber = 3;   // Delta(1232) term, enabled with --resonance
        config.delta.protonNumber = 1;
        config.delta.sigmaQ = 90.0;    // MeV/c, illustrative internal momentum

        // No measurement exists. Theory: arXiv:2606.23331, Table 9.1 (10^-3 m_pi^-1):
        //   pi- 3H = -127.4(55) -> -0.1801(78) fm   (charge mirror of pi+ 3He)
        //   pi+ 3H =   65.0(55) -> +0.0919(78) fm   (charge mirror of pi- 3He)
        // [0] Coulomb only, [1] nominal, [2,3] -/+ 1 sigma. Imaginary part not available.
        if (pionCharge < 0) {
            config.realScatteringLengths = {0.0, -0.1801, -0.1879, -0.1723}; // fm
        } else {
            config.realScatteringLengths = {0.0,  0.0919,  0.0841,  0.0997}; // fm
        }
        config.imagScatteringLengths = {0.0, 0.0, 0.0, 0.0}; // fm
        config.effectiveRanges       = {0.0, 0.0, 0.0, 0.0}; // fm
        return config;
    }

    CalculationConfig GetPiHe3Config(double pionCharge) {
        CalculationConfig config = PionNucleusBase(pionCharge, kHelion3Mass, 2.0, "He3");
        config.delta.massNumber = 3;   // Delta(1232) term, enabled with --resonance
        config.delta.protonNumber = 2;
        config.delta.sigmaQ = 90.0;    // MeV/c, illustrative internal momentum

        // Theory: arXiv:2606.23331, Table 9.1 (10^-3 m_pi^-1):
        //   pi- 3He =   64.9(55) -> +0.0918(78) fm
        //   pi+ 3He = -127.5(55) -> -0.1803(78) fm
        // pi- 3He pionic atoms (PSI): Schwanner 41(4), Mason 43(5), Abela 56(6) (10^-3 m_pi^-1)
        //   -> [4] Schwanner +0.0580 fm, [5] Abela +0.0792 fm
        // [0] Coulomb only, [1] nominal theory, [2,3] -/+ 1 sigma. Imaginary part not available.
        if (pionCharge < 0) {
            config.realScatteringLengths = {0.0, 0.0918, 0.0840, 0.0996, 0.0580, 0.0792}; // fm
            config.imagScatteringLengths = {0.0, 0.0,    0.0,    0.0,    0.0,    0.0};    // fm
            config.effectiveRanges       = {0.0, 0.0,    0.0,    0.0,    0.0,    0.0};    // fm
        } else {
            config.realScatteringLengths = {0.0, -0.1803, -0.1881, -0.1725}; // fm
            config.imagScatteringLengths = {0.0, 0.0, 0.0, 0.0};             // fm
            config.effectiveRanges       = {0.0, 0.0, 0.0, 0.0};             // fm
        }
        return config;
    }
}
