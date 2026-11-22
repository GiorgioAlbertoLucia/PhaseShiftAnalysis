#pragma once

#include "Config.h"

#include <string>
#include <vector>

// Delta(1232) contribution to pi-nucleus correlation functions.
//
// Picture: Delta -> pi N, and the nucleon then coalesces into the nucleus (A, Z).
// Two pi-N channels contribute to every system:
//     pi+ :  Delta++ -> pi+ p  (BR 1)     Delta+ -> pi+ n  (BR 1/3)
//     pi- :  Delta0  -> pi- p  (BR 1/3)   Delta- -> pi- n  (BR 1)
//
// For each channel, with the P33 wave psi_1 = F_1 + T (G_1 + i F_1):
//   scattering:  dC_sca(k) = int S_R(r)    (2L+1) w_J (|psi_1|^2 - F_1^2) / rho^2   (Coulomb cancels exactly)
//   decay:       dC_dec(k) = int S_Rdec(r) (2L+1) w_J |T (G_1 + i F_1)|^2 / rho^2
// The pi-N curves are mapped onto k*_piA with
//   k*_piN / mu_piN = k*_piA / mu_piA - q / m_N,   q ~ isotropic Gaussian (sigma per component)
// and summed with weights (normalised to 1):
//   decay      : w_p = BR_p Z/A,  w_n = BR_n N/A  (equal production of the Delta charge states)
//   scattering : w_p = Z/A,       w_n = N/A       (isospin factor already in the pi-N amplitude)
// The amplitudes (lambda, P) are fit parameters: add the output to C(k) with fitted weights.
class DeltaFeedDown {
public:
    explicit DeltaFeedDown(const CalculationConfig& config);

    // Writes <outputRootFolder>/<outputRootFile stem>_Delta.root
    void GenerateRootFile() const;

private:
    struct Channel {
        std::string name;      // "p" or "n"
        double nucleonMass;    // MeV
        double nucleonCharge;  // e
        double isospinWeight;  // elasticity of the I = 3/2 amplitude in this charge channel
        double branchingRatio; // BR(Delta -> pi N) for this charge channel
    };

    CalculationConfig config_;
    std::vector<Channel> channels_;
    std::vector<double> kPiN_;                // pi-N momentum grid
    std::vector<double> qSamples_;            // 3 * nSamplesQ components

    // pi-N curves on kPiN_ for one channel, one per Gaussian source size
    std::vector<std::vector<double>> PiNCurves(const Channel& ch, const std::vector<double>& sizes, bool decay) const;
    // pi-N curve -> pi-A curve on the config k grid
    std::vector<double> MapToNucleus(const std::vector<double>& piN) const;
    void DefaultWeights(bool decay, double& wp, double& wn) const;
    std::string OutputFileName() const;
};
