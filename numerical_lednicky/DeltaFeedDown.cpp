#include "DeltaFeedDown.h"
#include "CoulombWaveFunction.h"

#include <TFile.h>
#include <TGraph.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
    constexpr double kProtonMass = 938.27209;  // MeV
    constexpr double kNeutronMass = 939.56542; // MeV
    constexpr double kNucleonMass = 938.92;    // MeV, used in the mapping

    double ReducedMass(double m1, double m2) { return m1 * m2 / (m1 + m2); }

    // Linear interpolation on a uniform grid, constant outside
    double Interpolate(const std::vector<double>& x, const std::vector<double>& y, double xv) {
        if (xv <= x.front()) return y.front();
        if (xv >= x.back()) return y.back();
        const double step = x[1] - x[0];
        const size_t i = static_cast<size_t>((xv - x.front()) / step);
        const double t = (xv - x[i]) / step;
        return (1.0 - t) * y[i] + t * y[i + 1];
    }

    // Simpson rule on a uniform grid with an even number of intervals
    double Simpson(const std::vector<double>& f, double h) {
        size_t n = f.size() - 1;
        if (n % 2 == 1) n -= 1;
        double sum = f[0] + f[n];
        for (size_t i = 1; i < n; i++) sum += (i % 2 ? 4.0 : 2.0) * f[i];
        return sum * h / 3.0;
    }
}

DeltaFeedDown::DeltaFeedDown(const CalculationConfig& config) : config_(config) {
    const auto& d = config_.delta;
    if (d.massNumber < 1) throw std::runtime_error("Delta feed-down: not a pi-nucleus system (A not set)");

    const bool piPlus = config_.particle1.charge > 0;
    // name, nucleon mass, nucleon charge, isospin weight of the elastic channel, BR(Delta -> pi N)
    channels_.push_back({"p", kProtonMass, 1.0, piPlus ? 1.0 : 1.0 / 3.0, piPlus ? 1.0 : 1.0 / 3.0});
    channels_.push_back({"n", kNeutronMass, 0.0, piPlus ? 1.0 / 3.0 : 1.0, piPlus ? 1.0 / 3.0 : 1.0});

    for (double k = d.kStepPiN; k <= d.kMaxPiN + 1e-9; k += d.kStepPiN) kPiN_.push_back(k);

    // Fixed-seed q samples, shared by all curves (reproducible, smooth differences between radii)
    std::mt19937 rng(12345);
    std::normal_distribution<double> gaus(0.0, 1.0);
    const int nq = (d.sigmaQ > 0.0) ? d.nSamplesQ : 1;
    for (int i = 0; i < 3 * nq; i++) qSamples_.push_back(d.sigmaQ > 0.0 ? d.sigmaQ * gaus(rng) : 0.0);
}

void DeltaFeedDown::DefaultWeights(bool decay, double& wp, double& wn) const {
    const auto& d = config_.delta;
    double fp = double(d.protonNumber) / d.massNumber;
    double fn = double(d.massNumber - d.protonNumber) / d.massNumber;
    if (decay) {
        fp *= channels_[0].branchingRatio;
        fn *= channels_[1].branchingRatio;
    }
    wp = fp / (fp + fn);
    wn = fn / (fp + fn);
    if (d.weightProton >= 0.0) wp = d.weightProton;
    if (d.weightNeutron >= 0.0) wn = d.weightNeutron;
}

std::vector<std::vector<double>> DeltaFeedDown::PiNCurves(const Channel& ch, const std::vector<double>& sizes,
                                                         bool decay) const {
    const auto& d = config_.delta;

    // pi-N pair: Coulomb radius and the P33 resonance
    CalculationConfig piN;
    piN.particle1 = config_.particle1;
    piN.particle2 = ParticleProperties(ch.nucleonMass, ch.nucleonCharge);
    const double chargeRadius = piN.GetChargeRadius();

    ResonanceConfig res;
    res.enabled = true;
    res.L = 1;
    res.mass = d.mass;
    res.width = d.width;
    res.elasticity = ch.isospinWeight;
    res.spinWeight = 4.0 / 6.0; // J = 3/2 out of J = 1/2, 3/2 for L = 1
    res.barrierRadius = d.barrierRadius;
    res.regulatorRadius = decay ? d.decayRegulator : d.scatteringRegulator;
    res.decayOnly = decay;

    CoulombWaveFunction wf(chargeRadius);
    wf.SetResonance(res, piN.particle1.mass, piN.particle2.mass);

    // One r grid for all source sizes: step fine enough for the smallest, long enough for the largest
    const double smallest = *std::min_element(sizes.begin(), sizes.end());
    const double largest = *std::max_element(sizes.begin(), sizes.end());
    const double h = std::min(0.01, smallest / 40.0);
    const int nr = 2 * static_cast<int>(std::max(10.0 * largest, 5.0) / (2.0 * h)); // even
    const double toFm = 1.0 / std::pow(PhysicalConstants::FM_TO_NU, 3);

    std::vector<std::vector<double>> out(sizes.size(), std::vector<double>(kPiN_.size(), 0.0));
    #pragma omp parallel for schedule(dynamic)
    for (int ik = 0; ik < static_cast<int>(kPiN_.size()); ik++) {
        const double k = kPiN_[ik];
        const double eta = std::isfinite(chargeRadius)
                               ? 1.0 / (k * chargeRadius) / PhysicalConstants::FM_TO_NU : 0.0;
        // ResonanceTerm = 4 pi r^2 FM^3 (2L+1) w_J delta -> divide by FM^3 to integrate in fm
        std::vector<double> term(nr + 1, 0.0);
        for (int ir = 1; ir <= nr; ir++) term[ir] = wf.ResonanceTerm(k, ir * h, eta) * toFm;

        std::vector<double> f(nr + 1, 0.0);
        for (size_t is = 0; is < sizes.size(); is++) {
            const double R = sizes[is];
            const double norm = 1.0 / std::pow(4.0 * PhysicalConstants::PI * R * R, 1.5); // fm^-3
            for (int ir = 1; ir <= nr; ir++) {
                const double r = ir * h;
                f[ir] = norm * std::exp(-r * r / (4.0 * R * R)) * term[ir];
            }
            out[is][ik] = Simpson(f, h);
        }
    }
    return out;
}

std::vector<double> DeltaFeedDown::MapToNucleus(const std::vector<double>& piN) const {
    const double mPi = config_.particle1.mass;
    const double muPiN = ReducedMass(mPi, kNucleonMass);
    const double muPiA = ReducedMass(mPi, config_.delta.massNumber * kNucleonMass);
    const int nq = static_cast<int>(qSamples_.size() / 3);

    const int nk = config_.GetNumKBins();
    std::vector<double> out(nk, 0.0);
    for (int ik = 0; ik < nk; ik++) {
        const double kA = config_.kMin + ik * config_.kBinWidth;
        double sum = 0.0;
        for (int iq = 0; iq < nq; iq++) {
            // v = k_A / mu_A - q / m_N, with k_A along z
            const double vx = -qSamples_[3 * iq] / kNucleonMass;
            const double vy = -qSamples_[3 * iq + 1] / kNucleonMass;
            const double vz = kA / muPiA - qSamples_[3 * iq + 2] / kNucleonMass;
            sum += Interpolate(kPiN_, piN, muPiN * std::sqrt(vx * vx + vy * vy + vz * vz));
        }
        out[ik] = sum / nq;
    }
    return out;
}

std::string DeltaFeedDown::OutputFileName() const {
    std::string stem = config_.outputRootFile;
    const auto dot = stem.rfind(".root");
    if (dot != std::string::npos) stem = stem.substr(0, dot);
    char tag[96];
    std::snprintf(tag, sizeof(tag), "_Delta_rsca%.2f_rdec%.2f_sq%.0f.root", config_.delta.scatteringRegulator,
                  config_.delta.decayRegulator, config_.delta.sigmaQ);
    return config_.outputRootFolder + stem + tag;
}

void DeltaFeedDown::GenerateRootFile() const {
    const auto& d = config_.delta;
    double wpSca, wnSca, wpDec, wnDec;
    DefaultWeights(false, wpSca, wnSca);
    DefaultWeights(true, wpDec, wnDec);

    std::cout << "Delta(1232) contribution: A=" << d.massNumber << " Z=" << d.protonNumber
              << ", M=" << d.mass << " MeV, Gamma=" << d.width << " MeV" << std::endl;
    std::cout << "  scattering weights p/n = " << wpSca << "/" << wnSca
              << ", decay weights p/n = " << wpDec << "/" << wnDec << std::endl;
    std::cout << "  r_reg (sca/dec) = " << d.scatteringRegulator << "/" << d.decayRegulator
              << " fm, decay source R = " << d.decaySourceSize << " fm, sigma_q = " << d.sigmaQ << " MeV/c" << std::endl;

    const int nk = config_.GetNumKBins();
    std::vector<double> kA(nk);
    for (int ik = 0; ik < nk; ik++) kA[ik] = config_.kMin + ik * config_.kBinWidth;

    TFile fout(OutputFileName().c_str(), "RECREATE");
    if (fout.IsZombie()) throw std::runtime_error("Delta feed-down: cannot open " + OutputFileName());

    auto write = [&](const std::vector<double>& x, const std::vector<double>& y,
                     const std::string& name, const std::string& title) {
        TGraph g(static_cast<int>(x.size()), x.data(), y.data());
        g.SetName(name.c_str());
        g.SetTitle(title.c_str());
        g.Write();
    };

    char name[128], title[256];

    // Decay term: one curve (fixed small source)
    {
        std::vector<double> total(nk, 0.0);
        for (size_t ic = 0; ic < channels_.size(); ic++) {
            const auto& ch = channels_[ic];
            auto piN = PiNCurves(ch, {d.decaySourceSize}, true)[0];
            auto mapped = MapToNucleus(piN);
            const double w = (ic == 0) ? wpDec : wnDec;
            for (int ik = 0; ik < nk; ik++) total[ik] += w * mapped[ik];
            std::snprintf(name, sizeof(name), "gPiN_%s_DeltaDec", ch.name.c_str());
            std::snprintf(title, sizeof(title), "pi-%s Delta decay term (R=%.2f fm);k*_{#piN} (MeV/c);C_{dec}",
                          ch.name.c_str(), d.decaySourceSize);
            write(kPiN_, piN, name, title);
        }
        std::snprintf(title, sizeof(title),
                      "Delta decay term mapped to pi-A (R=%.2f fm, w_p=%.3f, w_n=%.3f);k* (MeV/c);#DeltaC",
                      d.decaySourceSize, wpDec, wnDec);
        write(kA, total, "gDeltaDec", title);
    }

    // Scattering term: one curve per source size
    std::vector<std::vector<std::vector<double>>> piNSca; // [channel][size][k]
    for (const auto& ch : channels_) piNSca.push_back(PiNCurves(ch, config_.sourceSizes, false));

    for (size_t is = 0; is < config_.sourceSizes.size(); is++) {
        const double R = config_.sourceSizes[is];
        std::vector<double> total(nk, 0.0);
        for (size_t ic = 0; ic < channels_.size(); ic++) {
            const auto& ch = channels_[ic];
            const auto& piN = piNSca[ic][is];
            auto mapped = MapToNucleus(piN);
            const double w = (ic == 0) ? wpSca : wnSca;
            for (int ik = 0; ik < nk; ik++) total[ik] += w * mapped[ik];
            std::snprintf(name, sizeof(name), "gPiN_%s_DeltaSca_R%.2f", ch.name.c_str(), R);
            std::snprintf(title, sizeof(title), "pi-%s Delta scattering - Coulomb (R=%.2f fm);k*_{#piN} (MeV/c);#DeltaC",
                          ch.name.c_str(), R);
            write(kPiN_, piN, name, title);
        }
        std::snprintf(name, sizeof(name), "gDeltaSca_R%.2f", R);
        std::snprintf(title, sizeof(title),
                      "Delta scattering mapped to pi-A (R=%.2f fm, w_p=%.3f, w_n=%.3f);k* (MeV/c);#DeltaC",
                      R, wpSca, wnSca);
        write(kA, total, name, title);
        std::cout << "  written " << name << std::endl;
    }

    fout.Close();
    std::cout << "Delta ROOT file: " << OutputFileName() << std::endl;
}
