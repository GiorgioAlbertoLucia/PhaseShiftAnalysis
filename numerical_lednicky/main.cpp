#include "Config.h"
#include "CorrelationCalculator.h"
#include "DeltaFeedDown.h"

#include "acb.h"
#include "acb_hypgeom.h"
#include "arb.h"
#include "flint.h"

#include <TROOT.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

void PrintUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --mode <generate|root|both|delta>  Operation mode (default: both; delta = Delta(1232) term only)" << std::endl;
    std::cout << "  --output-folder <path>         Output folder for data files" << std::endl;
    std::cout << "  --output-root-folder <path>    Folder for the final and Delta ROOT files" << std::endl;
    std::cout << "  --output-root <filename>       Output ROOT file name" << std::endl;
    std::cout << "  --particle1-mass <value>       Mass of particle 1 (MeV)" << std::endl;
    std::cout << "  --particle1-charge <value>     Charge of particle 1" << std::endl;
    std::cout << "  --particle2-mass <value>       Mass of particle 2 (MeV)" << std::endl;
    std::cout << "  --particle2-charge <value>     Charge of particle 2" << std::endl;
    std::cout << "  --dat                          Use .dat files for intermediate storage" << std::endl;
    std::cout << "  --system <XiPi|PHe|PiD|PiT|PiHe3|PiN>  Built-in configuration (default: PHe)" << std::endl;
    std::cout << "  --nucleon-charge <1|0>         Nucleon for PiN: 1 = proton, 0 = neutron (default: 1)" << std::endl;
    std::cout << "  --res-decay-only               Resonance-decay term only (|scattered wave|^2)" << std::endl;
    std::cout << "  --pion-charge <+1|-1>          Pion charge for PiD/PiT/PiHe3 (default: -1)" << std::endl;
    std::cout << "  --resonance                    PiN: add the Delta P33 wave to the pair." << std::endl;
    std::cout << "                                 PiD/PiT/PiHe3: also write the Delta(1232) term (<root>_Delta_*.root)" << std::endl;
    std::cout << "  --res-elasticity <x>           PiN: override Gamma_el/Gamma_tot (isospin weight) of the resonance" << std::endl;
    std::cout << "  --res-regulator <fm>           Small-r regulator of the resonant wave (scattering term)" << std::endl;
    std::cout << "  --res-decay-regulator <fm>     PiD/PiT/PiHe3: small-r regulator of the decay term (default 0.2)" << std::endl;
    std::cout << "  --delta-decay-radius <fm>      PiD/PiT/PiHe3: Gaussian R of the decay term (default 0.42)" << std::endl;
    std::cout << "  --sigma-q <MeV/c>              PiD/PiT/PiHe3: internal nucleon momentum per component" << std::endl;
    std::cout << "  --w-proton <w>, --w-neutron <w>  PiD/PiT/PiHe3: override the p/n channel weights" << std::endl;
    std::cout << "  --help                         Show this help message" << std::endl;
    std::cout << "\nExample:" << std::endl;
    std::cout << "  " << programName << " --mode both --output-folder ./output/" << std::endl;
}

int main(int argc, char* argv[]) {

    ROOT::EnableThreadSafety();

    std::cout << "======================================" << std::endl;
    std::cout << "Correlation Function Calculator" << std::endl;
    std::cout << "======================================" << std::endl;
    
    //CalculationConfig config = DefaultConfigs::GetXiPiConfig();
    // Pick the built-in configuration first, so the other options can override it
    std::string system = "PHe";
    double pionCharge = -1.0;
    double nucleonCharge = 1.0;
    for (int i = 1; i < argc - 1; i++) {
        std::string arg = argv[i];
        if (arg == "--system") system = argv[i + 1];
        else if (arg == "--pion-charge") pionCharge = atof(argv[i + 1]);
        else if (arg == "--nucleon-charge") nucleonCharge = atof(argv[i + 1]);
    }

    CalculationConfig config;
    if      (system == "XiPi")  config = DefaultConfigs::GetXiPiConfig();
    else if (system == "PHe")   config = DefaultConfigs::GetPHeConfig();
    else if (system == "PiD")   config = DefaultConfigs::GetPiDConfig(pionCharge);
    else if (system == "PiT")   config = DefaultConfigs::GetPiTConfig(pionCharge);
    else if (system == "PiHe3") config = DefaultConfigs::GetPiHe3Config(pionCharge);
    else if (system == "PiN")   config = DefaultConfigs::GetPiNConfig(pionCharge, nucleonCharge);
    else {
        std::cerr << "Unknown system: " << system << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }
    
    std::string mode = "both"; // generate, root, both, or delta
    const bool piNucleus = config.delta.massNumber > 0;
    bool wantResonance = false;
    double regulator = -1.0;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return 0;
        } else if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--output-folder" && i + 1 < argc) {
            config.outputFolder = argv[++i];
        } else if (arg == "--output-root-folder" && i + 1 < argc) {
            config.outputRootFolder = argv[++i];
            if (config.outputRootFolder.back() != '/') config.outputRootFolder += '/';
        } else if (arg == "--output-root" && i + 1 < argc) {
            config.outputRootFile = argv[++i];
        } else if (arg == "--particle1-mass" && i + 1 < argc) {
            config.particle1.mass = atof(argv[++i]);
        } else if (arg == "--particle1-charge" && i + 1 < argc) {
            config.particle1.charge = atof(argv[++i]);
        } else if (arg == "--particle2-mass" && i + 1 < argc) {
            config.particle2.mass = atof(argv[++i]);
        } else if (arg == "--particle2-charge" && i + 1 < argc) {
            config.particle2.charge = atof(argv[++i]);
        } else if ((arg == "--system" || arg == "--pion-charge" || arg == "--nucleon-charge") && i + 1 < argc) {
            ++i; // already handled above
        } else if (arg == "--resonance") {
            wantResonance = true;
        } else if (arg == "--res-elasticity" && i + 1 < argc) {
            config.resonance.elasticity = atof(argv[++i]);
        } else if (arg == "--res-decay-only") {
            config.resonance.decayOnly = true;
        } else if (arg == "--res-regulator" && i + 1 < argc) {
            regulator = atof(argv[++i]);
        } else if (arg == "--res-decay-regulator" && i + 1 < argc) {
            config.delta.decayRegulator = atof(argv[++i]);
        } else if (arg == "--delta-decay-radius" && i + 1 < argc) {
            config.delta.decaySourceSize = atof(argv[++i]);
        } else if (arg == "--sigma-q" && i + 1 < argc) {
            config.delta.sigmaQ = atof(argv[++i]);
        } else if (arg == "--w-proton" && i + 1 < argc) {
            config.delta.weightProton = atof(argv[++i]);
        } else if (arg == "--w-neutron" && i + 1 < argc) {
            config.delta.weightNeutron = atof(argv[++i]);
        } else if (arg == "--dat") {
            config.useDatFiles = true;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            PrintUsage(argv[0]);
            return 1;
        }
    }
    
    if (piNucleus) {
        // pi-nucleus: the Delta is a pi-N resonance -> separate Delta(1232) term, the pi-A wave is unchanged
        config.delta.enabled = wantResonance || mode == "delta";
        if (regulator > 0.0) config.delta.scatteringRegulator = regulator;
        if (config.resonance.decayOnly)
            std::cout << "Note: --res-decay-only is for PiN; the Delta file always contains both terms" << std::endl;
    } else {
        config.resonance.enabled = wantResonance;
        if (regulator > 0.0) config.resonance.regulatorRadius = regulator;
        if (mode == "delta") {
            std::cerr << "--mode delta is only available for PiD, PiT and PiHe3" << std::endl;
            return 1;
        }
    }

    if (config.resonance.enabled) {
        if (config.resonance.mass <= 0.0) {
            std::cerr << "--resonance: no resonance defined for this system" << std::endl;
            return 1;
        }
        // keep resonance runs separate from the s-wave-only ones
        char tag[64];
        snprintf(tag, sizeof(tag), "_%s_x%.2f_rreg%.2f", config.resonance.decayOnly ? "ResDecay" : "Res",
                 config.resonance.elasticity, config.resonance.regulatorRadius);
        auto dot = config.outputRootFile.rfind(".root");
        config.outputRootFile.insert(dot == std::string::npos ? config.outputRootFile.size() : dot, tag);
        config.intermediateRootFile = std::string("Cky_intermediate") + tag + ".root";
        std::cout << "  Resonance: M=" << config.resonance.mass << " MeV, Gamma=" << config.resonance.width
                  << " MeV, L=" << config.resonance.L << ", x=" << config.resonance.elasticity
                  << ", r_reg=" << config.resonance.regulatorRadius << " fm" << std::endl;
    }

    // Print configuration
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Mode: " << mode << std::endl;
    std::cout << "  Output folder: " << config.outputFolder << std::endl;
    std::cout << "  Output ROOT file: " << config.outputRootFile << std::endl;
    std::cout << "  Particle 1: mass=" << config.particle1.mass 
         << " MeV, charge=" << config.particle1.charge << std::endl;
    std::cout << "  Particle 2: mass=" << config.particle2.mass 
         << " MeV, charge=" << config.particle2.charge << std::endl;
    std::cout << "  Charge radius: " << config.GetChargeRadius() << " fm" << std::endl;
    std::cout << "  k range: [" << config.kMin << ", " << config.kMax 
         << "] MeV/c in steps of " << config.kBinWidth << std::endl;
    std::cout << "  r range: [" << config.rMin << ", " << config.rMax 
         << "] fm in steps of " << config.rBinWidth << std::endl;
    std::cout << std::endl;
    
    // Create the output folders (relative paths are relative to where the program is run)
    for (const std::string& folder : {config.outputFolder, config.outputRootFolder}) {
        if (folder.empty()) continue;
        std::error_code ec;
        std::filesystem::create_directories(folder, ec);
        if (ec) {
            std::cerr << "Error: cannot create output folder " << folder << ": " << ec.message() << std::endl;
            return 1;
        }
    }
    std::cout << "Output folders: " << std::filesystem::absolute(config.outputFolder).string() << ", "
              << std::filesystem::absolute(config.outputRootFolder).string() << std::endl;

    try {
        CorrelationCalculator calculator(config);
        
        if (mode == "generate" || mode == "both") {
            std::cout << "Generating data files..." << std::endl;
            calculator.GenerateDataFiles();
        }
        
        if (mode == "root" || mode == "both") {
            std::cout << "\nGenerating ROOT file..." << std::endl;
            calculator.GenerateRootFile();
        }

        if (config.delta.enabled) {
            std::cout << "\nGenerating Delta(1232) term..." << std::endl;
            DeltaFeedDown delta(config);
            delta.GenerateRootFile();
        }
        
        std::cout << "\n======================================" << std::endl;
        std::cout << "Calculation complete!" << std::endl;
        std::cout << "======================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    flint_cleanup(); // Clean up FLINT resources before exiting
    return 0;
}
