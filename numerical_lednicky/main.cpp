#include "Config.h"
#include "CorrelationCalculator.h"

#include "acb.h"
#include "acb_hypgeom.h"
#include "arb.h"
#include "flint.h"

#include <TROOT.h>

#include <cstdlib>
#include <iostream>
#include <string>

void PrintUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --mode <generate|root|both>    Operation mode (default: both)" << std::endl;
    std::cout << "  --output-folder <path>         Output folder for data files" << std::endl;
    std::cout << "  --output-root <filename>       Output ROOT file name" << std::endl;
    std::cout << "  --particle1-mass <value>       Mass of particle 1 (MeV)" << std::endl;
    std::cout << "  --particle1-charge <value>     Charge of particle 1" << std::endl;
    std::cout << "  --particle2-mass <value>       Mass of particle 2 (MeV)" << std::endl;
    std::cout << "  --particle2-charge <value>     Charge of particle 2" << std::endl;
    std::cout << "  --dat                          Use .dat files for intermediate storage" << std::endl;
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
    CalculationConfig config = DefaultConfigs::GetPHeConfig();
    
    std::string mode = "both"; // generate, root, or both
    
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
        } else if (arg == "--dat") {
            config.useDatFiles = true;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            PrintUsage(argv[0]);
            return 1;
        }
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
