#include <iostream>
#include <string>
#include <cstdlib>
#include "Config.h"
#include "CorrelationCalculator.h"

using namespace std;

void PrintUsage(const char* programName) {
    cout << "Usage: " << programName << " [options]" << endl;
    cout << "\nOptions:" << endl;
    cout << "  --mode <generate|root|both>    Operation mode (default: both)" << endl;
    cout << "  --output-folder <path>         Output folder for data files" << endl;
    cout << "  --output-root <filename>       Output ROOT file name" << endl;
    cout << "  --particle1-mass <value>       Mass of particle 1 (MeV)" << endl;
    cout << "  --particle1-charge <value>     Charge of particle 1" << endl;
    cout << "  --particle2-mass <value>       Mass of particle 2 (MeV)" << endl;
    cout << "  --particle2-charge <value>     Charge of particle 2" << endl;
    cout << "  --help                         Show this help message" << endl;
    cout << "\nExample:" << endl;
    cout << "  " << programName << " --mode both --output-folder ./output/" << endl;
}

int main(int argc, char* argv[]) {
    cout << "======================================" << endl;
    cout << "Correlation Function Calculator" << endl;
    cout << "======================================" << endl;
    
    //CalculationConfig config = DefaultConfigs::GetXiPiConfig();
    CalculationConfig config = DefaultConfigs::GetPHeConfig();
    
    string mode = "both"; // generate, root, or both
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return 0;
        }
        else if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        }
        else if (arg == "--output-folder" && i + 1 < argc) {
            config.outputFolder = argv[++i];
        }
        else if (arg == "--output-root" && i + 1 < argc) {
            config.outputRootFile = argv[++i];
        }
        else if (arg == "--particle1-mass" && i + 1 < argc) {
            config.particle1.mass = atof(argv[++i]);
        }
        else if (arg == "--particle1-charge" && i + 1 < argc) {
            config.particle1.charge = atof(argv[++i]);
        }
        else if (arg == "--particle2-mass" && i + 1 < argc) {
            config.particle2.mass = atof(argv[++i]);
        }
        else if (arg == "--particle2-charge" && i + 1 < argc) {
            config.particle2.charge = atof(argv[++i]);
        }
        else {
            cerr << "Unknown argument: " << arg << endl;
            PrintUsage(argv[0]);
            return 1;
        }
    }
    
    // Print configuration
    cout << "\nConfiguration:" << endl;
    cout << "  Mode: " << mode << endl;
    cout << "  Output folder: " << config.outputFolder << endl;
    cout << "  Output ROOT file: " << config.outputRootFile << endl;
    cout << "  Particle 1: mass=" << config.particle1.mass 
         << " MeV, charge=" << config.particle1.charge << endl;
    cout << "  Particle 2: mass=" << config.particle2.mass 
         << " MeV, charge=" << config.particle2.charge << endl;
    cout << "  Charge radius: " << config.GetChargeRadius() << " fm" << endl;
    cout << "  k range: [" << config.kMin << ", " << config.kMax 
         << "] MeV/c in steps of " << config.kBinWidth << endl;
    cout << "  r range: [" << config.rMin << ", " << config.rMax 
         << "] fm in steps of " << config.rBinWidth << endl;
    cout << endl;
    
    try {
        CorrelationCalculator calculator(config);
        
        if (mode == "generate" || mode == "both") {
            cout << "Generating data files..." << endl;
            calculator.GenerateDataFiles();
        }
        
        if (mode == "root" || mode == "both") {
            cout << "\nGenerating ROOT file..." << endl;
            calculator.GenerateRootFile();
        }
        
        cout << "\n======================================" << endl;
        cout << "Calculation complete!" << endl;
        cout << "======================================" << endl;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
