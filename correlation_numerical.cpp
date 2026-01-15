#include <TH1F.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TLegend.h>

#include "include/Potentials.h"
#include "include/NumericalCoulombStrongWavefunction.h"

void compare_potentials() {
    std::cout << "Computing correlation functions with different potentials..." << std::endl;
    
    TFile* outfile = TFile::Open("output/potential_comparison.root", "RECREATE");
    
    // Define k grid
    int nbins = 200;
    double kmin = 1.0, kmax = 201.0;
    std::vector<double> k_values(nbins);
    for (int i = 0; i < nbins; ++i) {
        k_values[i] = kmin + i * (kmax - kmin) / (nbins - 1) + 0.5;
    }
    
    double R = 2.0; // fm
    
    // Setup different potentials
    std::vector<std::pair<std::string, std::shared_ptr<Potential>>> potentials;
    
    // 1. Double Gaussian (attractive core + repulsive barrier)
    potentials.push_back({
        "DoubleGaussian",
        std::make_shared<DoubleGausPotential>(50.0, 0.5, -30.0, 1.5)
    });
    
    // 2. Double Square Well
    potentials.push_back({
        "DoubleSquareWell",
        std::make_shared<DoubleSquareWellPotential>(-2.3806, 1.796947, -114.5162, 2.049168)
    });
    
    
    // 3. Woods-Saxon
    potentials.push_back({
        "WoodsSaxon",
        std::make_shared<WoodsSaxonPotential>(-50.0, 1.2, 0.5)
    });
    
    // Compute for each potential
    for (const auto& [name, pot] : potentials) {
        std::cout << "\nProcessing potential: " << name << std::endl;
        
        // Use same potential for singlet and triplet for simplicity
        // (in reality, you'd use different parameters)
        NumericalCoulombStrongWavefunction wf(
            Constants::M_PROTON, Constants::M_DEUTERON,
            Constants::Z_PROTON, Constants::Z_DEUTERON,
            pot, pot,  // same for both spin states
            1.0/3.0, 2.0/3.0
        );
        
        TH1F* hist = new TH1F(
            Form("hCk_%s_R%.1f", name.c_str(), R),
            Form("%s, R = %.1f fm; k* (MeV/c); C(k*)", name.c_str(), R),
            nbins, kmin, kmax
        );
        
        for (size_t i = 0; i < k_values.size(); ++i) {
            double k_fm = k_values[i] / Constants::HBARC;
            double C = wf.correlation_function_optimized(k_fm, R, 1000000);
            
            hist->SetBinContent(i+1, C);
            
            if ((i+1) % 20 == 0) {
                std::cout << "  Progress: " << (i+1) << "/" << k_values.size() << std::endl;
            }
        }
        
        hist->Write();
        delete hist;
    }
    
    outfile->Close();
    delete outfile;
    
    std::cout << "\nDone! Output written to output/potential_comparison.root" << std::endl;
}

// Example with pd system using double Gaussian
void pd_double_gaussian() {
    std::cout << "p-d correlation with double Gaussian potential..." << std::endl;
    
    // Define potentials based on physical parameters
    // Singlet: more repulsive
    auto pot_singlet = std::make_shared<DoubleGausPotential>(
        60.0,  // V1 (attractive core strength, MeV)
        0.6,   // a1 (core range, fm)
        -40.0,  // V2 (repulsive barrier, MeV)
        1.5    // a2 (barrier range, fm)
    );
    
    // Triplet: more attractive
    auto pot_triplet = std::make_shared<DoubleGausPotential>(
        80.0,  // V1
        0.7,   // a1
        -20.0,  // V2
        1.8    // a2
    );
    
    NumericalCoulombStrongWavefunction wf(
        Constants::M_PROTON, Constants::M_DEUTERON,
        Constants::Z_PROTON, Constants::Z_DEUTERON,
        pot_singlet, pot_triplet,
        1.0/3.0, 2.0/3.0
    );
    
    TFile* outfile = TFile::Open("output/pd_double_gaussian.root", "RECREATE");
    
    int nbins = 200;
    double kmin = 1.0, kmax = 201.0;
    
    double R_values[] = {1.059, 2.0, 3.0, 4.0};
    int n_R = 4;
    
    for (int iR = 0; iR < n_R; ++iR) {
        double R = R_values[iR];
        std::cout << "\nProcessing R = " << R << " fm" << std::endl;
        
        TH1F* hist = new TH1F(
            Form("hCk_R%.3f", R),
            Form("p-d, R = %.3f fm; k* (MeV/c); C(k*)", R),
            nbins, kmin, kmax
        );
        
        for (int i = 0; i < nbins; ++i) {
            double k_val = kmin + i * (kmax - kmin) / (nbins - 1) + 0.5;
            double k_fm = k_val / Constants::HBARC;
            
            double C = wf.correlation_function_optimized(k_fm, R, 100000);
            hist->SetBinContent(i+1, C);
            
            if ((i+1) % 20 == 0) {
                std::cout << "  Progress: " << (i+1) << "/" << nbins << std::endl;
            }
        }
        
        hist->Write();
        delete hist;
    }
    
    outfile->Close();
    delete outfile;
    
    std::cout << "\nDone!" << std::endl;
}
