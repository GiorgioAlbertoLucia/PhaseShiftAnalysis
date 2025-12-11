#include <TH1F.h>
#include <TFile.h>

#include "include/LednickyCoulombWavefunction.h"

void lednicky_integration() {
    std::cout << "Starting Lednicky integration..." << std::endl;
    
    // Scattering parameters
    double a0s_pd = -2.73;   // fm - singlet
    double r0s_pd = 2.27;    // fm - singlet
    double a0t_pd = -11.88;  // fm - triplet
    double r0t_pd = 2.63;    // fm - triplet
    
    LednickyCoulombWavefunction wf(
        Constants::M_PROTON, Constants::M_DEUTERON,
        Constants::Z_PROTON, Constants::Z_DEUTERON,
        a0s_pd, r0s_pd, a0t_pd, r0t_pd,
        1.0/3.0, 2.0/3.0
    );
    
    TFile* outfile = TFile::Open("output/lednicky_integration.root", "RECREATE");
    if (!outfile || outfile->IsZombie()) {
        std::cerr << "Error: Cannot create output file!" << std::endl;
        return;
    }
    
    // k values
    int nbins = 200;
    double kmin = 1.0, kmax = 401.0;
    std::vector<double> k_values(nbins);
    for (int i = 0; i < nbins; ++i) {
        k_values[i] = kmin + i * (kmax - kmin) / (nbins - 1) + 0.5;
    }
    
    // r points
    std::vector<double> r_points(100);
    for (int i = 0; i < 100; ++i) {
        r_points[i] = 0.1 + i * (50.0 - 0.1) / 99.0;
    }
    
    double R_values[] = {1.059, 1.2, 2.0};
    int n_R = 3;
    
    for (int iR = 0; iR < n_R; ++iR) {
        double R = R_values[iR];
        std::cout << "\nProcessing R = " << R << " fm" << std::endl;
        
        TH1F* hist = new TH1F(
            Form("hCk_R%.3f", R),
            Form("R = %.3f fm; k* (MeV/c); C(k*)", R),
            nbins, kmin, kmax
        );
        
        for (size_t i = 0; i < k_values.size(); ++i) {
            double k_fm = k_values[i] / Constants::HBARC;
            double C = wf.correlation_function_optimized(k_fm, R, r_points, 10, 10);
            
            hist->Fill(k_values[i], C);
            
            if ((i+1) % 20 == 0) {
                std::cout << "  Progress: " << (i+1) << "/" << k_values.size() << std::endl;
            }
        }
        
        outfile->cd();
        hist->Write();
        delete hist;
    }
    
    outfile->Close();
    delete outfile;
    
    std::cout << "\nDone! Output written to output/lednicky_integration.root" << std::endl;
}
