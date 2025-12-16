#include <TH1F.h>
#include <TFile.h>

#include "include/CoulombWavefunction.h"
#include "include/LednickyCoulombWavefunction.h"

void lednicky_integration() {
    std::cout << "Starting Lednicky integration..." << std::endl;
    
    // Scattering parameters (Arvieux et al.)
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

    CoulombWavefunction wf_Coulomb(
        Constants::M_PROTON, Constants::M_DEUTERON,
        Constants::Z_PROTON, Constants::Z_DEUTERON
    );
    
    TFile* outfile = TFile::Open("output/lednicky_integration.root", "RECREATE");
    
    // k values
    int nbins = 400;
    double kmin = 1.0, kmax = 401.0;
    std::vector<double> k_values(nbins);
    for (int i = 0; i < nbins; ++i) {
        k_values[i] = kmin + i * (kmax - kmin) / (nbins - 1) + 0.5;
    }
    
    double R_values[] = {1.059, 1.2, 2.0, 3., 4.};
    int n_R = 5;
    
    for (int iR = 0; iR < n_R; ++iR) {
        double R = R_values[iR];
        std::cout << "\nProcessing R = " << R << " fm" << std::endl;
        
        TH1F* hist = new TH1F(
            Form("hCk_R%.3f", R),
            Form("R = %.3f fm; k* (MeV/c); C(k*)", R),
            nbins, kmin, kmax
        );

        TH1F* histCoulomb = new TH1F(
            Form("hCk_Coulomb_R%.3f", R),
            Form("Coulomb-only, R = %.3f fm; k* (MeV/c); C(k*)", R),
            nbins, kmin, kmax
        );
        
        for (size_t i = 0; i < k_values.size(); ++i) {
            double k_fm = k_values[i] / Constants::HBARC; // fm^-1
            double C = wf.correlation_function_optimized(k_fm, R, 100000);
            double C_Coulomb = wf_Coulomb.correlation_function_optimized(k_fm, R, 100000);
            
            hist->Fill(k_values[i], C);
            histCoulomb->Fill(k_values[i], C_Coulomb);
            //std::cout << "DEBUG: Computed values - k* = " << k_values[i] << "MeV/c, C(k*) = " << C << std::endl;
            
            if ((i+1) % 20 == 0) {
                std::cout << "  Progress: " << (i+1) << "/" << k_values.size() << std::endl;
            }
        }
        
        outfile->cd();
        hist->Write();
        histCoulomb->Write();
        delete hist;
        delete histCoulomb;
    }
    
    outfile->Close();
    delete outfile;
    
    std::cout << "\nDone! Output written to output/lednicky_integration.root" << std::endl;
}

#include <TH1F.h>
#include <TFile.h>

#include "include/CoulombWavefunction.h"
#include "include/LednickyCoulombWavefunction.h"

void lednicky_integration_pHe3() {
    std::cout << "Starting Lednicky integration..." << std::endl;
    
    // Scattering parameters (Rojik et al.)
    double a0s_pd = 11.26;   // fm - singlet
    double r0s_pd = 1.65;    // fm - singlet
    double a0t_pd = 9.06;    // fm - triplet
    double r0t_pd = 1.36;    // fm - triplet
    
    LednickyCoulombWavefunction wf(
        Constants::M_PROTON, Constants::M_HELIUM3,
        Constants::Z_PROTON, Constants::Z_HELIUM3,
        a0s_pd, r0s_pd, a0t_pd, r0t_pd,
        0.25, 0.75
    );

    CoulombWavefunction wf_Coulomb(
        Constants::M_PROTON, Constants::M_HELIUM3,
        Constants::Z_PROTON, Constants::Z_HELIUM3
    );
    
    TFile* outfile = TFile::Open("output/lednicky_integration_pHe3.root", "RECREATE");
    
    // k values
    int nbins = 400;
    double kmin = 1.0, kmax = 401.0;
    std::vector<double> k_values(nbins);
    for (int i = 0; i < nbins; ++i) {
        k_values[i] = kmin + i * (kmax - kmin) / (nbins - 1) + 0.5;
    }
    
    double R_values[] = {1.059, 1.2, 2.0, 3., 4., 5., 6.23, 7.};
    int n_R = 5;
    
    for (int iR = 0; iR < n_R; ++iR) {
        double R = R_values[iR];
        std::cout << "\nProcessing R = " << R << " fm" << std::endl;
        
        TH1F* hist = new TH1F(
            Form("hCk_R%.3f", R),
            Form("R = %.3f fm; k* (MeV/c); C(k*)", R),
            nbins, kmin, kmax
        );

        TH1F* histCoulomb = new TH1F(
            Form("hCk_Coulomb_R%.3f", R),
            Form("Coulomb-only, R = %.3f fm; k* (MeV/c); C(k*)", R),
            nbins, kmin, kmax
        );
        
        for (size_t i = 0; i < k_values.size(); ++i) {
            double k_fm = k_values[i] / Constants::HBARC; // fm^-1
            double C = wf.correlation_function_optimized(k_fm, R, 1000000);
            double C_Coulomb = wf_Coulomb.correlation_function_optimized(k_fm, R, 100000);
            
            hist->Fill(k_values[i], C);
            histCoulomb->Fill(k_values[i], C_Coulomb);
            //std::cout << "DEBUG: Computed values - k* = " << k_values[i] << "MeV/c, C(k*) = " << C << std::endl;
            
            if ((i+1) % 20 == 0) {
                std::cout << "  Progress: " << (i+1) << "/" << k_values.size() << std::endl;
            }
        }
        
        outfile->cd();
        hist->Write();
        histCoulomb->Write();
        delete hist;
        delete histCoulomb;
    }
    
    outfile->Close();
    delete outfile;
    
    std::cout << "\nDone! Output written to output/lednicky_integration.root" << std::endl;
}

