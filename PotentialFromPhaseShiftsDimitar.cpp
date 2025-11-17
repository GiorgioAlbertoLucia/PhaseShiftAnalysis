#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <complex>
#include <algorithm>

// ROOT includes
#include "TCanvas.h"
#include "TGraphErrors.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TLegend.h"
#include "TAxis.h"
#include "TF1.h"
#include "TMinuit.h"
#include "TMath.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TROOT.h"
#include <TFile.h>
#include <TRandom3.h>
#include <TSystem.h>

#include "include/HankelFunctions.h"
#include "include/Physics.h"
#include "include/Potentials.h"
#include "include/NewtonRaphsonPhaseShift.h"
#include "include/NumerovSolver.h"
#include "include/PhaseShiftExtractor.h"


// Global data structure for phase shifts
struct PhaseShiftData {
    int l_value;    // angular momenutum
    std::vector<double> energies;
    std::vector<double> exp_phases;
    std::vector<double> errors;
};

// Global instance for TMinuit access
PhaseShiftData g_data;

// Global potential for TMinuit
Potential* g_potential;

typedef struct initialisationParameter {
    double value, step, min, max;
    initialisationParameter() = default;
    initialisationParameter(const double _value, const double _step, const double _min, const double _max) :
        value(_value), step(_step), min(_min), max(_max) {}
} initialisationParameter;

double findOptimalMatchingRadius(const std::vector<double>& u_num, 
                                  double k, double dr) {
    // Find where potential becomes negligible (< 0.1 MeV)
    double r_start = 10.0;  // Start looking from 10 fm
    
    for (double r = r_start; r < 25.0; r += 0.5) {
        size_t idx = static_cast<size_t>(r / dr);
        if (idx >= u_num.size()) break;
        
        // Check if we're in asymptotic regime:
        // Solution should be smoothly varying
        if (idx + 2 < u_num.size()) {
            double u1 = u_num[idx];
            double u2 = u_num[idx + 1];
            double u3 = u_num[idx + 2];
            
            // Check smoothness and non-zero
            if (std::abs(u1) > 1e-10 && std::abs(u2) > 1e-10) {
                double ratio1 = u2 / u1;
                double ratio2 = u3 / u2;
                
                // If ratios are similar, we're in smooth regime
                if (std::abs(ratio1 - ratio2) < 0.1) {
                    return r;
                }
            }
        }
    }
    
    return 20.0;  // Default fallback
}

// Chi-squared function for TMinuit
void chi2Function(Int_t& /*npar*/, Double_t* /*gin*/, Double_t& f, Double_t* par, Int_t /*iflag*/) {

    g_potential->setParameters(par);
    
    double chi2 = 0.0;
    const double mu = reducedMass(constant::M_PROTON, constant::M_HE3);
    const double dr = 0.0001;
    const double r_asymptotic = 28.0;
    NumerovSolver solver(dr, 30.0, r_asymptotic, 1.0);
    PhaseShiftExtractor extractor(mu, dr, g_data.l_value);
    std::vector<double> numerical_solution; // numerical solution to the Schroedinger's equation
    
    for (size_t i = 0; i < g_data.exp_phases.size(); ++i) {

        const double v = TMath::Sqrt(2. * g_data.energies[i] / constant::M_PROTON); // incident proton beam velocity
        const double incidentMomentum = v * constant::M_PROTON;
        const double kstar = getKstar(incidentMomentum);
        const double k = waveNumber(kstar);
        const double dr_asymptotic = 0.1 / k;
        //const double relativeVelocity = 2 * kstar / (constant::M_PROTON+constant::M_HE3);
        const double relativeVelocity = kstar / mu; // to be checked
        const double eta = constant::ALPHA_EM * constant::Z_PROTON * constant::Z_HE3 / relativeVelocity;

        numerical_solution = solver.solveEquation(g_data.l_value, k, *g_potential, constant::Z_PROTON, constant::Z_HE3);
        
        const double r_asymptotic_stable = findOptimalMatchingRadius(numerical_solution, k, dr);
        const double calc_phase = extractor.extractPhaseShift(numerical_solution, r_asymptotic_stable, dr_asymptotic, k, eta);
        //const double calc_phase = extractor.extractPhaseShift(numerical_solution, r_asymptotic, dr_asymptotic, k, eta);
        //double calc_phase = solver.solveAndExtractPhase(g_data.l_value, k, *g_potential, constant::Z_PROTON, constant::Z_HE3);
        double diff = calc_phase - g_data.exp_phases[i];
        
        while (diff > TMath::Pi() / 2.) diff -= TMath::Pi();
        while (diff < -TMath::Pi() / 2.) diff += TMath::Pi();
        
        chi2 += (diff * diff) / (g_data.errors[i] * g_data.errors[i]);
    }
    
    f = chi2;
}

// TO UPDATE
//void chi2FunctionRegularization(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag) {
//
//    g_potential->setParameters(par);
//    
//    double chi2 = 0.0;
//    double smoothness_penalty = 0.0;
//    double wrapping_penalty = 0.0;
//    
//    double mu = reducedMass(constant::M_PROTON, constant::M_HE3);
//    NumerovSolver solver(0.0005, 30.0, 28.0, 1.0);
//    
//    std::vector<double> calc_phases;
//    calc_phases.reserve(g_data.exp_phases.size());
//    
//    for (size_t i = 0; i < g_data.exp_phases.size(); ++i) {
//        const double v = TMath::Sqrt(2. * g_data.energies[i] / constant::M_PROTON);
//        const double incidentMomentum = v * constant::M_PROTON;
//        const double kstar = getKstar(incidentMomentum);
//        double k = waveNumber(kstar);
//        
//        double calc_phase = solver.solveAndExtractPhase(g_data.l_value, k, 
//                                                        *g_potential, constant::Z_PROTON, constant::Z_HE3);
//        calc_phases.push_back(calc_phase);
//        
//        // Direct comparison (unwrapped)
//        double diff = calc_phase - g_data.exp_phases[i];
//        
//        // Penalize wrapping: if the difference is large, it means we had to wrap
//        // This discourages solutions that only match after wrapping
//        double raw_diff = TMath::Abs(diff);
//        if (raw_diff > TMath::Pi() / 2.) {
//            wrapping_penalty += 100.0 * (raw_diff - TMath::Pi() / 2.);  // Linear penalty beyond π/2
//        }
//        
//        // Standard chi-squared with minimal wrapping (only for numerical precision)
//        while (diff > TMath::Pi()) diff -= 2.0 * TMath::Pi();
//        while (diff < -TMath::Pi()) diff += 2.0 * TMath::Pi();
//        
//        chi2 += (diff * diff) / (g_data.errors[i] * g_data.errors[i]);
//    }
//    
//    // Smoothness constraint: penalize large changes in phase shift between adjacent energies
//    // Phase shifts should vary smoothly with energy
//    double lambda_smooth = 10.0;  // Smoothness weight (tune this)
//    for (size_t i = 1; i < calc_phases.size(); ++i) {
//        double delta_E = g_data.energies[i] - g_data.energies[i-1];
//        double delta_phase = calc_phases[i] - calc_phases[i-1];
//        
//        // Expected smooth change (from experimental data)
//        double exp_delta_phase = g_data.exp_phases[i] - g_data.exp_phases[i-1];
//        
//        // Penalize deviations from expected smoothness
//        double smoothness_diff = (delta_phase - exp_delta_phase) / delta_E;
//        smoothness_penalty += lambda_smooth * smoothness_diff * smoothness_diff;
//    }
//    
//    // Total chi-squared with penalties
//    f = chi2 + smoothness_penalty + wrapping_penalty;
//}

void drawPotential(const Potential& pot, double chi2, int ndf, TFile* outfile) {
    
    gStyle->SetOptStat(0);
    gStyle->SetPalette(kViridis);
    
    TCanvas canvas("cPotential", "Nuclear Optical Potential", 1200, 800);
    gPad->SetLeftMargin(0.12);
    gPad->SetRightMargin(0.05);
    
    TF1 f_real("f_real", [&pot](double *x, double * /*p*/) {
        return pot.eval(x[0]);
    }, 0.1, 15.0, 0);
    f_real.SetLineColor(kRed);
    f_real.SetLineWidth(3);
    
    TF1 f_coulomb("f_coulomb", [](double *x, double * /*p*/) {
        return coulombPotential(x[0], constant::Z_PROTON, constant::Z_HE3);
    }, 0.1, 15.0, 0);
    f_coulomb.SetLineColor(kBlue);
    f_coulomb.SetLineWidth(3);
    f_coulomb.SetLineStyle(2);
    
    TF1 f_total("f_total", [&pot](double *x, double * /*p*/) {
        return pot.evalWithCoulomb(x[0], constant::Z_HE3, constant::Z_PROTON);
    }, 0.1, 15.0, 0);
    f_total.SetLineColor(kBlack);
    f_total.SetLineWidth(3);
    
    double ymin = std::min({f_total.GetMinimum(), f_real.GetMinimum(), f_coulomb.GetMinimum()});
    canvas.DrawFrame(0, ymin, 15, 20, ";#it{r} (fm); V(#it{r}) (MeV)");
    f_real.Draw("same");
    f_coulomb.Draw("same");
    f_total.Draw("same");
    
    TLegend legend(0.55, 0.74, 0.88, 0.88);
    legend.SetBorderSize(0);
    legend.SetFillStyle(0);
    legend.AddEntry(&f_real, "Strong", "l");
    legend.AddEntry(&f_coulomb, "Coulomb", "l");
    legend.AddEntry(&f_total, "Total", "l");
    legend.Draw();
    
    TLatex latex;
    latex.SetTextSize(0.035);
    latex.SetTextAlign(12);
    for (int iparam = 0; iparam < pot.getNPars(); iparam++) {
        latex.DrawLatexNDC(0.55, 0.35-(0.05*iparam), Form("%s = %.2f", pot.getParameterName(iparam).c_str(), pot.getParameter(iparam)));    
    }
    latex.DrawLatexNDC(0.55, 0.40, Form("#chi^{2}/ndf = %.2f/%d", chi2, ndf));
    
    canvas.Update();
    canvas.SaveAs("output/potential_fit.pdf");

    outfile->cd();
    canvas.Write();
    
    std::cout << "\nPlot saved as potential_fit.pdf and potential_fit.pdf" << std::endl;
}

void drawPhaseShiftComparison(const Potential& pot, double chi2, int ndf, TFile* outfile) {
    
    gStyle->SetOptStat(0);
    
    TCanvas canvas("canvas", "Phase Shift Comparison", 1200, 800);
    canvas.DrawFrame(4, -90, 12, -50, ";#it{E}_{lab} (MeV);#delta (deg.)");
    gPad->SetLeftMargin(0.12);
    gPad->SetRightMargin(0.05);
    
    // Calculate fitted phase shifts
    //double mu = reducedMass(constant::M_PROTON, constant::M_HE3);
    //NumerovSolver solver(0.0005, 30.0, 28.0, 1.0);

    //double chi2 = 0.0;
    const double mu = reducedMass(constant::M_PROTON, constant::M_HE3);
    const double dr = 0.0001;
    const double r_asymptotic = 15.0; // 28.0
    NumerovSolver solver(dr, 30.0, r_asymptotic, 1.0);
    PhaseShiftExtractor extractor(mu, dr, g_data.l_value);
    std::vector<double> numerical_solution; // numerical solution to the Schroedinger's equation
    
    std::cout << "\nCalculated phases: [ ";
    std::vector<double> fitted_phases;
    for (size_t i = 0; i < g_data.exp_phases.size(); ++i) {

        const double v = TMath::Sqrt(2. * g_data.energies[i] / constant::M_PROTON); // incident proton beam velocity
        const double incidentMomentum = v * constant::M_PROTON;
        const double kstar = getKstar(incidentMomentum);
        const double k = waveNumber(kstar);
        const double dr_asymptotic = 0.1 / k;
        //const double relativeVelocity = 2 * kstar / (constant::M_PROTON+constant::M_HE3);
        const double relativeVelocity = kstar / mu; // to be checked
        const double eta = constant::ALPHA_EM * constant::Z_PROTON * constant::Z_HE3 / relativeVelocity;

        numerical_solution = solver.solveEquation(g_data.l_value, k, *g_potential, constant::Z_PROTON, constant::Z_HE3);
        const double calc_phase = extractor.extractPhaseShift(numerical_solution, r_asymptotic, dr_asymptotic, k, eta);
        //double calc_phase = solver.solveAndExtractPhase(g_data.l_value, k, *g_potential, constant::Z_PROTON, constant::Z_HE3);

        //const double v = TMath::Sqrt(2. * g_data.energies[i] / constant::M_PROTON); // incident proton beam velocity
        //const double incidentMomentum = v * constant::M_PROTON;
        //const double kstar = getKstar(incidentMomentum);
        //double k = waveNumber(kstar);
        //double calc_phase = solver.solveAndExtractPhase(g_data.l_value, k, 
        //                                                pot, constant::Z_PROTON, constant::Z_HE3);
        
        //std::cout << "calc_phase (rad)" << calc_phase << ", ";
        fitted_phases.push_back(calc_phase * 180.0 / TMath::Pi()); // Convert to degrees
        std::cout << fitted_phases[i] << ", ";
    }
    std::cout << "]\n ";
    
    std::cout << "\nExperimental phases: [ ";
    std::vector<double> exp_phases_deg;
    std::vector<double> errors_deg;
    for (size_t i = 0; i < g_data.exp_phases.size(); ++i) {
        exp_phases_deg.push_back(g_data.exp_phases[i] * 180.0 / TMath::Pi());
        errors_deg.push_back(g_data.errors[i] * 180.0 / TMath::Pi());
        std::cout << exp_phases_deg[i] << ", ";
    }
    std::cout << "]\n ";
    
    TGraphErrors g_exp(g_data.energies.size(), g_data.energies.data(),
                       exp_phases_deg.data(), nullptr, errors_deg.data());
    g_exp.SetMarkerStyle(20);
    g_exp.SetMarkerSize(1.2);
    g_exp.SetMarkerColor(kBlue);
    g_exp.SetLineColor(kBlue);
    g_exp.SetTitle("Phase Shift Comparison;E_{lab} (MeV);#delta_{0} (deg)");
    
    TGraph g_fit(g_data.energies.size(), g_data.energies.data(), fitted_phases.data());
    g_fit.SetLineColor(kRed);
    g_fit.SetLineWidth(3);
    g_fit.SetMarkerStyle(24);
    g_fit.SetMarkerSize(1.0);
    g_fit.SetMarkerColor(kRed);
    
    g_fit.Draw("P same");
    g_exp.Draw("P same");
    
    TLegend legend(0.15, 0.75, 0.45, 0.88);
    legend.SetBorderSize(0);
    legend.SetFillStyle(0);
    legend.AddEntry(&g_exp, "Experimental", "lep");
    legend.AddEntry(&g_fit, "Fitted", "lp");
    legend.Draw();
    
    TLatex latex;
    latex.SetTextSize(0.035);
    latex.SetTextAlign(12);
    latex.DrawLatexNDC(0.65, 0.85, Form("#chi^{2}/ndf = %.2f/%d", chi2, ndf));
    
    canvas.Update();
    canvas.SaveAs("output/phase_shift_comparison.pdf");

    outfile->cd();
    canvas.Write("cPhseShiftsComparison");
    
    std::cout << "Phase shift comparison saved as phase_shift_comparison.pdf" << std::endl;
}

void loadData() {

    g_data.l_value = 0;
    g_data.energies = {4.55, 5.51, 6.52, 7.51, 8.51, 9.51, 10.38, 11.45}; // MeV
    g_data.exp_phases = {-57.1, -61.97, -66.33, -70.69, -74.45, -77.70, -80.20, -83.70}; // deg
    g_data.errors = {0.53, 0.5, 0.46, 0.41, 0.38, 0.36, 0.35, 0.33}; // deg

    for (auto& phase: g_data.exp_phases)
        phase = phase * TMath::Pi() / 180.;
    for (auto& phase_error: g_data.errors)
        phase_error = phase_error * TMath::Pi() / 180.;
}

void roughScanParameterGeneral(TMinuit* minuit, int ierflg, std::vector<initialisationParameter>& initialisationParameters ) {

    std::cout << "\n=== Parameter Space Scan ===" << std::endl;
    double best_chi2 = 1e10;
    const int nPars = initialisationParameters.size();
    std::vector<double> bestPars(nPars, 0.);

    const int nTrials = 50;
    for (int itrial = 0; itrial < nTrials; itrial++) {
        std::vector<double> trialPars(nPars, 0.);
        for (int iparam = 0; iparam < nPars; iparam++) {
            trialPars[iparam] = gRandom->Uniform(initialisationParameters[iparam].min, initialisationParameters[iparam].max);
        }
        Int_t npar_dummy = 3;
        Double_t gin_dummy[10];
        Double_t f_test;
        chi2Function(npar_dummy, gin_dummy, f_test, trialPars.data(), 0);
                
        if (f_test < best_chi2) {
            best_chi2 = f_test;
            for (int iparam = 0; iparam < nPars; iparam++) {
                bestPars[iparam] = trialPars[iparam];
            }
        }
    }

    for (int iparam = 0; iparam < g_potential->getNPars(); iparam++) {
        minuit->mnparm(iparam, g_potential->getParameterName(iparam), 
                      bestPars[iparam], initialisationParameters[iparam].step, 
                      initialisationParameters[iparam].min, initialisationParameters[iparam].max, ierflg);
    }
}

void PotentialFromPhaseShifts() {

    gROOT->SetBatch(true);

    loadData();
    
    std::cout << "==================================================" << std::endl;
    std::cout << "  Nuclear Optical Potential Fitter for p-3He" << std::endl;
    std::cout << "==================================================" << std::endl;

    //g_potential = new WoodsSaxonPotential();
    //g_potential = new GausPotential();
    g_potential = new DoubleGausPotential();

    std::vector<initialisationParameter> initPars(g_potential->getNPars());
    for (int iparam = 0; iparam < g_potential->getNPars(); iparam++) {
        switch(iparam) {
            // Woods-Saxon
            //case 0: initPars[iparam] = initialisationParameter(80.0, 5.0, 10.0, 150.0); break;
            //case 1: initPars[iparam] = initialisationParameter(2.5, 0.1, 1.5, 4.0); break;
            //case 2: initPars[iparam] = initialisationParameter(0.65, 0.05, 0.3, 2.);    break;
            // Double Gaus
            case 0: initPars[iparam] = initialisationParameter(80.0, 5.0, 10.0, 200.0);   break;
            case 1: initPars[iparam] = initialisationParameter(2.5, 0.1, 1.5, 4.0);   break;
            case 2: initPars[iparam] = initialisationParameter(80.0, 5.0, 10.0, 200.0);  break;
            //case 2: initPars[iparam] = initialisationParameter(-80.0, 5.0, -200.0, -10.0);  break;
            case 3: initPars[iparam] = initialisationParameter(2.5, 0.1, 1.5, 4.0); break;
        }
    }

    std::cout << "n entries in initpars = " << initPars.size() << std::endl;
    
    TMinuit *minuit = new TMinuit(g_potential->getNPars());
    
    minuit->SetPrintLevel(1); // Print level (-1: quiet, 0: normal, 1: verbose)
    
    // Parameters: name, start value, step, min, max
    Double_t arglist[10];
    Int_t ierflg = 0;
    for (int iparam = 0; iparam < g_potential->getNPars(); iparam++) {
        minuit->mnparm(iparam, g_potential->getParameterName(iparam), initPars[iparam].value, initPars[iparam].step, initPars[iparam].min, initPars[iparam].max, ierflg);
    }

    roughScanParameterGeneral(minuit, ierflg, initPars);

    std::cout << "\n=== First Pass: Simple Chi2 ===" << std::endl;
    minuit->SetFCN(chi2Function);
    arglist[0] = 2;
    minuit->mnexcm("SET STR", arglist, 1, ierflg);
    arglist[0] = 10000;
    arglist[1] = 0.5;
    minuit->mnexcm("MIGRAD", arglist, 2, ierflg);

    //std::vector<double> fitPars(g_potential->getNPars());
    //std::vector<double> fitParErrors(g_potential->getNPars());
    //double dummyParam, dummyError;
    //
    //for (int iparam = 0; iparam < g_potential->getNPars(); iparam++) {
    //    minuit->GetParameter(iparam, dummyParam, dummyError);    
    //    fitPars[iparam] = dummyParam;
    //    fitParErrors[iparam] = dummyError;
    //}

    // Parameters from Dimitar
    std::vector<double> fitPars = {-18.17, 1.689, -16.75, 1.821};
    std::vector<double> fitParErrors = {0., 0., 0., 0.};

    g_potential->setParameters(&fitPars[0]);
    g_potential->setParameterErrors(&fitParErrors[0]);
    
    Double_t amin, edm, errdef;
    Int_t nvpar, nparx, icstat;
    minuit->mnstat(amin, edm, errdef, nvpar, nparx, icstat);
    
    // Output results
    std::cout << "\n==================================================" << std::endl;
    std::cout << "            Best Fit Parameters" << std::endl;
    std::cout << "==================================================" << std::endl;
    for (int iparam = 0; iparam < g_potential->getNPars(); iparam++) {
        std::cout << g_potential->getParameterName(iparam) << "    = " << g_potential->getParameter(iparam) << " ± " << g_potential->getParameterError(iparam) << " (unit) " << std::endl;    
    }
    std::cout << "--------------------------------------------------" << std::endl;
    int ndf = g_data.exp_phases.size() - g_potential->getNPars();
    std::cout << "χ²    = " << amin << std::endl;
    std::cout << "ndf   = " << ndf << std::endl;
    std::cout << "χ²/ndf = " << amin/ndf << std::endl;
    std::cout << "==================================================" << std::endl;

    TFile * outfile = new TFile("output/potential_from_phase_shifts_doublegaus.root", "recreate");
    drawPotential(*g_potential, amin, ndf, outfile);
    drawPhaseShiftComparison(*g_potential, amin, ndf, outfile);
    outfile->Close();
    
    delete minuit;

}