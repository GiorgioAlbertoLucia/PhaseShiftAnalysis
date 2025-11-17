#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <complex>
#include <algorithm>
#include <format>

// ROOT includes
#include "TCanvas.h"
#include "TString.h"
#include "TLegend.h"
#include "TH1F.h"
#include "TStyle.h"
#include "TLatex.h"
#include <TFile.h>
#include <TSystem.h>
#include <TROOT.h>

#include "include/Physics.h"
#include "include/Potentials.h"
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

void drawExpectedWavefunctions(const std::vector<double>& r, const std::vector<double>& ylNumerical, 
                               const std::vector<double>& ylAsymptotic, TDirectory* outfile) {
    
    gStyle->SetOptStat(0);
    gStyle->SetPalette(kViridis);
    
    TCanvas canvas("cPotential", "Nuclear Optical Potential", 1200, 800);
    gPad->SetLeftMargin(0.12);
    gPad->SetRightMargin(0.05);

    const double rStep = r[1] - r[0];
    std::vector<double> rEdges(r.size()+1);
    rEdges[0] = r[0] - rStep/2.;
    for (size_t ir = 0; ir < r.size(); ir++)
        rEdges[ir+1] = r[ir] + rStep/2.;
    
    TH1F hNumerical("hNumerical", "; #it{r} (fm); #it{y}_#it{l}(#it{r})", r.size(), &rEdges[0]);
    hNumerical.SetLineColor(kRed);
    TH1F hAsymptotic("hAsymptotic", "; #it{r} (fm); #it{y}_#it{l}(#it{r})", r.size(), &rEdges[0]);
    hAsymptotic.SetLineColor(kBlue);

    const double maxNumerical = *std::max_element(ylNumerical.begin(), ylNumerical.end());
    const double maxAsymptotic = *std::max_element(ylAsymptotic.begin(), ylAsymptotic.end());

    for (size_t ir = 0; ir < r.size(); ir++) {
        hNumerical.SetBinContent(ir, ylNumerical[ir]);
        hAsymptotic.SetBinContent(ir, ylAsymptotic[ir]*maxNumerical/maxAsymptotic);
    }

    TLegend legend(0.55, 0.74, 0.88, 0.88);
    legend.SetBorderSize(0);
    legend.SetFillStyle(0);
    legend.AddEntry(&hNumerical, "Numerical", "l");
    legend.AddEntry(&hAsymptotic, "Asymptotic", "l");
    
    canvas.cd();
    hNumerical.Draw("hist same");
    hAsymptotic.Draw("hist same");
    legend.Draw("same");

    outfile->cd();
    canvas.Write();
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

void PlotExpectedWavefunctions() {

    gROOT->SetBatch(true);

    loadData();

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

    // Input potential parameters (Dimitar)
    std::vector<double> fitPars = {-18.17, 1.689, -16.75, 1.821};
    std::vector<double> fitParErrors = {0., 0., 0., 0.};

    g_potential->setParameters(&fitPars[0]);
    g_potential->setParameterErrors(&fitParErrors[0]);

    const double mu = reducedMass(constant::M_PROTON, constant::M_HE3);
    const double dr = 0.0001;
    const double rAsymptotic = 28.0;
    const double rMax = 30.0;
    NumerovSolver solver(dr, rMax, rAsymptotic, 1.0);
    PhaseShiftExtractor extractor(mu, dr, g_data.l_value);
    std::vector<double> numericalSolution, asymptoticSolution; // numerical solution to the Schroedinger's equation

    const int nRpoints = static_cast<int>(rMax / dr);
    std::vector<double> r(nRpoints);
    for (int ir = 0; ir < nRpoints; ++ir)
        r[ir] = (ir + 1) * dr;

    TFile * outfile = new TFile("output/wavefunctions.root", "recreate");
    for (const auto& energy : g_data.energies) {
        
        const double v = std::sqrt(2. * energy / constant::M_PROTON); // incident proton beam velocity
        const double incidentMomentum = v * constant::M_PROTON;
        const double kstar = getKstar(incidentMomentum);
        const double k = waveNumber(kstar);
        const double drAsymptotic = 0.1 / k;

        const double relativeVelocity = kstar / mu; // to be checked
        const double eta = constant::ALPHA_EM * constant::Z_PROTON * constant::Z_HE3 / relativeVelocity;
        numericalSolution = solver.solveEquation(g_data.l_value, k, *g_potential, constant::Z_PROTON, constant::Z_HE3);

        double F_l[1], Fp_l[1];
        for (int ir = 0; ir < nRpoints; ++ir) {
            const double kr = k * r[ir];
            gsl_sf_coulomb_wave_F_array(g_data.l_value, /*L_G*/ 0, eta, std::abs(kr), F_l, Fp_l);
            asymptoticSolution.emplace_back(F_l[0]);
        }

        auto idir = outfile->mkdir(Form("cK_%f", k));
        drawExpectedWavefunctions(r, numericalSolution, asymptoticSolution, idir);
        asymptoticSolution.clear();
    }
    
    outfile->Close();

}