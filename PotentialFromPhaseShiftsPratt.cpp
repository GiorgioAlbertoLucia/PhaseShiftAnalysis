#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <complex>
#include <algorithm>
#include <Math/SpecFunc.h>
#include <gsl/gsl_sf_coulomb.h>

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

#include "include/HankelFunctions.h"
#include "include/Physics.h"
#include "include/Potentials.h"



// Global data structure for phase shifts
struct PhaseShiftData {
    std::vector<int> l_values;
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


// Coulomb phase shift (analytical)
double coulombPhaseShift(int l, double k, double Z1, double Z2) {
    double eta = Z1 * Z2 * constant::E_CHARGE_SQ * reducedMass(constant::M_PROTON, constant::M_HE3) / (constant::HBARC * constant::HBARC * k);
    double sigma = 0.0;
    for (int i = 1; i <= l; ++i) {
        sigma += TMath::ATan(eta / i);
    }
    return sigma;
}

namespace Bessel {
    
    void CalcJN_real(int ell, double x, double &jl, double &nl, double &jlprime, double &nlprime){
    	int l=0;
    	double j[4], n[4], jprime[4], nprime[4];
    	const double s = sin(x), c = cos(x);
    	if(ell>3){
    		printf("increase array size in CalcJN_real\n");
    		exit(1);
    	}
    	j[0] = s;
    	n[0] = c;
    	jprime[0] = c;
    	nprime[0] = -s;
    	while(l<ell){
    		l += 1;
    		j[l] = (double(l)/x)*j[l-1]-jprime[l-1];
    		n[l] = (double(l)/x)*n[l-1]-nprime[l-1];
    		jprime[l] = j[l-1]-(double(l)/x)*j[l];
    		nprime[l] = n[l-1]-(double(l)/x)*n[l];
    	};
    	jl = j[ell];
    	nl = n[ell];
    	jlprime = jprime[ell];
    	nlprime = nprime[ell];
    }

    void CalcJN_imag(int ell, double x, double &jl, double &nl, double &jlprime, double &nlprime){
    	int l = 0;
    	double j[4], n[4], jprime[4], nprime[4];
    	double s = sinh(x);
    	double c = cosh(x);
    	if(ell>3){
    		printf("increase array size in CalcJN_imag\n");
    		exit(1);
    	}
    	j[0] = s;
    	n[0] = c;
    	jprime[0] = c;
    	nprime[0] = s;
    	while(l<ell){
    		l += 1;
    		j[l] = (double(l)/x)*j[l-1]-jprime[l-1];
    		n[l] = (double(l)/x)*n[l-1]-nprime[l-1];
    		jprime[l] = -j[l-1]-(double(l)/x)*j[l];
    		nprime[l] = -n[l-1]-(double(l)/x)*n[l];
    	};
    	jl = j[ell];
    	nl = n[ell];
    	jlprime = jprime[ell];
    	nlprime = nprime[ell];
    }
}

namespace CoulWave {
    
    void GetFGprime_ImagQ(int ellmax,double x,double eta,double *FL,double *GL,double *FLprime,double *GLprime){
    	double *F,*G,*Fprime,*Gprime;
    	F=new double[ellmax+1]; G=new double[ellmax+1]; Fprime=new double[ellmax+1]; Gprime=new double[ellmax+1];
    	double ff,root,sign;
    	int n,l,nmax;
    	double *A,*B;
    	nmax=24+lrint(fabs(x));
    	A=new double[nmax+1];
    	B=new double[nmax+1];
    	// Calc F and Fprime
    	A[0]=0.0; A[1]=1.0;
    	for(n=0;n<=nmax;n++) B[n]=0.0;
    	for(n=0;n<=nmax-2;n++) A[n+2]=(2.0*eta*A[n+1]+A[n])/double((n+1)*(n+2));
    	F[0]=Fprime[0]=0.0;	
    	for(n=1;n<=nmax;n++){
    		F[0]+=A[n]*pow(x,n);
    		Fprime[0]+=double(n)*A[n]*pow(x,n-1);
    	}
    	for(l=0;l<ellmax;l++){
    		root=(l+1.0)*(l+1.0)-eta*eta;
    		if(root>=0){
    			root=sqrt(root);
    			sign=1;
    		}
    		else{
    			root=sqrt(-root);
    			sign=-1;
    		}
    		//printf("root=%g\n",root);
    		ff=(((l+1.0)*(l+1.0)/x)+eta);
    		F[l+1]=(ff*F[l]-(l+1.0)*Fprime[l])/root;
    		Fprime[l+1]=-(sign*root*F[l]+ff*F[l+1])/(l+1.0);
    	}
    	// CALC G and Gprime
    	A[0]=1.0; A[1]=0.0; B[1]=2.0*eta*A[0]; B[0]=0.0;
    	for(n=0;n<=nmax-2;n++){
    		B[n+2]=(2.0*eta*B[n+1]+B[n])/((n+1.0)*(n+2.0));
    		A[n+2]=(2.0*eta*A[n+1]+A[n]+(1.0-2.0*(n+2.0))*B[n+2])/((n+1.0)*(n+2.0));
    	}
    	G[0]=1; Gprime[0]=0.0;
    	for(n=1;n<=nmax;n++){
    		G[0]+=(A[n]+B[n]*log(fabs(x)))*pow(x,n);
    		Gprime[0]+=double(n)*(A[n]+B[n]*log(fabs(x)))*pow(x,n-1)+B[n]*pow(x,n-1);
    	}
    	for(l=0;l<ellmax;l++){
    		root=(l+1.0)*(l+1.0)-eta*eta;
    		if(root>=0){
    			root=sqrt(root);
    			sign=1;
    		}
    		else{
    			root=sqrt(-root);
    			sign=-1;
    		}
    		ff=((l+1.0)*(l+1.0)/x)+eta;
    		G[l+1]=(ff*G[l]-(l+1.0)*Gprime[l])/root;
    		Gprime[l+1]=-(sign*root*G[l]+ff*G[l+1])/(l+1.0);
    	}
    
    	*FL=F[ellmax];
    	*GL=G[ellmax];
    	*FLprime=Fprime[ellmax];
    	*GLprime=Gprime[ellmax];

    	delete[] A;
    	delete[] B;
    	delete[] F;
    	delete [] G;
    	delete [] Fprime;
    	delete [] Gprime;
    }
}

class PhaseShiftCoulomb {
    
    private:

        double m1, m2, Z1, Z2, mu;

    public:

        PhaseShiftCoulomb(const double _m1, const double _m2, const double _Z1, const double _Z2) :
            m1(_m1), m2(_m2), Z1(_Z1), Z2(_Z2) { mu = reducedMass(m1, m2); }

        static void coulombWavefunction(const int l, const std::complex<double>& x, const std::complex<double>& eta, 
                                        double * Fl, double * Gl, double * Flprime, double * Glprime) {
            
            if (std::abs(std::imag(x)) > std::abs(std::real(x))) {
                if (std::abs(eta) > 1e-08) {
                    CoulWave::GetFGprime_ImagQ(l, std::imag(x), -std::imag(eta), Fl, Gl, Flprime, Glprime);
                } else {
                    Bessel::CalcJN_imag(l, std::imag(x), *Fl, *Gl, *Flprime, *Glprime);
                }

            } else {
                if (std::abs(eta) > 1e-08) {
                    double Fexp, Gexp;
	                gsl_sf_coulomb_wave_FGp_array(l, 0, std::real(eta), std::real(x), Fl, Flprime, Gl, Glprime, &Fexp, &Gexp);
                    *Fl = *Fl * std::exp(Fexp);
                    *Gl = *Gl * std::exp(Gexp);
                    *Flprime = *Flprime * std::exp(Fexp);
                    *Glprime = *Glprime * std::exp(Gexp);
                } else {
                    Bessel::CalcJN_real(l, std::real(x), *Fl, *Gl, *Flprime, *Glprime);
                }
            }
        }

        double solveAndExtractPhase(const double rAsymptotic, const int l, const double k, const Potential& pot) {
            
            const double kstar = k * constant::HBARC;
            const double E = std::sqrt(kstar*kstar + m1*m1)+std::sqrt(kstar*kstar + m2*m2);
			const double muCoulomb = 0.25 * ( E - pow(m1*m1 - m2*m2, 2)/pow(E, 3) );
            const std::complex<double> eta{constant::ALPHA_EM * Z1 * Z2 * muCoulomb / kstar, 0};
            //const std::complex<double> eta{constant::ALPHA_EM * Z1 * Z2 * mu / kstar, 0};
            const std::complex<double> kr = k * rAsymptotic;

            const double kstartShiftedSquared = kstar*kstar - 2*mu*pot.eval(rAsymptotic);
            std::complex<double> kstarShifted;
            if (kstartShiftedSquared > 0) 
                kstarShifted = std::sqrt(kstartShiftedSquared);
            else
                kstarShifted = std::complex<double>{0, std::sqrt(std::abs(kstartShiftedSquared))};
            const std::complex<double> etaShifted = constant::ALPHA_EM * Z1 * Z2 * muCoulomb / kstarShifted;
            //const std::complex<double> etaShifted = constant::ALPHA_EM * Z1 * Z2 * mu / kstarShifted;
            const std::complex<double> krShifted = kstarShifted * rAsymptotic / constant::HBARC;
            
            double Fl, Gl, Flprime, Glprime;    // Coulomb real and imaginary solutions (and their derivatives)
            double FlShifted, GlShifted, FlprimeShifted, GlprimeShifted;

            PhaseShiftCoulomb::coulombWavefunction(l, kr, eta, &Fl, &Gl, &Flprime, &Glprime);
            PhaseShiftCoulomb::coulombWavefunction(l, krShifted, etaShifted, &FlShifted, &GlShifted, &FlprimeShifted, &GlprimeShifted);

            const double beta = std::abs(kstarShifted)/kstar * FlprimeShifted / FlShifted;
            double delta = -std::atan2(beta*Fl - Flprime, beta*Gl - Glprime);

            while (delta > TMath::Pi() / 2.) delta -= TMath::Pi();
            while (delta < -TMath::Pi() / 2.) delta += TMath::Pi();

            return delta;
        }
};

// Chi-squared function for TMinuit
void chi2Function(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag) {

    g_potential->setParameters(par);
    
    double chi2 = 0.0;
    const double mu = reducedMass(constant::M_PROTON, constant::M_HE3);
    const double rAsymptotic = 1.0; // fm
    PhaseShiftCoulomb solver(constant::M_PROTON, constant::M_HE3, constant::Z_PROTON, constant::Z_HE3);
    
    for (size_t i = 0; i < g_data.l_values.size(); ++i) {

        const double v = TMath::Sqrt(2. * g_data.energies[i] / constant::M_PROTON); // incident proton beam velocity
        const double incidentMomentum = v * constant::M_PROTON;
        const double kstar = getKstar(incidentMomentum);
        double k = waveNumber(kstar);
        double calc_phase = solver.solveAndExtractPhase(rAsymptotic, g_data.l_values[i], k, *g_potential);
        double diff = calc_phase - g_data.exp_phases[i];
        
        while (diff > TMath::Pi() / 2.) diff -= TMath::Pi();
        while (diff < -TMath::Pi() / 2.) diff += TMath::Pi();
        
        chi2 += (diff * diff) / (g_data.errors[i] * g_data.errors[i]);
    }
    
    f = chi2;
}

void drawPotential(const Potential& pot, double chi2, int ndf, TFile* outfile) {
    
    gStyle->SetOptStat(0);
    gStyle->SetPalette(kViridis);
    
    TCanvas canvas("cPotential", "Nuclear Optical Potential", 1200, 800);
    gPad->SetLeftMargin(0.12);
    gPad->SetRightMargin(0.05);
    
    TF1 f_real("f_real", [&pot](double *x, double *p) {
        return pot.eval(x[0]);
    }, 0.1, 15.0, 0);
    f_real.SetLineColor(kRed);
    f_real.SetLineWidth(3);
    
    TF1 f_coulomb("f_coulomb", [](double *x, double *p) {
        return coulombPotential(x[0], constant::Z_PROTON, constant::Z_HE3);
    }, 0.1, 15.0, 0);
    f_coulomb.SetLineColor(kBlue);
    f_coulomb.SetLineWidth(3);
    f_coulomb.SetLineStyle(2);
    
    TF1 f_total("f_total", [&pot](double *x, double *p) {
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
    
    std::cout << "\nPlot saved as potential_fit.pdf and potential_fit.png" << std::endl;
}

void drawPhaseShiftComparison(const Potential& pot, double chi2, int ndf, TFile* outfile) {
    
    gStyle->SetOptStat(0);
    
    TCanvas canvas("canvas", "Phase Shift Comparison", 1200, 800);
    canvas.DrawFrame(4, -90, 12, -50, ";#it{E}_{lab} (MeV);#delta (deg.)");
    gPad->SetLeftMargin(0.12);
    gPad->SetRightMargin(0.05);
    
    const double rAsymptotic = 1.0; // fm
    const double mu = reducedMass(constant::M_PROTON, constant::M_HE3);
    PhaseShiftCoulomb solver(constant::M_PROTON, constant::M_HE3, constant::Z_PROTON, constant::Z_HE3);
    
    std::cout << "\nCalculated phases: [ ";
    std::vector<double> fitted_phases;
    for (size_t i = 0; i < g_data.l_values.size(); ++i) {
        const double v = TMath::Sqrt(2. * g_data.energies[i] / constant::M_PROTON); // incident proton beam velocity
        const double incidentMomentum = v * constant::M_PROTON;
        const double kstar = getKstar(incidentMomentum);
        double k = waveNumber(kstar);
        double calc_phase = solver.solveAndExtractPhase(rAsymptotic, g_data.l_values[i], k, *g_potential);

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

    g_data.l_values = {0, 0, 0, 0, 0, 0, 0, 0};
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
    double bestPars[nPars];

    const int nTrials = 50;
    for (int itrial = 0; itrial < nTrials; itrial++) {
        double trialPars[nPars];
        for (int iparam = 0; iparam < nPars; iparam++) {
            trialPars[iparam] = gRandom->Uniform(initialisationParameters[iparam].min, initialisationParameters[iparam].max);
        }
        Int_t npar_dummy = 3;
        Double_t gin_dummy[10];
        Double_t f_test;
        chi2Function(npar_dummy, gin_dummy, f_test, trialPars, 0);
                
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
            case 2: initPars[iparam] = initialisationParameter(80.0, 5.0, 10.0, 200.0);   break;
            case 3: initPars[iparam] = initialisationParameter(2.5, 0.1, 1.5, 4.0); break;
        }
    }

    std::cout << "n entries in initpars = " << initPars.size() << std::endl;
    
    TMinuit *minuit = new TMinuit(g_potential->getNPars());
    
    // Set print level (-1: quiet, 0: normal, 1: verbose)
    minuit->SetPrintLevel(1);
    
    // Define parameters: name, start value, step, min, max
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
    int ndf = g_data.l_values.size() - g_potential->getNPars();
    std::cout << "χ²    = " << amin << std::endl;
    std::cout << "ndf   = " << ndf << std::endl;
    std::cout << "χ²/ndf = " << amin/ndf << std::endl;
    std::cout << "==================================================" << std::endl;

    TFile * outfile = new TFile("output/potential_from_phase_shifts_doublegaus_pratt.root", "recreate");
    drawPotential(*g_potential, amin, ndf, outfile);
    drawPhaseShiftComparison(*g_potential, amin, ndf, outfile);
    outfile->Close();
    
    delete minuit;

}