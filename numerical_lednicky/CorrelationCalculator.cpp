#include "CorrelationCalculator.h"
#include <TGraph.h>
#include <TFile.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <cmath>

using namespace std;

CorrelationCalculator::CorrelationCalculator(const CalculationConfig& config)
    : config_(config), 
      waveFunction_(config.GetChargeRadius(), config.effectiveRange) {
}

void CorrelationCalculator::GenerateDataFiles() {
    int nkbins = config_.GetNumKBins();
    
    cout << "Generating data files..." << endl;
    cout << "k bins: " << nkbins << endl;
    cout << "Real scattering lengths: " << config_.realScatteringLengths.size() << endl;
    cout << "Imag scattering lengths: " << config_.imagScatteringLengths.size() << endl;
    
    for (int kBin = 0; kBin < nkbins; kBin++) {
        double kValue = config_.kMin + (kBin * config_.kBinWidth);
        
        for (double aRe : config_.realScatteringLengths) {
            for (double aIm : config_.imagScatteringLengths) {
                GenerateDataFile(kValue, aRe, aIm);
            }
        }
    }
    
    cout << "Data file generation complete!" << endl;
}

void CorrelationCalculator::GenerateDataFile(double kValue, double aRe, double aIm) {
    string filename = GetDataFileName(kValue, aRe, aIm);
    ofstream outfile(filename);
    
    if (!outfile.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return;
    }
    
    complex<double> scatteringLength(aRe, aIm);
    int nrbins = config_.GetNumRBins();
    
    for (int rBin = 0; rBin < nrbins; rBin++) {
        double rValue = config_.rMin + (rBin * config_.rBinWidth);
        double ckValue = waveFunction_.CalculateDCky(kValue, rValue, scatteringLength,
                                                     config_.integrationSteps);
        
        outfile << fixed << setprecision(3) << rValue << "\t" 
                << scientific << setprecision(4) << ckValue << endl;
    }
    
    outfile.close();
}

string CorrelationCalculator::GetDataFileName(double kValue, double aRe, double aIm) const {
    stringstream ss;
    ss << config_.outputFolder << "Cky_k" 
       << fixed << setprecision(0) << kValue 
       << "_aRe" << setprecision(1) << aRe 
       << "_aIm" << setprecision(1) << aIm << ".dat";
    return ss.str();
}

TGraph* CorrelationCalculator::CalculateCorrelationFunction(
    double sourceSize, double aRe, double aIm) {
    
    const int nSamples = 200; // Integration samples for r
    const double h = 0.2;     // Step size
    
    TGraph* graph = new TGraph();
    graph->SetName(Form("g%.2f_aRe%.1f_aIm%.1f", sourceSize, aRe, aIm));
    graph->SetTitle(Form("Correlation Function (R=%.2f fm, aRe=%.1f fm, aIm=%.1f fm);#it{k} (MeV/#it{c});C(#it{k})", 
                             sourceSize, aRe, aIm));
    
    int nkbins = config_.GetNumKBins();
    
    for (int kBin = 0; kBin < nkbins; kBin++) {
        double kValue = config_.kMin + (kBin * config_.kBinWidth);
        
        // Read data from file
        string filename = GetDataFileName(kValue, aRe, aIm);
        ifstream infile(filename);
        
        if (!infile.is_open()) {
            cerr << "Warning: Could not open " << filename << endl;
            continue;
        }
        
        vector<double> ckValues;
        string line;
        while (getline(infile, line) && ckValues.size() <= nSamples) {
            double r, ck;
            if (sscanf(line.c_str(), "%lf %lf", &r, &ck) == 2) {
                // Apply source size weighting
                double weight = 1.0 / pow(4.0 * PhysicalConstants::PI * 
                                        sourceSize * sourceSize * 0.00506773123 * 
                                        0.00506773123, 1.5) *
                               exp(-r * r / (4.0 * sourceSize * sourceSize));
                ckValues.push_back(ck * weight);
            }
        }
        infile.close();
        
        // Integrate using Simpson's rule
        if (ckValues.size() > nSamples) {
            double sum = 0.0;
            for (int i = 1; i < nSamples; i++) {
                sum += h * ckValues[i];
            }
            double integral = h / 2.0 * (ckValues[0] + ckValues[nSamples]) + sum;
            graph->SetPoint(kBin, kValue, integral);
        }
    }
    
    return graph;
}

void CorrelationCalculator::GenerateRootFile() {
    cout << "Generating ROOT file: " << config_.outputRootFile << endl;
    
    TFile* fout = new TFile((config_.outputRootFolder + config_.outputRootFile).c_str(), "RECREATE");
    
    for (double aRe : config_.realScatteringLengths) {
        for (double aIm : config_.imagScatteringLengths) {
            for (double sourceSize : config_.sourceSizes) {
                TGraph* graph = CalculateCorrelationFunction(sourceSize, aRe, aIm);
                cout << "Writing graph: " << graph->GetName() << endl;
                graph->Write();
                delete graph;
            }
        }
    }
    
    fout->Close();
    delete fout;
    
    cout << "ROOT file generation complete!" << endl;
}
