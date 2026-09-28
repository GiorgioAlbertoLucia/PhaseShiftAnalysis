#include "CorrelationCalculator.h"

#include <TFile.h>
#include <TGraph.h>
#include <TH2D.h>

#include <atomic>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {
    void PrintProgress(int current, int total) {
        const int barWidth = 40;
        double frac = static_cast<double>(current) / total;
        int pos = static_cast<int>(barWidth * frac);

        std::cout << "\r[";
        for (int i = 0; i < barWidth; i++) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << std::setw(3) << static_cast<int>(frac * 100.0) << "% "
             << "(" << current << "/" << total << ")" << std::flush;

        if (current == total) std::cout << std::endl;
    }
}

CorrelationCalculator::CorrelationCalculator(const CalculationConfig& config)
    : config_(config), 
      waveFunction_(config.GetChargeRadius()) {
}
CorrelationCalculator::~CorrelationCalculator() {
    for (auto& [key, hist] : dataHists_) {
        delete hist;
    }
}

TH2D* CorrelationCalculator::GetOrCreateHist(double aRe, double aIm, double r) {
    DataKey key{aRe, aIm, r};
    auto it = dataHists_.find(key);
    if (it != dataHists_.end()) return it->second;

    int nk = config_.GetNumKBins(), nr = config_.GetNumRBins();
    TH2D* h = new TH2D(Form("hCky_aRe%.2f_aIm%.2f_r%.2f", aRe, aIm, r), "",
                        nk, config_.kMin, config_.kMin + nk * config_.kBinWidth,
                        nr, config_.rMin, config_.rMin + nr * config_.rBinWidth);
    h->SetDirectory(0); // detach from any open TFile so we control writing explicitly
    dataHists_[key] = h;
    return h;
}

std::string CorrelationCalculator::GetIntermediateRootFileName() const {
    return config_.outputFolder + config_.intermediateRootFile;
}

void CorrelationCalculator::LoadIntermediateHistograms() {
    if (!dataHists_.empty()) return; // already populated in this run

    std::string filename = GetIntermediateRootFileName();
    TFile fin(filename.c_str(), "READ");
    if (fin.IsZombie()) {
        std::cerr << "Error: could not open intermediate ROOT file " << filename << std::endl;
        return;
    }

    const int nvalues = config_.realScatteringLengths.size();
    for (int i = 0; i < nvalues; i++) {
        double aRe = config_.realScatteringLengths[i];
        double aIm = config_.imagScatteringLengths[i];
        double r = config_.effectiveRanges.empty() ? 0. : config_.effectiveRanges[i];

        TH2D* h = (TH2D*)fin.Get(Form("hCky_aRe%.2f_aIm%.2f_r%.2f", aRe, aIm, r));
        if (h) { h->SetDirectory(0); dataHists_[{aRe, aIm, r}] = h; }
    }
    fin.Close();
}

void CorrelationCalculator::GenerateDataFiles() {
    int nkbins = config_.GetNumKBins();
    const int nvalues = config_.realScatteringLengths.size();
    int totalFiles = nkbins * nvalues;
    
    std::cout << "Generating data files..." << std::endl;
    std::cout << "k bins: " << nkbins << std::endl;
    std::cout << "Real scattering lengths: " << config_.realScatteringLengths.size() << std::endl;
    std::cout << "Imag scattering lengths: " << config_.imagScatteringLengths.size() << std::endl;
    
    //int doneFiles = 0;
    std::atomic<int> doneFiles{0};

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int kBin = 0; kBin < nkbins; kBin++) {
        for (int i = 0; i < nvalues; i++) {
            double kValue = config_.kMin + (kBin * config_.kBinWidth);
            double aRe = config_.realScatteringLengths[i];
            double aIm = config_.imagScatteringLengths[i];
            double r = config_.effectiveRanges.empty() ? 0. : config_.effectiveRanges[i];

            GenerateData(kValue, aRe, aIm, r);

            int done = ++doneFiles;
            #pragma omp critical
            PrintProgress(done, totalFiles);
        }
    }

    if (!config_.useDatFiles) {
        std::string filename = GetIntermediateRootFileName();
        std::cout << "Writing intermediate ROOT file: " << filename << std::endl;
        TFile fout(filename.c_str(), "RECREATE");
        for (auto& [key, hist] : dataHists_) hist->Write();
        fout.Close();
    }
    
    std::cout << "Data file generation complete!" << std::endl;
}

void CorrelationCalculator::GenerateData(double kValue, double aRe, double aIm, double r) {
    if (config_.useDatFiles) {
        std::string filename = GetDataFileName(kValue, aRe, aIm, r);
        std::cout << "Generating data file: " << filename << std::endl;
        std::ofstream outfile(filename);
        
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }
        
        std::complex<double> scatteringLength(aRe, aIm);
        int nrbins = config_.GetNumRBins();
        
        for (int rBin = 0; rBin < nrbins; rBin++) {
            double rValue = config_.rMin + (rBin * config_.rBinWidth);
            double ckValue = waveFunction_.CalculateDCky(kValue, rValue, scatteringLength, r, config_.integrationSteps);
                
         outfile << std::fixed << std::setprecision(3) << rValue << "\t" 
         << std::scientific << std::setprecision(4) << ckValue << std::endl;
        }
        
        outfile.close();
    } else {
        TH2D* h = GetOrCreateHist(aRe, aIm, r); // h stored in dataHists_ inside GetOrCreateHist
        int kBin = static_cast<int>(std::round((kValue - config_.kMin) / config_.kBinWidth)) + 1;
        std::complex<double> scatteringLength(aRe, aIm);
        int nrbins = config_.GetNumRBins();

        for (int rBin = 0; rBin < nrbins; rBin++) {
            double rValue = config_.rMin + (rBin * config_.rBinWidth);
            double ckValue = waveFunction_.CalculateDCky(kValue, rValue, scatteringLength, r,
                                                        config_.integrationSteps);
            h->SetBinContent(kBin, rBin + 1, ckValue);
        }
    }
}

std::string CorrelationCalculator::GetDataFileName(double kValue, double aRe, double aIm, double r) const {
    std::stringstream ss;
    ss << config_.outputFolder << "Cky_k" 
       << std::fixed << std::setprecision(0) << kValue 
       << "_aRe" << std::setprecision(1) << aRe 
       << "_aIm" << std::setprecision(1) << aIm 
       << "_r" << std::setprecision(1) << r 
       << ".dat";
    return ss.str();
}

TGraph* CorrelationCalculator::CalculateCorrelationFunction(
    double sourceSize, double aRe, double aIm, double r) {
    
    const int nSamples = 200; // Integration samples for r
    const double h = 0.2;     // Step size

    char nameBuf[128], titleBuf[256];
    snprintf(nameBuf, sizeof(nameBuf), "g%.2f_aRe%.2f_aIm%.2f_r%.2f", sourceSize, aRe, aIm, r);
    snprintf(titleBuf, sizeof(titleBuf),
             "Correlation Function (R=%.2f fm, aRe=%.2f fm, aIm=%.2f fm, r=%.2f fm);#it{k} (MeV/#it{c});C(#it{k})",
             sourceSize, aRe, aIm, r);

    TGraph* graph = new TGraph();
    graph->SetName(nameBuf);
    graph->SetTitle(titleBuf);

    int nkbins = config_.GetNumKBins();
    TH2D* hist = nullptr;
    if (!config_.useDatFiles) {
        auto it = dataHists_.find({aRe, aIm, r});
        hist = (it != dataHists_.end()) ? it->second : nullptr;
    }

    for (int kBin = 0; kBin < nkbins; kBin++) {
        double kValue = config_.kMin + (kBin * config_.kBinWidth);
        std::vector<double> ckValues;
        
        if (config_.useDatFiles) {

            // Read data from file
            std::string filename = GetDataFileName(kValue, aRe, aIm, r);
            std::ifstream infile(filename);
            
            if (!infile.is_open()) {
                std::cerr << "Warning: Could not open " << filename << std::endl;
                continue;
            }
            
            std::string line;
            while (std::getline(infile, line) && ckValues.size() <= nSamples) {
                double rFile, ck;
                if (std::sscanf(line.c_str(), "%lf %lf", &rFile, &ck) == 2) {
                    // Apply source size weighting
                    double weight = 1.0 / std::pow(4.0 * PhysicalConstants::PI * 
                        sourceSize * sourceSize * 0.00506773123 * 
                        0.00506773123, 1.5) *
                        std::exp(-r * r / (4.0 * sourceSize * sourceSize));
                    ckValues.push_back(ck * weight);
                }
            }
            infile.close();
        } else {
            if (!hist) { std::cerr << "Warning: no histogram for aRe=" << aRe << " aIm=" << aIm << std::endl; continue; }
            int histKBin = static_cast<int>(std::round((kValue - config_.kMin) / config_.kBinWidth)) + 1;
            for (int rBin = 1; rBin <= hist->GetNbinsY() && (int)ckValues.size() <= nSamples; rBin++) {
                double rValue = config_.rMin + (rBin - 1) * config_.rBinWidth;
                double weight = 1.0 / std::pow(4.0 * PhysicalConstants::PI *
                                        sourceSize * sourceSize * 0.00506773123 *
                                        0.00506773123, 1.5) *
                               std::exp(-rValue * rValue / (4.0 * sourceSize * sourceSize));
                ckValues.push_back(hist->GetBinContent(histKBin, rBin) * weight);
            }
        }
        
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
    if (!config_.useDatFiles) LoadIntermediateHistograms();

    std::cout << "Generating ROOT file: " << config_.outputRootFile << std::endl;

    const int nvalues = config_.realScatteringLengths.size();
    const int nSizes = config_.sourceSizes.size();
    const int nTasks = nvalues * nSizes;

    // Compute graphs in parallel; ROOT I/O (TFile/Write) must stay serial.
    std::vector<TGraph*> graphs(nTasks, nullptr);

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < nvalues; i++) {
        for (int j = 0; j < nSizes; j++) {
            double aRe = config_.realScatteringLengths[i];
            double aIm = config_.imagScatteringLengths[i];
            double r = config_.effectiveRanges.empty() ? 0. : config_.effectiveRanges[i];
            double sourceSize = config_.sourceSizes[j];

            graphs[i * nSizes + j] = CalculateCorrelationFunction(sourceSize, aRe, aIm, r);
        }
    }

    // Serial write
    TFile* fout = new TFile((config_.outputRootFolder + config_.outputRootFile).c_str(), "RECREATE");
    for (TGraph* graph : graphs) {
        if (!graph) continue;
        std::cout << "Writing graph: " << graph->GetName() << std::endl;
        graph->Write();
        delete graph;
    }
    fout->Close();
    delete fout;

    std::cout << "ROOT file generation complete!" << std::endl;
}
