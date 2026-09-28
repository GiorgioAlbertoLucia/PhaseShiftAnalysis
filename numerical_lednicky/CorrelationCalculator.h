#pragma once

#include "Config.h"
#include "CoulombWaveFunction.h"

#include <TGraph.h>
#include <TH2D.h>

#include <map>
#include <tuple>
#include <string>

class CorrelationCalculator {
public:
    explicit CorrelationCalculator(const CalculationConfig& config);
    ~CorrelationCalculator();   // new: clean up owned histograms

    void GenerateDataFiles();
    void GenerateRootFile();

private:
    CalculationConfig config_;
    CoulombWaveFunction waveFunction_;

    using DataKey = std::tuple<double, double, double>; // aRe, aIm, r
    std::map<DataKey, TH2D*> dataHists_;

    void GenerateData(double kValue, double aRe, double aIm, double r);
    TGraph* CalculateCorrelationFunction(double sourceSize, double aRe, double aIm, double r);
    std::string GetDataFileName(double kValue, double aRe, double aIm, double r) const;

    // ROOT TH2 intermediate storage helpers
    TH2D* GetOrCreateHist(double aRe, double aIm, double r);
    std::string GetIntermediateRootFileName() const;
    void LoadIntermediateHistograms();
};
