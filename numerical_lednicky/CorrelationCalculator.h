#pragma once

#include <string>
#include <complex>
#include "Config.h"
#include "CoulombWaveFunction.h"

class TGraph;

class CorrelationCalculator {
public:
    explicit CorrelationCalculator(const CalculationConfig& config);
    
    // Generate all correlation function data files
    void GenerateDataFiles();
    
    // Generate ROOT file with correlation function graphs
    void GenerateRootFile();
    
private:
    CalculationConfig config_;
    CoulombWaveFunction waveFunction_;
    
    // Generate single data file for given parameters
    void GenerateDataFile(double kValue, double aRe, double aIm);
    
    // Calculate correlation function TGraph for given source size and scattering parameters
    TGraph* CalculateCorrelationFunction(double sourceSize, double aRe, double aIm);
    
    // Get filename for data output
    std::string GetDataFileName(double kValue, double aRe, double aIm) const;
};
