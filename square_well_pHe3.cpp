#include <iostream>
#include <string>
#include <vector>

#include "include/pHe3SquareWell.h"

void square_well_pHe3() {
    
    std::cout << "=== p-He3 Square Well Potential Analysis ===" << std::endl;
    std::cout << std::endl;
    
    // Create the square well object
    pHe3SquareWell sqwell;
    
    // Initialize: compute Coulomb factors and phase shifts
    std::cout << "Initializing with Coulomb wavefunctions and computing phase shifts..." << std::endl;
    sqwell.Initialize();
    
    // Print current parameters
    std::cout << "\nCurrent potential parameters:" << std::endl;
    for (int ch = 0; ch < 4; ch++) {
        std::vector<double> a_vals, V_vals;
        sqwell.GetParameters(ch, a_vals, V_vals);
        
        std::string channel_name;
        if (ch == 0) channel_name = "L=0, S=0 (singlet)";
        else if (ch == 1) channel_name = "L=0, S=1 (triplet)";
        else if (ch == 2) channel_name = "L=1, S=0 (singlet)";
        else channel_name = "L=1, S=1 (triplet)";
        
        std::cout << "Channel " << ch << " (" << channel_name << "):" << std::endl;
        std::cout << "  Radii (fm): {" << a_vals[0] << ", " << a_vals[1] << "}" << std::endl;
        std::cout << "  Depths (MeV): {" << V_vals[0] << ", " << V_vals[1] << "}" << std::endl;
    }
    
    // Save phase shifts to file
    std::cout << "\nSaving phase shifts to 'pHe3_phase_shifts.dat'..." << std::endl;
    sqwell.PrintPhaseShifts("pHe3_phase_shifts.dat");
    
    // Compare with experimental data at specific momenta
    std::cout << "\nComparison with experimental data:" << std::endl;
    std::cout << "Data from T.V. Daniels et al, PRC 82, 034002 (2010)" << std::endl;
    
    std::vector<double> q_exp = {48.706, 57.630, 69.941, 76.496};
    std::vector<std::vector<double>> delta_exp = {
        {-39.1, -48.7, -56.3, -67.8}, // L=0, S=0
        {-34.5, -42.9, -49.3, -58.6}, // L=0, S=1
        {8.0, 13.4, 17.3, 21.2},      // L=1, S=0
        {15.4, 25.5, 34.1, 46.0}      // L=1, S=1
    };
    
    for (int ch = 0; ch < 4; ch++) {
        std::string channel_name;
        if (ch == 0) channel_name = "L=0, S=0";
        else if (ch == 1) channel_name = "L=0, S=1";
        else if (ch == 2) channel_name = "L=1, S=0";
        else channel_name = "L=1, S=1";
        
        std::cout << "\nChannel " << ch << " (" << channel_name << "):" << std::endl;
        std::cout << "q (MeV/c) | δ_exp (deg) | δ_calc (deg) | Difference" << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;
        
        for (size_t i = 0; i < q_exp.size(); i++) {
            double delta_calc = sqwell.GetPhaseShift(ch, q_exp[i]) * 180.0 / Constants::PI;
            double diff = delta_calc - delta_exp[ch][i];
            printf("%8.2f  | %11.2f | %12.2f | %10.2f\n", 
                   q_exp[i], delta_exp[ch][i], delta_calc, diff);
        }
    }
    
    // Plot phase shifts for all channels
    std::cout << "\nPlotting phase shifts..." << std::endl;
    sqwell.PlotPhaseShifts(-1, "output/pHe3SquareWell.pdf"); // -1 means all channels
    
    const int channel = 0;
    std::cout << "\nFitting channel " << channel << " (L=0, S=0)..." << std::endl;
    sqwell.FitToPhaseShifts(channel, 10000);
    
    std::vector<double> a_fitted, V_fitted;
    sqwell.GetParameters(channel, a_fitted, V_fitted);
    std::cout << "Fitted parameters for channel " << channel << ":" << std::endl;
    std::cout << "  a = {" << a_fitted[0] << ", " << a_fitted[1] << "} fm" << std::endl;
    std::cout << "  V = {" << V_fitted[0] << ", " << V_fitted[1] << "} MeV" << std::endl;
    
    std::cout << "\nAnalysis complete! Close the plot window to exit." << std::endl;
}