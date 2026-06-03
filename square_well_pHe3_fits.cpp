#include "include/pHe3SquareWell.h"
#include <TCanvas.h>
#include <TMultiGraph.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TLegend.h>
#include <iomanip>
#include <TAxis.h>

void square_well_pHe3_fits() {
    
    std::cout << "=== p-He3 Square Well Potential Analysis ===" << std::endl;
    std::cout << "Fitting parameters to experimental phase shifts" << std::endl;
    std::cout << "Data from T.V. Daniels et al, PRC 82, 034002 (2010)" << std::endl;
    std::cout << std::endl;
    
    // Create the square well object
    pHe3SquareWell sqwell;
    
    // Experimental targets from T.V. Daniels et al, PRC (2010)
    std::vector<double> q_exp = {48.706, 57.630, 69.941, 76.496};
    std::vector<std::vector<double>> delta_exp = {
        {-39.1, -48.7, -56.3, -67.8}, // L=0, S=0
        {-34.5, -42.9, -49.3, -58.6}, // L=0, S=1
        {8.0, 13.4, 17.3, 21.2},      // L=1, S=0
        {15.4, 25.5, 34.1, 46.0}      // L=1, S=1
    };
    std::vector<std::vector<double>> delta_exp_error = {
        {1.7, 0.9, 0.6, 0.9}, // L=0, S=0
        {0.7, 0.09, 0.5, 0.3}, // L=0, S=1
        {2, 0.4, 1.6, 1.7},      // L=1, S=0
        {6, 0.8, 0.9, 0.7}      // L=1, S=1
    };

    sqwell.LoadExperimentalData(q_exp, delta_exp, delta_exp_error);
    
    std::vector<std::string> channel_names = {
        "L=0, S=0 (singlet, s-wave)", 
        "L=0, S=1 (triplet, s-wave)", 
        "L=1, S=0 (singlet, p-wave)", 
        "L=1, S=1 (triplet, p-wave)"
    };
    
    // Fit all channels
    std::cout << "========================================" << std::endl;
    std::cout << "FITTING ALL CHANNELS" << std::endl;
    std::cout << "========================================" << std::endl;
    
    for (int channel = 0; channel < 4; channel++) {
        std::cout << "\n>>> Fitting Channel " << channel << ": " 
                  << channel_names[channel] << std::endl;
        std::cout << "Target phase shifts (deg):";
        for (size_t i = 0; i < q_exp.size(); i++) {
            std::cout << " " << std::fixed << std::setprecision(1) << delta_exp[channel][i] << "±" << std::setprecision(1) << delta_exp_error[channel][i] << ",";
        }
        std::cout << std::endl;
        std::cout << "Starting fit with 1000 iterations..." << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        sqwell.FitToPhaseShifts(channel, 1000);
        
        std::cout << std::string(60, '-') << std::endl;
        std::cout << "✓ Fit completed for channel " << channel << std::endl;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "FINAL FITTED PARAMETERS" << std::endl;
    std::cout << "========================================" << std::endl;
    
    for (int channel = 0; channel < 4; channel++) {
        std::vector<double> a_vals, V_vals;
        sqwell.GetParameters(channel, a_vals, V_vals);
        
        std::cout << "\nChannel " << channel << " (" << channel_names[channel] << "):" << std::endl;
        std::cout << "  Radii (fm):  a = {";
        for (size_t i = 0; i < a_vals.size(); i++) {
            std::cout << std::fixed << std::setprecision(6) << a_vals[i];
            if (i < a_vals.size() - 1) std::cout << ", ";
        }
        std::cout << "}" << std::endl;
        
        std::cout << "  Depths (MeV): V = {";
        for (size_t i = 0; i < V_vals.size(); i++) {
            std::cout << std::fixed << std::setprecision(4) << V_vals[i];
            if (i < V_vals.size() - 1) std::cout << ", ";
        }
        std::cout << "}" << std::endl;
    }
    
    // Detailed comparison with experimental data
    std::cout << "\n========================================" << std::endl;
    std::cout << "COMPARISON WITH EXPERIMENTAL DATA" << std::endl;
    std::cout << "========================================" << std::endl;
    
    for (int channel = 0; channel < 4; channel++) {
        std::cout << "\n" << channel_names[channel] << ":" << std::endl;
        std::cout << std::string(90, '-') << std::endl;
        std::cout << "  q (MeV/c)  | δ_exp (deg) | δ_err (deg) | δ_calc (deg) | Difference | % Error "<< std::endl;
        std::cout << std::string(90, '-') << std::endl;
        
        double total_error = 0.0;
        double chi2 = 0.0;
        for (size_t i = 0; i < q_exp.size(); i++) {
            double delta_calc = sqwell.GetPhaseShift(channel, q_exp[i]) * 180.0 / Constants::PI;
            double diff = delta_calc - delta_exp[channel][i];
            double percent_error = 100.0 * std::abs(diff) / std::abs(delta_exp[channel][i]);
            total_error += diff * diff;
            chi2 += diff * diff / (delta_exp_error[channel][i] * delta_exp_error[channel][i]);
            
            std::cout << "  " << std::fixed << std::setprecision(2) << std::setw(9) << q_exp[i] 
                      << "  | " << std::setw(11) << delta_exp[channel][i]
                      << "  | " << std::setw(6) << delta_exp_error[channel][i]
                      << " | " << std::setw(12) << delta_calc
                      << " | " << std::setw(10) << diff
                      << " | " << std::setw(7) << std::setprecision(2) << percent_error << "%"
                      << std::endl;
        }
        std::cout << std::string(90, '-') << std::endl;
        std::cout << "  χ² = " << std::scientific << std::setprecision(3) << chi2 << std::endl;
    }
    
    // Save phase shifts to file
    std::cout << "\n========================================" << std::endl;
    std::cout << "Saving complete phase shift data to 'pHe3_phase_shifts_fitted.dat'..." << std::endl;
    sqwell.PrintPhaseShifts("output/pHe3_phase_shifts_fitted.dat");
    std::cout << "✓ Data saved" << std::endl;
    
    // Create comprehensive plot with experimental data
    std::cout << "\n========================================" << std::endl;
    std::cout << "GENERATING PLOTS" << std::endl;
    std::cout << "========================================" << std::endl;

    TCanvas* cIndivisuals = new TCanvas("cIndivisuals", "p-^{3}He Phase Shifts - Individual Channels", 1200, 900);
    cIndivisuals->Print("output/pHe3_phase_shifts_individual_channels.pdf["); // Open multi-page PDF
    
    TCanvas* c1 = new TCanvas("c1", "p-^{3}He Phase Shifts", 1200, 900);
    c1->Divide(2, 2);
    
    int colors[4] = {kBlue, kRed, kGreen+2, kMagenta};
    
    for (int channel = 0; channel < 4; channel++) {
        c1->cd(channel + 1);
        
        // Theory (smooth curve)
        TGraph* gr_theory = new TGraph();
        for (int iq = 0; iq < 100; iq++) {
            double q = iq * 1.0;
            double phase = sqwell.GetPhaseShift(channel, q) * 180.0 / Constants::PI;
            gr_theory->SetPoint(iq, q, phase);
        }
        gr_theory->SetLineColor(colors[channel]);
        gr_theory->SetLineWidth(3);
        gr_theory->SetTitle(channel_names[channel].c_str());
        gr_theory->GetXaxis()->SetTitle("#it{k}* (MeV/#it{c})");
        gr_theory->GetYaxis()->SetTitle("#delta (degrees)");
        gr_theory->Draw("AL");
        
        // Experimental points
        TGraphErrors* gr_exp = new TGraphErrors();
        for (size_t i = 0; i < q_exp.size(); i++) {
            gr_exp->SetPoint(i, q_exp[i], delta_exp[channel][i]);
            gr_exp->SetPointError(i, 0, delta_exp_error[channel][i]);
        }
        gr_exp->SetMarkerStyle(20);
        gr_exp->SetMarkerSize(1.5);
        gr_exp->SetMarkerColor(kBlack);
        gr_exp->SetLineColor(kBlack);
        gr_exp->Draw("P SAME");
        
        // Legend
        std::array<double, 4> leg_pos = {0.15, 0.70, 0.45, 0.85};
        if (channel == 0 | channel == 1) {
            leg_pos[0] = 0.55;  leg_pos[2] = 0.85;
            leg_pos[1] = 0.70;  leg_pos[4] = 0.85;
        }
        TLegend* leg = new TLegend(leg_pos[0], leg_pos[1], leg_pos[2], leg_pos[3]);
        leg->AddEntry(gr_theory, "Fit", "l");
        leg->AddEntry(gr_exp, "Daniels et al. (2010)", "p");
        leg->SetBorderSize(0);
        leg->SetFillStyle(0);
        leg->Draw();


        cIndivisuals->cd();
        gr_theory->Draw("AL");
        gr_exp->Draw("P SAME");
        leg->Draw();
        cIndivisuals->Print("output/pHe3_phase_shifts_individual_channels.pdf"); // Add page to multi-page PDF
        cIndivisuals->Clear();
    }
    
    c1->Update();
    c1->Print("output/pHe3_phase_shifts_comparison.pdf");
    std::cout << "✓ Plots saved as PDF and PNG" << std::endl;
    cIndivisuals->Print("output/pHe3_phase_shifts_individual_channels.pdf]"); // Close multi-page PDF
    
    // Also create a combined plot on one canvas
    TCanvas* c2 = new TCanvas("c2", "p-He3 All Phase Shifts", 1000, 700);
    TMultiGraph* mg = new TMultiGraph();
    TLegend* leg2 = new TLegend(0.15, 0.15, 0.38, 0.40);
    leg2->SetBorderSize(0);
    leg2->SetFillStyle(1001);
    
    for (int channel = 0; channel < 4; channel++) {
        // Theory curves
        TGraph* gr_theory = new TGraph();
        for (int iq = 0; iq < 100; iq++) {
            double q = iq * 1.0;
            double phase = sqwell.GetPhaseShift(channel, q) * 180.0 / Constants::PI;
            gr_theory->SetPoint(iq, q, phase);
        }
        gr_theory->SetLineColor(colors[channel]);
        gr_theory->SetLineWidth(3);
        mg->Add(gr_theory, "L");
        
        // Experimental points
        TGraphErrors* gr_exp = new TGraphErrors();
        for (size_t i = 0; i < q_exp.size(); i++) {
            gr_exp->SetPoint(i, q_exp[i], delta_exp[channel][i]);
            gr_exp->SetPointError(i, 0, delta_exp_error[channel][i]);
        }
        gr_exp->SetMarkerStyle(20);
        gr_exp->SetMarkerSize(1.2);
        gr_exp->SetMarkerColor(colors[channel]);
        gr_exp->SetLineColor(colors[channel]);
        mg->Add(gr_exp, "P");
        
        leg2->AddEntry(gr_theory, channel_names[channel].c_str(), "lp");
    }
    
    mg->Draw("A");
    mg->SetTitle("p-He3 Phase Shifts: Fitted Theory vs Experimental Data; #it{k}* (MeV/#it{c});#delta (degrees)");
    leg2->Draw();
    
    c2->Update();
    c2->SaveAs("output/pHe3_all_phase_shifts.pdf");
    std::cout << "✓ Combined plot saved" << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "ANALYSIS COMPLETE" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nGenerated files:" << std::endl;
    std::cout << "  - pHe3_phase_shifts_fitted.dat (numerical data)" << std::endl;
    std::cout << "  - output/pHe3_phase_shifts_comparison.pdf/png (4-panel plot)" << std::endl;
    std::cout << "  - output/pHe3_all_phase_shifts.pdf/png (combined plot)" << std::endl;
    std::cout << "\nClose the plot windows to exit." << std::endl;
}