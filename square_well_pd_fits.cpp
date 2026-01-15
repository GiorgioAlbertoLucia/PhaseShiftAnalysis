#include "include/pdSquareWell.h"
#include <TCanvas.h>
#include <TMultiGraph.h>
#include <TGraph.h>
#include <TLegend.h>
#include <TLatex.h>
#include <iomanip>
#include <TAxis.h>

void square_well_pd_fits() {
    
    std::cout << "=== p-d Square Well Potential Analysis ===" << std::endl;
    std::cout << "Fitting parameters to target phase shifts" << std::endl;
    std::cout << "Based on tune_pd_s12.cc and tune_pd_s32.cc" << std::endl;
    std::cout << std::endl;
    
    // Create the square well object
    pdSquareWell sqwell;
    
    std::vector<double> Ep_targets_s12 = {1.0, 2.0, 3.0}, q_targets_s12(3);
    for (int i = 0; i < Ep_targets_s12.size(); i++) {
        double Ep = Ep_targets_s12[i];
        double q = computeKstarFromEprojectile(Ep, Constants::M_PROTON, Constants::M_DEUTERON);
        q_targets_s12[i] = q;
    }
    std::vector<double> delta_targets_s12 = {-4.5, -20.0, -27.5};  // degrees
    const double scatt_target_s12 = -0.13;  // fm
    
    std::vector<double> Ep_targets_s32 = {1.0, 2.0, 3.0}, q_targets_s32(3);
    for (int i = 0; i < Ep_targets_s32.size(); i++) {
        double Ep = Ep_targets_s32[i];
        double q = computeKstarFromEprojectile(Ep, Constants::M_PROTON, Constants::M_DEUTERON);
        q_targets_s32[i] = q;
    }
    std::vector<double> delta_targets_s32 = {-37.5, -52.5, -64.0};  // degrees
    const double scatt_target_s32 = 0.0;  // Not used in tune_pd_s32.cc
    
    std::vector<std::string> channel_names = {
        "s=1/2 (doublet)", 
        "s=3/2 (quartet)"
    };
    
    // Fit all channels
    std::cout << "========================================" << std::endl;
    std::cout << "FITTING ALL CHANNELS" << std::endl;
    std::cout << "========================================" << std::endl;
    
    for (int channel = 0; channel < 2; channel++) {
        std::cout << "\n>>> Fitting Channel " << channel << ": " 
                  << channel_names[channel] << std::endl;
        
        if (channel == 0) {
            std::cout << "Target scattering length: " << scatt_target_s12 << " fm" << std::endl;
            std::cout << "Target phase shifts (deg) at q = ";
            for (size_t i = 0; i < q_targets_s12.size(); i++) {
                std::cout << q_targets_s12[i];
                if (i < q_targets_s12.size() - 1) std::cout << ", ";
            }
            std::cout << " MeV/c: ";
            for (size_t i = 0; i < delta_targets_s12.size(); i++) {
                std::cout << delta_targets_s12[i];
                if (i < delta_targets_s12.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
            std::cout << "Binding energy constraint: " << 8.481 << " MeV (triton)" << std::endl;
        }
        else {
            std::cout << "Target phase shifts (deg) at q = ";
            for (size_t i = 0; i < q_targets_s32.size(); i++) {
                std::cout << q_targets_s32[i];
                if (i < q_targets_s32.size() - 1) std::cout << ", ";
            }
            std::cout << " MeV/c: ";
            for (size_t i = 0; i < delta_targets_s32.size(); i++) {
                std::cout << delta_targets_s32[i];
                if (i < delta_targets_s32.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "Starting fit with 10000 iterations..." << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        sqwell.FitToPhaseShifts(channel, 200000, 0);
        
        std::cout << std::string(60, '-') << std::endl;
        std::cout << "✓ Fit completed for channel " << channel << std::endl;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "FINAL FITTED PARAMETERS" << std::endl;
    std::cout << "========================================" << std::endl;
    
    for (int channel = 0; channel < 2; channel++) {
        std::vector<double> a_vals, V_vals;
        sqwell.GetParameters(channel, a_vals, V_vals);
        
        std::cout << "\nChannel " << channel << " (" << channel_names[channel] << "):" << std::endl;
        std::cout << "  Number of wells: " << a_vals.size() << std::endl;
        std::cout << "  Radii (fm):  a = {";
        for (size_t i = 0; i < a_vals.size(); i++) {
            std::cout << std::fixed << std::setprecision(5) << a_vals[i];
            if (i < a_vals.size() - 1) std::cout << ", ";
        }
        std::cout << "}" << std::endl;
        
        std::cout << "  Depths (MeV): V = {";
        for (size_t i = 0; i < V_vals.size(); i++) {
            std::cout << std::fixed << std::setprecision(3) << V_vals[i];
            if (i < V_vals.size() - 1) std::cout << ", ";
        }
        std::cout << "}" << std::endl;
        
        std::cout << "  Scattering length: " << std::fixed << std::setprecision(4) 
                  << sqwell.GetScatteringLength(channel) << " fm" << std::endl;
    }
    
    // Detailed comparison with target data
    std::cout << "\n========================================" << std::endl;
    std::cout << "COMPARISON WITH TARGET PHASE SHIFTS" << std::endl;
    std::cout << "========================================" << std::endl;
    
    for (int channel = 0; channel < 2; channel++) {
        std::cout << "\n" << channel_names[channel] << ":" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        if (channel == 0) {
            std::cout << "  Scattering length:" << std::endl;
            std::cout << "    Target: " << std::fixed << std::setprecision(3) << scatt_target_s12 
                      << " fm, Calculated: " << sqwell.GetScatteringLength(channel) << " fm" << std::endl;
            std::cout << std::endl;
        }
        
        std::cout << "  q (MeV/c)  | δ_target (deg) | δ_calc (deg) | Difference | % Error" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        double total_error = 0.0;
        const auto& q_targets = (channel == 0) ? q_targets_s12 : q_targets_s32;
        const auto& delta_targets = (channel == 0) ? delta_targets_s12 : delta_targets_s32;
        
        for (size_t i = 0; i < q_targets.size(); i++) {
            double delta_calc = sqwell.GetPhaseShift(channel, q_targets[i]) * 180.0 / Constants::PI;
            double diff = delta_calc - delta_targets[i];
            double percent_error = 100.0 * std::abs(diff) / std::abs(delta_targets[i]);
            total_error += diff * diff;
            
            std::cout << "  " << std::fixed << std::setprecision(2) << std::setw(9) << q_targets[i] 
                      << "  | " << std::setw(14) << delta_targets[i]
                      << " | " << std::setw(12) << delta_calc
                      << " | " << std::setw(10) << diff
                      << " | " << std::setw(7) << std::setprecision(2) << percent_error << "%"
                      << std::endl;
        }
        std::cout << std::string(70, '-') << std::endl;
        std::cout << "  χ² (phase shifts) = " << std::scientific << std::setprecision(3) << total_error << std::endl;
        
        if (channel == 0) {
            double scatt_chi = 8.0 * std::pow(sqwell.GetScatteringLength(channel) - scatt_target_s12, 2);
            std::cout << "  χ² (scatt. length) = " << scatt_chi << std::endl;
            std::cout << "  Total χ² = " << (total_error + scatt_chi) << std::endl;
        }
    }
    
    // Save phase shifts to file
    std::cout << "\n========================================" << std::endl;
    std::cout << "Saving complete phase shift data to 'pd_phase_shifts_fitted.dat'..." << std::endl;
    sqwell.PrintPhaseShifts("output/pd_phase_shifts_fitted.dat");
    std::cout << "✓ Data saved" << std::endl;
    
    // Create comprehensive plots
    std::cout << "\n========================================" << std::endl;
    std::cout << "GENERATING PLOTS" << std::endl;
    std::cout << "========================================" << std::endl;
    
    TCanvas* c1 = new TCanvas("c1", "p-d Phase Shifts", 1200, 600);
    c1->Divide(2, 1);
    
    int colors[2] = {kBlue, kRed};
    
    for (int channel = 0; channel < 2; channel++) {
        c1->cd(channel + 1);
        
        // Theory (smooth curve)
        TGraph* gr_theory = new TGraph();
        int n_plot_points = 0;
        for (int iq = 0; iq < 80; iq++) {
            double q = iq * 2.0;
            if (q > 0) {
                double phase = sqwell.GetPhaseShift(channel, q) * 180.0 / Constants::PI;
                gr_theory->SetPoint(n_plot_points++, q, phase);
            }
        }
        gr_theory->SetLineColor(colors[channel]);
        gr_theory->SetLineWidth(3);
        gr_theory->SetTitle(channel_names[channel].c_str());
        gr_theory->GetXaxis()->SetTitle("q (MeV/c)");
        gr_theory->GetYaxis()->SetTitle("#delta (degrees)");
        gr_theory->Draw("AL");
        
        // Target points
        const auto& q_targets = (channel == 0) ? q_targets_s12 : q_targets_s32;
        const auto& delta_targets = (channel == 0) ? delta_targets_s12 : delta_targets_s32;
        
        TGraph* gr_target = new TGraph();
        for (size_t i = 0; i < q_targets.size(); i++) {
            gr_target->SetPoint(i, q_targets[i], delta_targets[i]);
        }
        gr_target->SetMarkerStyle(20);
        gr_target->SetMarkerSize(1.5);
        gr_target->SetMarkerColor(kBlack);
        gr_target->SetLineColor(kBlack);
        gr_target->Draw("P SAME");
        
        // Legend
        TLegend* leg = new TLegend(0.15, 0.70, 0.45, 0.85);
        if (channel == 1) {
            leg->SetX1(0.55);
            leg->SetX2(0.85);
        }
        leg->AddEntry(gr_theory, "Fitted potential", "l");
        leg->AddEntry(gr_target, "Target values", "p");
        leg->SetBorderSize(0);
        leg->SetFillStyle(0);
        leg->Draw();
        
        // Add text with scattering length
        if (channel == 0) {
            TLatex* latex = new TLatex();
            latex->SetNDC();
            latex->SetTextSize(0.035);
            latex->DrawLatex(0.18, 0.62, Form("a_{s} = %.3f fm", sqwell.GetScatteringLength(channel)));
            latex->DrawLatex(0.18, 0.55, Form("Target: %.3f fm", scatt_target_s12));
        }
    }
    
    c1->Update();
    c1->SaveAs("output/pd_phase_shifts_comparison.pdf");
    std::cout << "✓ Individual plots saved" << std::endl;
    
    // Combined plot
    TCanvas* c2 = new TCanvas("c2", "p-d All Phase Shifts", 1000, 700);
    TMultiGraph* mg = new TMultiGraph();
    TLegend* leg2 = new TLegend(0.15, 0.70, 0.45, 0.88);
    leg2->SetBorderSize(0);
    leg2->SetFillStyle(1001);
    
    for (int channel = 0; channel < 2; channel++) {
        // Theory curves
        TGraph* gr_theory = new TGraph();
        int n_plot_points = 0;
        for (int iq = 0; iq < 80; iq++) {
            double q = iq * 2.0;
            if (q > 0) {
                double phase = sqwell.GetPhaseShift(channel, q) * 180.0 / Constants::PI;
                gr_theory->SetPoint(n_plot_points++, q, phase);
            }
        }
        gr_theory->SetLineColor(colors[channel]);
        gr_theory->SetLineWidth(3);
        mg->Add(gr_theory, "L");
        
        // Target points
        const auto& q_targets = (channel == 0) ? q_targets_s12 : q_targets_s32;
        const auto& delta_targets = (channel == 0) ? delta_targets_s12 : delta_targets_s32;
        
        TGraph* gr_target = new TGraph();
        for (size_t i = 0; i < q_targets.size(); i++) {
            gr_target->SetPoint(i, q_targets[i], delta_targets[i]);
        }
        gr_target->SetMarkerStyle(20);
        gr_target->SetMarkerSize(1.2);
        gr_target->SetMarkerColor(colors[channel]);
        gr_target->SetLineColor(colors[channel]);
        mg->Add(gr_target, "P");
        
        leg2->AddEntry(gr_theory, channel_names[channel].c_str(), "lp");
    }
    
    mg->Draw("A");
    mg->SetTitle("p-d Phase Shifts: Fitted Theory vs Target Values;q (MeV/c);#delta (degrees)");
    mg->GetXaxis()->SetLimits(0, 160);
    leg2->Draw();
    
    c2->Update();
    c2->SaveAs("output/pd_all_phase_shifts.pdf");
    std::cout << "✓ Combined plot saved" << std::endl;
    
    // Create additional plot comparing full momentum range
    TCanvas* c3 = new TCanvas("c3", "p-d Phase Shifts - Full Range", 1000, 700);
    TMultiGraph* mg2 = new TMultiGraph();
    
    for (int channel = 0; channel < 2; channel++) {
        TGraph* gr = new TGraph();
        for (int iq = 1; iq < 80; iq++) {
            double q = (iq + 0.5) * 2.0;
            double phase = sqwell.GetPhaseShift(channel, q) * 180.0 / Constants::PI;
            gr->SetPoint(iq - 1, q, phase);
        }
        gr->SetLineColor(colors[channel]);
        gr->SetLineWidth(3);
        gr->SetLineStyle(1);
        mg2->Add(gr, "L");
    }
    
    mg2->Draw("A");
    mg2->SetTitle("p-d Phase Shifts (Full Momentum Range);q (MeV/c);#delta (degrees)");
    mg2->GetXaxis()->SetLimits(0, 160);
    
    TLegend* leg3 = new TLegend(0.65, 0.15, 0.88, 0.30);
    leg3->SetBorderSize(0);
    leg3->SetFillStyle(1001);
    for (int channel = 0; channel < 2; channel++) {
        TGraph* dummy = new TGraph();
        dummy->SetLineColor(colors[channel]);
        dummy->SetLineWidth(3);
        leg3->AddEntry(dummy, channel_names[channel].c_str(), "l");
    }
    leg3->Draw();
    
    c3->Update();
    c3->SaveAs("output/pd_phase_shifts_full_range.pdf");
    std::cout << "✓ Full range plot saved" << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "ANALYSIS COMPLETE" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nGenerated files:" << std::endl;
    std::cout << "  - output/pd_phase_shifts_fitted.dat (numerical data)" << std::endl;
    std::cout << "  - output/pd_phase_shifts_comparison.pdf (2-panel plot)" << std::endl;
    std::cout << "  - output/pd_all_phase_shifts.pdf (combined plot with targets)" << std::endl;
    std::cout << "  - output/pd_phase_shifts_full_range.pdf (full momentum range)" << std::endl;
    std::cout << "\nClose the plot windows to exit." << std::endl;
}