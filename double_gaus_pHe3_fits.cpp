#include "include/pHe3DoubleGaus.h"
#include "phase_shift_data.h"

#include <TCanvas.h>
#include <TMultiGraph.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TLegend.h>
#include <TLorentzVector.h>
#include <TVector3.h>
#include <TAxis.h>
#include <TFile.h>

#include <iomanip>
#include <set>

// ─── Constants ───────────────────────────────────────────────────────────────
static const double M_PROTON  = 938.2720813;   // MeV/c²
static const double M_HE3     = 2808.391383;   // MeV/c²

// ─── Kinematics ──────────────────────────────────────────────────────────────
// Convert lab kinetic energy (MeV) → CM momentum k* (MeV/c)
double ELabToKstar(double T_lab) {
    double p_lab = std::sqrt(T_lab * T_lab + 2.0 * T_lab * M_PROTON);
    TLorentzVector beam(0, 0, p_lab, std::sqrt(p_lab*p_lab + M_PROTON*M_PROTON));
    TLorentzVector target(0, 0, 0, M_HE3);
    TLorentzVector total = beam + target;
    TVector3 boost = total.BoostVector();
    beam.Boost(-boost);
    return beam.P();
}

// ─── Data loading ────────────────────────────────────────────────────────────
// Spectroscopic label → channel index: 1S0→0, 3S1→1, 3P0→2, 3P1→3
static const std::map<std::string, int> CHANNEL_MAP = {
    {"1S0", 0}, {"3S1", 1}, {"3P0", 2}, {"3P1", 3}
};

// Merge data points from all three tables into per-channel vectors
void LoadAllData(std::vector<double>& q_merged,
                 std::vector<std::vector<double>>& delta_merged,
                 std::vector<std::vector<double>>& error_merged)
{
    q_merged.clear();
    delta_merged.assign(4, {});
    error_merged.assign(4, {});

    // Collect all unique k* values across relevant channels
    std::set<double> kstar_set;
    auto collectFrom = [&](const std::map<std::string, std::vector<DataPoint>>& table) {
        for (auto& [label, points] : table) {
            if (CHANNEL_MAP.count(label) == 0) continue;
            for (auto& dp : points) kstar_set.insert(ELabToKstar(dp.energy_mev));
        }
    };
    collectFrom(TABLE_15_14_23);
    collectFrom(TABLE_15_16_40);
    collectFrom(TABLE_15_17_51);

    q_merged.assign(kstar_set.begin(), kstar_set.end()); // already sorted
    delta_merged.assign(4, std::vector<double>(q_merged.size(), std::numeric_limits<double>::quiet_NaN()));
    error_merged.assign(4, std::vector<double>(q_merged.size(), 0.0));

    auto fillFrom = [&](const std::map<std::string, std::vector<DataPoint>>& table) {
        for (auto& [label, points] : table) {
            auto it = CHANNEL_MAP.find(label);
            if (it == CHANNEL_MAP.end()) continue;
            int ch = it->second;
            for (auto& dp : points) {
                double kstar = ELabToKstar(dp.energy_mev);
                auto pos = std::lower_bound(q_merged.begin(), q_merged.end(), kstar);
                size_t idx = std::distance(q_merged.begin(), pos);
                delta_merged[ch][idx] = dp.value;
                error_merged[ch][idx] = dp.error;
            }
        }
    };
    fillFrom(TABLE_15_14_23);
    fillFrom(TABLE_15_16_40);
    fillFrom(TABLE_15_17_51);
}

// ─── Plotting ────────────────────────────────────────────────────────────────
static const int   COLORS[4]       = {kBlue, kRed, kGreen+2, kMagenta};
static const char* CHANNEL_NAMES[4] = {
    "L=0, S=0 (^{1}S_{0})", "L=0, S=1 (^{3}S_{1})",
    "L=1, S=0 (^{3}P_{0})", "L=1, S=1 (^{3}P_{1})"
};

TGraph* MakeTheoryCurve(pHe3DoubleGaus& sw, int ch, int color) {
    TGraph* gr = new TGraph();
    for (int i = 0; i < 200; ++i) {
        double q = i * 0.5;
        gr->SetPoint(i, q, sw.GetPhaseShift(ch, q) * 180.0 / Constants::PI);
    }
    gr->SetLineColor(color);
    gr->SetLineWidth(3);
    return gr;
}

TGraphErrors* MakeExpGraph(const std::vector<double>& q,
                           const std::vector<double>& delta,
                           const std::vector<double>& err, int color)
{
    TGraphErrors* gr = new TGraphErrors();
    int n = 0;
    for (size_t i = 0; i < q.size(); ++i) {
        if (std::isnan(delta[i])) continue;
        gr->SetPoint(n, q[i], delta[i]);
        gr->SetPointError(n, 0, err[i]);
        ++n;
    }
    gr->SetMarkerStyle(20);
    gr->SetMarkerSize(1.2);
    gr->SetMarkerColor(color);
    gr->SetLineColor(color);
    return gr;
}

// ─── Main ─────────────────────────────────────────────────────────────────────
void double_gaus_pHe3_fits() {

    // --- Load & convert experimental data ---
    std::vector<double> q_all;
    std::vector<std::vector<double>> delta_all, error_all;
    LoadAllData(q_all, delta_all, error_all);

    pHe3DoubleGaus sqwell;
    sqwell.LoadExperimentalData(q_all, delta_all, error_all);

    // --- Fit all channels ---
    std::cout << "=== p-He3 Double Gaussian Potential Analysis ===\n\n";
    for (int ch = 0; ch < 4; ++ch) {
        std::cout << "Fitting channel " << ch << ": " << CHANNEL_NAMES[ch] << "\n";
        sqwell.FitToPhaseShifts(ch, 1000);
        std::cout << "  ✓ Done\n";
    }

    // --- Print fitted parameters ---
    std::cout << "\n=== Fitted Parameters ===\n";
    for (int ch = 0; ch < 4; ++ch) {
        std::vector<double> sigma, V;
        sqwell.GetParameters(ch, sigma, V);
        std::cout << CHANNEL_NAMES[ch] << "\n";
        std::cout << "  σ (fm): ";
        for (double v : sigma) std::cout << std::fixed << std::setprecision(6) << v << "  ";
        std::cout << "\n  V (MeV): ";
        for (double v : V) std::cout << std::fixed << std::setprecision(4) << v << "  ";
        std::cout << "\n";
    }

    // --- χ² summary table ---
    std::cout << "\n=== Comparison with Data ===\n";
    for (int ch = 0; ch < 4; ++ch) {
        std::cout << "\n" << CHANNEL_NAMES[ch] << "\n";
        std::cout << std::string(80, '-') << "\n";
        std::cout << "  k* (MeV/c) | δ_exp (°) | δ_err (°) | δ_calc (°) | Δ (°)  | χ²_i\n";
        std::cout << std::string(80, '-') << "\n";
        double chi2 = 0;
        for (size_t i = 0; i < q_all.size(); ++i) {
            if (std::isnan(delta_all[ch][i])) continue;
            double calc = sqwell.GetPhaseShift(ch, q_all[i]) * 180.0 / Constants::PI;
            double diff = calc - delta_all[ch][i];
            double chi2i = error_all[ch][i] > 0 ? diff*diff / (error_all[ch][i]*error_all[ch][i]) : 0;
            chi2 += chi2i;
            std::cout << std::fixed << std::setprecision(2)
                      << "  " << std::setw(10) << q_all[i]
                      << " | " << std::setw(9) << delta_all[ch][i]
                      << " | " << std::setw(9) << error_all[ch][i]
                      << " | " << std::setw(10) << calc
                      << " | " << std::setw(6) << diff
                      << " | " << std::setw(6) << chi2i << "\n";
        }
        std::cout << std::string(80, '-') << "\n";
        std::cout << "  Total χ² = " << std::scientific << std::setprecision(3) << chi2 << "\n";
    }

    // --- Save data ---
    sqwell.PrintPhaseShifts("output/pHe3_double_gaus_phase_shifts_fitted.dat");
    std::cout << "\n✓ Phase shifts saved\n";

    // --- Open ROOT output file ---
    TFile* outFile = new TFile("output/pHe3_double_gaus_phase_shifts_fitted.root", "RECREATE");

    // --- 4-panel plot ---
    TCanvas* c1 = new TCanvas("c1", "p-^{3}He Phase Shifts", 1200, 900);
    c1->Divide(2, 2);
    for (int ch = 0; ch < 4; ++ch) {
        c1->cd(ch + 1);
        TGraph*       gr_th  = MakeTheoryCurve(sqwell, ch, COLORS[ch]);
        TGraphErrors* gr_exp = MakeExpGraph(q_all, delta_all[ch], error_all[ch], kBlack);
        gr_th->SetTitle(CHANNEL_NAMES[ch]);
        gr_th->GetXaxis()->SetTitle("#it{k}* (MeV/#it{c})");
        gr_th->GetYaxis()->SetTitle("#delta (degrees)");
        gr_th->Draw("AL");
        gr_exp->Draw("P SAME");
        bool phase_negative = (ch < 2);
        TLegend* leg = new TLegend(phase_negative ? 0.55 : 0.15, 0.70, phase_negative ? 0.85 : 0.45, 0.85);
        leg->AddEntry(gr_th,  "Fit",           "l");
        leg->AddEntry(gr_exp, "Data (merged)", "p");
        leg->SetBorderSize(0); leg->SetFillStyle(0);
        leg->Draw();

        // Save graphs into per-channel directory
        TDirectory* dir = outFile->mkdir(Form("channel_%d", ch), CHANNEL_NAMES[ch]);
        dir->cd();
        gr_th ->SetName("theory");
        gr_exp->SetName("experiment");
        gr_th ->Write();
        gr_exp->Write();
        outFile->cd();
    }
    c1->Print("output/pHe3_double_gaus_phase_shifts_comparison.pdf");
    outFile->cd();
    c1->Write("canvas_4panel");

    // --- Multi-page individual channel PDF ---
    TCanvas* cInd = new TCanvas("cInd", "Individual Channels", 1200, 900);
    cInd->Print("output/pHe3_phase_shifts_individual.pdf[");
    for (int ch = 0; ch < 4; ++ch) {
        TDirectory* dir = outFile->GetDirectory(Form("channel_%d", ch));
        TGraph*       gr_th  = MakeTheoryCurve(sqwell, ch, COLORS[ch]);
        TGraphErrors* gr_exp = MakeExpGraph(q_all, delta_all[ch], error_all[ch], kBlack);
        gr_th->SetTitle(CHANNEL_NAMES[ch]);
        gr_th->GetXaxis()->SetTitle("#it{k}* (MeV/#it{c})");
        gr_th->GetYaxis()->SetTitle("#delta (degrees)");
        gr_th->Draw("AL");
        gr_exp->Draw("P SAME");
        cInd->Print("output/pHe3_phase_shifts_individual.pdf");

        dir->cd();
        cInd->SetName(Form("canvas_channel_%d", ch));
        cInd->Write();
        outFile->cd();
        cInd->Clear();
    }
    cInd->Print("output/pHe3_double_gaus_phase_shifts_individual.pdf]");

    // --- Combined overlay plot ---
    TCanvas* c2  = new TCanvas("c2", "All Channels", 1000, 700);
    TMultiGraph* mg   = new TMultiGraph();
    TLegend*     leg2 = new TLegend(0.15, 0.15, 0.38, 0.45);
    leg2->SetBorderSize(0); leg2->SetFillStyle(1001);
    for (int ch = 0; ch < 4; ++ch) {
        TGraph*       gr_th  = MakeTheoryCurve(sqwell, ch, COLORS[ch]);
        TGraphErrors* gr_exp = MakeExpGraph(q_all, delta_all[ch], error_all[ch], COLORS[ch]);
        mg->Add(gr_th,  "L");
        mg->Add(gr_exp, "P");
        leg2->AddEntry(gr_th, CHANNEL_NAMES[ch], "lp");
    }
    mg->Draw("A");
    mg->SetTitle("p-^{3}He Phase Shifts;#it{k}* (MeV/#it{c});#delta (degrees)");
    leg2->Draw();
    c2->Print("output/pHe3_all_phase_shifts.pdf");
    outFile->cd();
    c2->Write("canvas_all_channels");

    outFile->Close();
    std::cout << "\n✓ All plots saved. ROOT file written to output/pHe3_double_gaus_phase_shifts_fitted.root\n";
    std::cout << "  Structure: channel_0..3/ (theory, experiment, canvas_channel_N)\n";
    std::cout << "             canvas_4panel, canvas_all_channels\n";
}
