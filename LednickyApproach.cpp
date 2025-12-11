#include <complex>

#include <TGraph.h>
#include <TFile.h>

#include "include/WaveFunctionCore.h"
#include "include/Physics.h"

void drawWaveFunction()
{
    const double k = 100; // MeV - relative momentum
    const double Rsource = 7; // fm - typical radius in central PbPb collisions
    const double mu = reducedMass(constant::M_PROTON, constant::M_HE3);
    const double eta = SommerfeldParameter(k, mu, constant::Z_HE3, constant::Z_PROTON);

    const double a0 = 11.1; // fm - scattering length
    const double a0_natural_units = a0 / constant::HBARC;
    const double r0 = 1.8; // fm - effective range
    const double r0_natural_units = r0 / constant::HBARC;

    TGraph graph_psi0_real(70), graph_phi0_real(70), graph_psi0_imag(70), graph_phi0_imag(70);

    int iterator = 0;
    for (double r = 0.1; r < Rsource; r += 0.1)
    {
        const double r_natural_units = r / constant::HBARC;

        const std::complex<double> psi0_value = psi0(r_natural_units, k, eta); // pure Coulomb
        const std::complex<double> phi0_value = phi0(r_natural_units, k, eta, mu, a0_natural_units, r0_natural_units, 1.);

        graph_psi0_real.SetPoint(iterator, r, std::real(psi0_value));
        graph_psi0_imag.SetPoint(iterator, r, std::imag(psi0_value));
        graph_phi0_real.SetPoint(iterator, r, std::real(phi0_value));
        graph_phi0_imag.SetPoint(iterator, r, std::imag(phi0_value));
        iterator++;
    }

    graph_psi0_real.SetTitle(";#it{r}* (fm); Re[#psi_{0}](#it{r})");
    graph_psi0_imag.SetTitle(";#it{r}* (fm); Im[#psi_{0}](#it{r})");
    graph_phi0_real.SetTitle(";#it{r}* (fm); Re[#phi_{0}](#it{r})");
    graph_phi0_imag.SetTitle(";#it{r}* (fm); Im[#phi_{0}](#it{r})");
    TFile outfile("output/lednicky.root", "recreate");
    graph_psi0_real.Write("psi0_real");
    graph_psi0_imag.Write("psi0_imag");
    graph_phi0_real.Write("phi0_real");
    graph_phi0_imag.Write("phi0_imag");
}