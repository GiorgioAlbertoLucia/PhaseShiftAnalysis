#pragma once

#include "Constants.h"
#include "Potentials.h"

#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>

class NumericalCoulombStrongWavefunction {
public:
    NumericalCoulombStrongWavefunction(
        double m1, double m2,
        double charge1, double charge2,
        std::vector<std::shared_ptr<Potential>> potential_singlet,
        std::vector<std::shared_ptr<Potential>> potential_triplet,
        double cg_singlet = 0.25,
        double cg_triplet = 0.75,
        int lmax = 30);
    
    ~NumericalCoulombStrongWavefunction();
    
    // Solve radial Schrödinger equation for given k
    void solve_radial_wavefunction(double k, int spin_state);
    void solve_radial_wavefunction_numerov(double k, int spin_state);
    
    // Compute full 3D wavefunction
    void compute_wavefunction(const double k_vec[3], const double r_vec[3],
                             std::complex<double>& psi_s, std::complex<double>& psi_t);
    
    // Correlation function
    double correlation_function_optimized(double k, double R_source, int n_iterations = 100000);
    
private:
    double m1_, m2_, mu_;
    double charge1_, charge2_;
    double a_C_;

    // The idea is that you can provide a potential for each value of l, from which the 
    // wavefunction will be computed. If a  potential is not provided, the Coulomb wave for that
    // value of will be used
    std::vector<std::shared_ptr<Potential>> pot_singlet_;   // potential for each desired value of l
    std::vector<std::shared_ptr<Potential>> pot_triplet_;   // potential for each desired value of l
    double cg_singlet_, cg_triplet_;
    
    // Current momentum and spin state for ODE solving
    double k_current_;
    int spin_current_;
    int lmax_;
    
    // Radial wavefunction storage
    std::vector<double> r_grid_;
    std::vector<std::complex<double>> u_singlet_;  // u(r) = r * psi(r)
    std::vector<std::complex<double>> u_triplet_;
    
    double _eta(double k) const;
    double V_coulomb(double r) const;
    
    // ODE system for radial equation
    static int radial_ode(double r, const double y[], double dydr[], void* params);
    
    // Interpolate wavefunction at arbitrary r
    std::complex<double> interpolate_u(double r, int spin_state) const;
};

// Constructor
NumericalCoulombStrongWavefunction::NumericalCoulombStrongWavefunction(
    double m1, double m2,
    double charge1, double charge2,
    std::vector<std::shared_ptr<Potential>> potential_singlet,
    std::vector<std::shared_ptr<Potential>> potential_triplet,
    double cg_singlet, double cg_triplet, int lmax)
    : m1_(m1), m2_(m2), charge1_(charge1), charge2_(charge2),
      pot_singlet_(potential_singlet), pot_triplet_(potential_triplet),
      cg_singlet_(cg_singlet), cg_triplet_(cg_triplet), lmax_(lmax)
{
    mu_ = (m1_ * m2_) / (m1_ + m2_);
    a_C_ = Constants::HBARC / (charge1_ * charge2_ * mu_ * Constants::ALPHA_EM);
}

NumericalCoulombStrongWavefunction::~NumericalCoulombStrongWavefunction() {}

double NumericalCoulombStrongWavefunction::_eta(double k) const {
    return 1.0 / (k * a_C_);
}

double NumericalCoulombStrongWavefunction::V_coulomb(double r) const {
    if (r < 1e-10) return 0.0; // Regularize
    return charge1_ * charge2_ * Constants::ALPHA_EM * Constants::HBARC / r;
}

// ODE system: d²u/dr² = [2μ(V(r) - E)/ℏ² + l(l+1)/r²]u
// Convert to first order: y[0] = u, y[1] = du/dr
int NumericalCoulombStrongWavefunction::radial_ode(double r, const double y[], 
                                                    double dydr[], void* params) {
    auto* wf = static_cast<NumericalCoulombStrongWavefunction*>(params);
    
    double k = wf->k_current_;
    double E = Constants::HBARC * Constants::HBARC * k * k / (2.0 * wf->mu_); // MeV
    
    double V_total = 0.0;
    if (wf->spin_current_ == 0) {
        V_total = wf->pot_singlet_[0] ? wf->pot_singlet_[0]->evalWithCoulomb(r) : 0.0;
    } else {
        V_total = wf->pot_triplet_[0] ? wf->pot_triplet_[0]->evalWithCoulomb(r) : 0.0;
    }
    
    // For l=0: d²u/dr² = 2μ(V - E)/ℏ² * u
    double coeff = 2.0 * wf->mu_ / (Constants::HBARC * Constants::HBARC) * (V_total - E);
    
    dydr[0] = y[1];           // du/dr
    dydr[1] = coeff * y[0];   // d²u/dr²
    
    return GSL_SUCCESS;
}

void NumericalCoulombStrongWavefunction::solve_radial_wavefunction(double k, int spin_state) {
    k_current_ = k;
    spin_current_ = spin_state;
    
    // Setup grid
    double r_min = 0.01;  // fm
    double r_max = 50.0;  // fm
    int n_points = 1000;
    
    r_grid_.resize(n_points);
    auto& u_vec = (spin_state == 0) ? u_singlet_ : u_triplet_;
    u_vec.resize(n_points);
    
    for (int i = 0; i < n_points; ++i) {
        r_grid_[i] = r_min + i * (r_max - r_min) / (n_points - 1);
    }
    
    // Initial conditions (Coulomb-like at small r)
    double eta = _eta(k);
    double r0 = r_grid_[0];
    double rho0 = k * r0;
    
    // For small r: u(r) ≈ C * r * exp(-πη/2) * (2kr)^(iη) / Γ(1+iη)
    // Simplified: u(r) ≈ r
    double y[2] = {r0, 1.0};
    
    // GSL ODE solver
    const gsl_odeiv2_step_type* T = gsl_odeiv2_step_rk8pd;
    gsl_odeiv2_step* s = gsl_odeiv2_step_alloc(T, 2);
    gsl_odeiv2_control* c = gsl_odeiv2_control_y_new(1e-8, 0.0);
    gsl_odeiv2_evolve* e = gsl_odeiv2_evolve_alloc(2);
    
    gsl_odeiv2_system sys = {radial_ode, nullptr, 2, this};
    
    double r = r0;
    double h = 0.01;
    
    u_vec[0] = std::complex<double>(y[0], 0.0);
    
    for (int i = 1; i < n_points; ++i) {
        double r_target = r_grid_[i];
        
        while (r < r_target) {
            int status = gsl_odeiv2_evolve_apply(e, c, s, &sys, &r, r_target, &h, y);
            if (status != GSL_SUCCESS) {
                std::cerr << "Error in ODE solver" << std::endl;
                break;
            }
        }
        
        u_vec[i] = std::complex<double>(y[0], 0.0);
    }
    
    // Normalize
    double norm = 0.0;
    for (int i = 1; i < n_points; ++i) {
        double dr = r_grid_[i] - r_grid_[i-1];
        norm += std::norm(u_vec[i]) * dr;
    }
    norm = std::sqrt(norm);
    
    for (auto& u : u_vec) {
        u /= norm;
    }
    
    gsl_odeiv2_evolve_free(e);
    gsl_odeiv2_control_free(c);
    gsl_odeiv2_step_free(s);
}

void NumericalCoulombStrongWavefunction::solve_radial_wavefunction_numerov(double k, int spin_state) {

    using namespace std::complex_literals;

    k_current_ = k;
    spin_current_ = spin_state;
    
    // Setup grid
    double r_min = 0.01;  // fm
    double r_max = 50.0;  // fm
    int n_points = 1000;
    double dr = (r_max - r_min) / (n_points - 1);
    
    r_grid_.resize(n_points);
    auto& u_vec = (spin_state == 0) ? u_singlet_ : u_triplet_;
    u_vec.resize(n_points);
    
    for (int i = 0; i < n_points; ++i) {
        r_grid_[i] = r_min + i * dr;
    }
    
    // Initial conditions (Coulomb-like at small r)
    double eta = _eta(k);
    double r0 = r_grid_[0];
    double rho0 = k * r0;
    
    const double e_pi_eta_2 = std::exp(-0.5*Constants::PI*eta);
    
    // For small r: u(r) ≈ C * r * exp(-πη/2) * (2kr)^(iη) / Γ(1+iη)
    // Simplified: u(r) ≈ r
    double y[2] = {r0, 1.0};
    
    double r = r0;


    const double h2 = dr*dr/12.;
    
    //u_vec[0] = std::complex<double>(y[0], 0.0);

    std::vector<std::shared_ptr<Potential>>& potentials = spin_state == 0 ? pot_singlet_ : pot_triplet_;
    
    for (int i = 1; i < n_points; ++i) {

        double factorial = 1.;
        double prefactor = 1.;
        const double rho = k * r_grid_[i];
        const std::complex<double> e_ikr = std::exp(1.i * rho);
        double rho_lplus1 = rho;
        
        for (int l = 0; l < lmax_; l++) {

            factorial = factorial * (2. * l + 1.);
            rho_lplus1 *= rho;
            prefactor *= 2;
            
            // Numerov requires initial condition: use Coulomb Wave
            if ((static_cast<size_t>(l) < potentials.size()) || i < 2) {
                
                const std::complex<double> gamma = math::complex_gamma(l + 1. + 1.i*eta);
                const std::complex<double> F = math::hypergeometric_1F1_complex(l + 1. + 1.i*eta, 2.*l + 2, -2.*1.i*rho);

                // from textbook
                u_vec[i] = prefactor * e_pi_eta_2 * gamma * e_ikr * rho_lplus1 * F / factorial;


            } else {

                // from Numerov


            }
        }
    }
    
    // Normalize
    double norm = 0.0;
    for (int i = 1; i < n_points; ++i) {
        double dr = r_grid_[i] - r_grid_[i-1];
        norm += std::norm(u_vec[i]) * dr;
    }
    norm = std::sqrt(norm);
    
    for (auto& u : u_vec) {
        u /= norm;
    }
    
}

std::complex<double> NumericalCoulombStrongWavefunction::interpolate_u(double r, int spin_state) const {
    const auto& u_vec = (spin_state == 0) ? u_singlet_ : u_triplet_;
    
    if (r <= r_grid_.front()) return u_vec.front();
    if (r >= r_grid_.back()) {
        // Asymptotic form: u(r) ~ sin(kr - eta*ln(2kr) + delta_l)
        return std::complex<double>(0.0, 0.0);
    }
    
    // Linear interpolation
    auto it = std::lower_bound(r_grid_.begin(), r_grid_.end(), r);
    int idx = std::distance(r_grid_.begin(), it);
    if (idx == 0) idx = 1;
    
    double r1 = r_grid_[idx-1];
    double r2 = r_grid_[idx];
    double t = (r - r1) / (r2 - r1);
    
    return u_vec[idx-1] * (1.0 - t) + u_vec[idx] * t;
}

void NumericalCoulombStrongWavefunction::compute_wavefunction(
    const double k_vec[3], const double r_vec[3],
    std::complex<double>& psi_s, std::complex<double>& psi_t) 
{
    double k = std::hypot(k_vec[0], k_vec[1], k_vec[2]);
    double r = std::hypot(r_vec[0], r_vec[1], r_vec[2]);
    
    // Solve if not already done for this k
    static double k_cached = -1.0;
    if (std::abs(k - k_cached) > 1e-6) {
        solve_radial_wavefunction(k, 0); // singlet
        solve_radial_wavefunction(k, 1); // triplet
        k_cached = k;
    }
    
    // For l=0: psi(r) = u(r) / r * Y_00 = u(r) / (r * sqrt(4π))
    std::complex<double> u_s = interpolate_u(r, 0);
    std::complex<double> u_t = interpolate_u(r, 1);
    
    double Y00 = 1.0 / std::sqrt(4.0 * Constants::PI);
    
    psi_s = u_s / r * Y00;
    psi_t = u_t / r * Y00;
}

double NumericalCoulombStrongWavefunction::correlation_function_optimized(
    double k, double R_source, int n_iterations) 
{
    double k_vec[3] = {0.0, 0.0, k};
    double r_vec[3];
    const double sqrt2 = std::sqrt(2.0);
    
    std::complex<double> psi_s, psi_t;
    double C_s = 0.0, C_t = 0.0;
    
    // Pre-solve wavefunctions
    solve_radial_wavefunction(k, 0);
    solve_radial_wavefunction(k, 1);
    
    for (int iter = 0; iter < n_iterations; ++iter) {
        r_vec[0] = gRandom->Gaus() * R_source * sqrt2;
        r_vec[1] = gRandom->Gaus() * R_source * sqrt2;
        r_vec[2] = gRandom->Gaus() * R_source * sqrt2;
        
        compute_wavefunction(k_vec, r_vec, psi_s, psi_t);
        
        C_s += std::norm(psi_s);
        C_t += std::norm(psi_t);
    }
    
    C_s /= n_iterations;
    C_t /= n_iterations;
    
    return cg_singlet_ * C_s + cg_triplet_ * C_t;
}
