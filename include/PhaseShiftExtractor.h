#include <cmath>
#include <vector>
#include <gsl/gsl_sf_coulomb.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>

struct MatchingData {
    double r_match;
    double dr_match;    // distance between the first and the second matching point
    double k;
    double eta;
    int l;
    double u_num;      // Numerical solution value at r_match
    double udr_num;     // Numerical solution value at the r + dr match
    double up_num;     // Derivative at r_match
};

class PhaseShiftExtractor {
    private:
        double hbar_c = 197.327;  // MeV*fm
        double m_reduced;
        double dr;
        int l;
        
    public:
        PhaseShiftExtractor(double mass, double step, int ang_mom)
            : m_reduced(mass), dr(step), l(ang_mom) {}
        
        // Function to minimize: difference between numerical and shifted analytical
        static double matchingFunction(double delta, void* params);
        
        // Better matching: use ratio method to eliminate normalization
        static double matchingFunctionRatio(double delta, void* params);
        
        // Most robust: match using both value and derivative ratios
        double extractPhaseShift(const std::vector<double>& u_num,
                                   double r_match, double dr_match, double k, double eta);
        
        // Alternative: Simple grid search for verification
        double extractPhaseShiftGrid(const std::vector<double>& u_num,
                                        double r_match, double k, double eta);
};


double PhaseShiftExtractor::matchingFunction(double delta, void* params) {
    MatchingData* data = static_cast<MatchingData*>(params);
    
    // Shifted position for analytical solution
    double r_shifted = data->r_match + delta / data->k;
    double kr_shifted = data->k * r_shifted;
    
    // Get Coulomb wavefunction at shifted position
    gsl_sf_result F_l, Fp_l, G_l, Gp_l;
    gsl_sf_coulomb_wave_FG_e(data->eta, kr_shifted, data->l, 0,
                             &F_l, &Fp_l, &G_l, &Gp_l,
                             nullptr, nullptr);
    
    // The analytical solution (we use F_l, the regular Coulomb function)
    // Normalize it to match at the matching point
    // Actually, we want: u_num = A * F_l(kr + delta/k)
    // So we need to find delta such that the ratio is constant
    
    // Better: match both value and derivative
    // This gives us the correct normalization automatically
    
    // Return difference (we'll minimize this)
    return data->u_num - F_l.val;  // This is too simple, see below
}

// Better matching: use ratio method to eliminate normalization
double PhaseShiftExtractor::matchingFunctionRatio(double delta, void* params) {
    
    MatchingData* data = static_cast<MatchingData*>(params);
    
    const double r_shifted = data->r_match + delta / data->k;
    const double kr_shifted1 = data->k * r_shifted;
    //const double kr_shifted2 = data->k * (r_shifted + 0.001);
    const double krdr_shifted = data->k * (r_shifted + data->dr_match);
    
    double F_l1[1], F_l2[1], Fp_l[1]; // F_l(eta, x) computed in point 1 and 2 and their derivative
    if (kr_shifted1 == 0) {
        F_l1[0] = 0;
    } else {
        gsl_sf_coulomb_wave_F_array(data->l, /*L_G*/ 0, data->eta, std::abs(kr_shifted1), F_l1, Fp_l);
    }
    if (krdr_shifted == 0) {
        F_l2[0] = 0;
    } else {
        //gsl_sf_coulomb_wave_F_array(data->l, /*L_G*/ 0, data->eta, kr_shifted2, F_l2, Fp_l);
        gsl_sf_coulomb_wave_F_array(data->l, /*L_G*/ 0, data->eta, std::abs(krdr_shifted), F_l2, Fp_l);
    }
    
    //const double logderiv_num = data->r_match * data->up_num / data->u_num;
    //const double logderiv_ana = r_shifted * Fp_l[0] / F_l1[0];
    //const double result = logderiv_num - logderiv_ana;
    const double term1 = data->u_num / data->udr_num;
    const double term2 = F_l1[0] / F_l2[0];
    
    //return logderiv_num - logderiv_ana;
    return term1 - term2;
}

//// Most robust: match using both value and derivative ratios
//double PhaseShiftExtractor::extractPhaseShift(const std::vector<double>& u_num,
//                           double r_match, double k, double eta) {
//    int i_match = static_cast<int>(r_match / dr);
//    
//    // Get numerical solution and derivative at matching point
//    double u_at_match = u_num[i_match];
//    double up_at_match = (u_num[i_match+1] - u_num[i_match-1]) / (2.0 * dr);
//    
//    // Prepare data for matching
//    MatchingData data = {.r_match = r_match,
//                         .k = k,
//                         .eta = eta,
//                         .l = l,
//                         .u_num = u_at_match,
//                         .up_num = up_at_match};
//    
//    // Set up GSL root finder
//    gsl_function F;
//    F.function = &matchingFunctionRatio;
//    F.params = &data;
//    
//    // Initialize solver
//    const gsl_root_fsolver_type* T = gsl_root_fsolver_brent;
//    gsl_root_fsolver* solver = gsl_root_fsolver_alloc(T);
//    
//    // Initial bracket for delta (phase shift typically between -π and π)
//    double delta_low = -M_PI;
//    double delta_high = M_PI;
//    
//    gsl_root_fsolver_set(solver, &F, delta_low, delta_high);
//    
//    // Iterate to find root
//    int status;
//    int iter = 0;
//    int max_iter = 100;
//    double delta = 0.0;
//    
//    do {
//        iter++;
//        status = gsl_root_fsolver_iterate(solver);
//        delta = gsl_root_fsolver_root(solver);
//        delta_low = gsl_root_fsolver_x_lower(solver);
//        delta_high = gsl_root_fsolver_x_upper(solver);
//        status = gsl_root_test_interval(delta_low, delta_high, 1e-6, 1e-6);
//    } while (status == GSL_CONTINUE && iter < max_iter);
//    
//    gsl_root_fsolver_free(solver);
//    
//    return delta;  // This is the phase shift!
//}

//double PhaseShiftExtractor::extractPhaseShift(const std::vector<double>& u_num,
//                           double r_match, double dr_match, double k, double eta) {
//    
//    size_t i_match = static_cast<size_t>(r_match / dr);
//    size_t j_match = static_cast<size_t>((r_match + dr_match)/ dr); // the matching is performed at two separate points in order to eliminate dependance on the normalisation
//    if (i_match <= static_cast<size_t>(0) || i_match > u_num.size()) {
//        std::cerr << "ERROR: i_match out of bounds!" << std::endl;
//        return 0.0;
//    }
//    
//    double u_at_match = u_num[i_match];
//    double udr_at_match = u_num[j_match];
//    //double up_at_match = (u_num[i_match+1] - u_num[i_match-1]) / (2.0 * dr);
//    
//    if (std::abs(u_at_match) < 1e-15) {
//        std::cerr << "ERROR: u_at_match is essentially zero!" << std::endl;
//        return 0.0;
//    }
//    
//    MatchingData data = {.r_match = r_match,
//                         .dr_match = dr_match,
//                         .k = k,
//                         .eta = eta,
//                         .l = l,
//                         .u_num = u_at_match,
//                         .udr_num = udr_at_match,
//                         .up_num = /*up_at_match*/ 0.};
//    
//    gsl_function F;
//    F.function = &matchingFunctionRatio;
//    F.params = &data;
//    
//    const gsl_root_fsolver_type* T = gsl_root_fsolver_brent;
//    gsl_root_fsolver* solver = gsl_root_fsolver_alloc(T);
//    
//    double delta_low = -M_PI;
//    double delta_high = M_PI;
//    
//    int status = 0;
//    status = gsl_root_fsolver_set(solver, &F, delta_low, delta_high);
//    if (status != GSL_SUCCESS) {
//        std::cerr << "ERROR: gsl_root_fsolver_set failed: " << gsl_strerror(status) << std::endl;
//        gsl_root_fsolver_free(solver);
//        return 0.0;
//    }
//
//    int iter = 0;
//    int max_iter = 100;
//    double delta = 0.0;
//    
//    do {
//        iter++;
//        status = gsl_root_fsolver_iterate(solver);
//        
//        delta = gsl_root_fsolver_root(solver);
//        delta_low = gsl_root_fsolver_x_lower(solver);
//        delta_high = gsl_root_fsolver_x_upper(solver);
//        status = gsl_root_test_interval(delta_low, delta_high, 1e-6, 1e-6);
//    } while (status == GSL_CONTINUE && iter < max_iter);
//    if (status != GSL_SUCCESS) {
//        std::cerr << "ERROR: gsl_root_fsolver_set failed: " << gsl_strerror(status) << std::endl;
//    }
//
//    gsl_root_fsolver_free(solver);
//    
//    return delta;
//}

double PhaseShiftExtractor::extractPhaseShift(const std::vector<double>& u_num,
                           double r_match, double dr_match, double k, double eta) {
    
    size_t i_match = static_cast<size_t>(r_match / dr);
    size_t j_match = static_cast<size_t>((r_match + dr_match) / dr);
    
    if (i_match <= 0 || j_match >= u_num.size()) {
        std::cerr << "ERROR: indices out of bounds!" << std::endl;
        return 0.0;
    }
    
    double u_at_match = u_num[i_match];
    double udr_at_match = u_num[j_match];
    
    //std::cout << "\n=== Phase Shift Extraction ===" << std::endl;
    //std::cout << "k = " << k << ", eta = " << eta << ", l = " << l << std::endl;
    //std::cout << "r_match = " << r_match << ", dr_match = " << dr_match << std::endl;
    //std::cout << "u[i] = " << u_at_match << ", u[j] = " << udr_at_match << std::endl;
    //std::cout << "Ratio u[i]/u[j] = " << u_at_match/udr_at_match << std::endl;
    
    if (std::abs(u_at_match) < 1e-15 || std::abs(udr_at_match) < 1e-15) {
        std::cerr << "ERROR: numerical solution too small!" << std::endl;
        return 0.0;
    }
    
    MatchingData data = {.r_match = r_match,
                         .dr_match = dr_match,
                         .k = k,
                         .eta = eta,
                         .l = l,
                         .u_num = u_at_match,
                         .udr_num = udr_at_match,
                         .up_num = 0.};
    
    // Sample the matching function to find where it crosses zero
    std::vector<std::pair<double, double>> samples;
    const int n_samples = 100;
    const double delta_min = -M_PI/2.;
    const double delta_max = M_PI/2.;
    //const double delta_min = -M_PI;
    //const double delta_max = M_PI;
    
    //std::cout << "\nScanning for zero crossing..." << std::endl;
    for (int i = 0; i <= n_samples; ++i) {
        double delta = delta_min + i * (delta_max - delta_min) / n_samples;
        double f_val = matchingFunctionRatio(delta, &data);
        samples.push_back({delta, f_val});
        
        //if (i % 10 == 0) {
        //    std::cout << "  delta = " << std::setw(8) << delta 
        //              << " -> f = " << f_val << std::endl;
        //}
    }

    const double f_low = matchingFunctionRatio(-M_PI/2, &data);
    const double f_high = matchingFunctionRatio(M_PI/2, &data);
    
    if (f_low * f_high > 0) {
        std::cerr << "WARNING: No zero crossing found!" << std::endl;
        std::cerr << "  f(min) = " << f_low << std::endl;
        std::cerr << "  f(max) = " << f_high << std::endl;
        
        // Instead of returning default, find minimum |f|
        double best_delta = -M_PI/2;
        double min_abs_f = std::abs(f_low);
        
        for (double delta = -M_PI/2; delta <= M_PI/2; delta += 0.01) {
            double f_val = matchingFunctionRatio(delta, &data);
            if (std::abs(f_val) < min_abs_f) {
                min_abs_f = std::abs(f_val);
                best_delta = delta;
            }
        }
        
        std::cerr << "Returning delta with minimum |f|: " << best_delta 
                  << " (f = " << min_abs_f << ")" << std::endl;
        return best_delta;
    }
    
    // Find bracket containing zero
    double bracket_low = -M_PI;
    double bracket_high = M_PI;
    bool found_bracket = false;
    
    for (size_t i = 0; i < samples.size() - 1; ++i) {
        if (samples[i].second * samples[i+1].second < 0) {
            bracket_low = samples[i].first;
            bracket_high = samples[i+1].first;
            found_bracket = true;
            //std::cout << "Found zero crossing between " << bracket_low 
            //          << " and " << bracket_high << std::endl;
            break;
        }
    }
    
    if (!found_bracket) {
        std::cerr << "ERROR: No zero crossing found! Function values:" << std::endl;
        std::cerr << "  f(min) = " << samples.front().second << std::endl;
        std::cerr << "  f(max) = " << samples.back().second << std::endl;
        
        // Fall back to grid search for minimum absolute value
        auto min_elem = std::min_element(samples.begin(), samples.end(),
            [](const auto& a, const auto& b) { return std::abs(a.second) < std::abs(b.second); });
        
        std::cout << "Returning delta with minimum |f|: " << min_elem->first 
                  << " (f = " << min_elem->second << ")" << std::endl;
        return min_elem->first;
    }
    
    // Now use the root finder with the proper bracket
    gsl_function F;
    F.function = &matchingFunctionRatio;
    F.params = &data;
    
    const gsl_root_fsolver_type* T = gsl_root_fsolver_brent;
    gsl_root_fsolver* solver = gsl_root_fsolver_alloc(T);
    
    int status = gsl_root_fsolver_set(solver, &F, bracket_low, bracket_high);
    if (status != GSL_SUCCESS) {
        std::cerr << "ERROR: Failed to initialize solver even with bracket!" << std::endl;
        gsl_root_fsolver_free(solver);
        return (bracket_low + bracket_high) / 2.0;  // Return midpoint as guess
    }
    
    int iter = 0;
    int max_iter = 100;
    double delta = 0.0;
    
    do {
        iter++;
        status = gsl_root_fsolver_iterate(solver);
        
        if (status != GSL_SUCCESS) {
            std::cerr << "Iteration failed: " << gsl_strerror(status) << std::endl;
            break;
        }
        
        delta = gsl_root_fsolver_root(solver);
        double delta_low = gsl_root_fsolver_x_lower(solver);
        double delta_high = gsl_root_fsolver_x_upper(solver);
        
        status = gsl_root_test_interval(delta_low, delta_high, 1e-6, 1e-6);
        
    } while (status == GSL_CONTINUE && iter < max_iter);
    
    //std::cout << "Converged: delta = " << delta << " rad = " 
    //          << delta * 180.0 / M_PI << " deg" << std::endl;
    
    gsl_root_fsolver_free(solver);
    
    return delta;
}

// Alternative: Simple grid search for verification
double PhaseShiftExtractor::extractPhaseShiftGrid(const std::vector<double>& u_num,
                                double r_match, double k, double eta) {
    int i_match = static_cast<int>(r_match / dr);
    double u_at_match = u_num[i_match];
    double up_at_match = (u_num[i_match+1] - u_num[i_match-1]) / (2.0 * dr);
    
    double logderiv_num = r_match * up_at_match / u_at_match;
    
    // Grid search over possible phase shifts
    double best_delta = 0.0;
    double min_difference = 1e10;
    
    for (double delta = -M_PI; delta <= M_PI; delta += 0.001) {
        double r_shifted = r_match + delta / k;
        double kr_shifted = k * r_shifted;
        
        gsl_sf_result F_l, Fp_l, G_l, Gp_l;
        gsl_sf_coulomb_wave_FG_e(eta, kr_shifted, l, 0,
                                 &F_l, &Fp_l, &G_l, &Gp_l,
                                 nullptr, nullptr);
        
        // Convert Fp (derivative w.r.t. kr) to derivative w.r.t. r
        Fp_l.val *= k;
        
        double logderiv_ana = r_shifted * Fp_l.val / F_l.val;
        double diff = std::abs(logderiv_num - logderiv_ana);
        
        if (diff < min_difference) {
            min_difference = diff;
            best_delta = delta;
        }
    }
    
    return best_delta;
}