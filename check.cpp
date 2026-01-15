#include <TH2F.h>
#include <TFile.h>
#include <TRandom3.h>

namespace Constants {
    const double M_PROTON = 938.272;    // GeV/c^2
    const double M_DEUTERON = 1875.613;  // GeV/c^2
    const double M_HELIUM3 = 2808.391;  // GeV/c^2
    const double Z_PROTON = 1.0;
    const double Z_DEUTERON = 1.0;
    const double Z_HELIUM3 = 2.0;
    const double HBARC = 197.3;          // MeV·fm
    const double ALPHA_EM = 1.0/137.036;
    const double PI = 3.141592653589793238462643383279502884;	/* pi */
    const double EULER = 5.772156649015328606065120900824024310e-01;
}

void check()
{
    double r_vec[3] = {0., 0., 0.};
    const double sqrt2 = std::sqrt(2);
    double r, costheta, phi, sintheta;
    const int n_iterations = 100000;
    const double R_source = 2.;

    TH2F h_xy("h_xy", ";x; y;", 160, -8, 8, 160, -8, 8);
    TH2F h_yz("h_yz", ";y; z;", 160, -8, 8, 160, -8, 8);
    TH2F h_xz("h_xz", ";x; z;", 160, -8, 8, 160, -8, 8);
    
    for (size_t iter = 0; static_cast<int>(iter) < n_iterations; ++iter) {
        
        //r_vec[0] = gRandom->Gaus(0., R_source * sqrt2);
        //r_vec[1] = gRandom->Gaus(0., R_source * sqrt2);
        //r_vec[2] = gRandom->Gaus(0., R_source * sqrt2);

        r = gRandom->Gaus(0., R_source * sqrt2);
        costheta = gRandom->Uniform(-1., 1.);
        phi = gRandom->Uniform(0., 2. * Constants::PI);

        sintheta = std::sqrt(1. - costheta*costheta);

        r_vec[0] = r * sintheta* std::cos(phi);
        r_vec[1] = r * sintheta* std::sin(phi);
        r_vec[2] = r * costheta;

        gRandom->Sphere(r_vec[0], r_vec[1], r_vec[2], R_source*sqrt2);

        h_xy.Fill(r_vec[0], r_vec[1]);
        h_yz.Fill(r_vec[1], r_vec[2]);
        h_xz.Fill(r_vec[0], r_vec[2]);
    }

    TFile outfile("check_sphere.root", "recreate");
    h_xy.Write();
    h_yz.Write();
    h_xz.Write();
    outfile.Close();
}