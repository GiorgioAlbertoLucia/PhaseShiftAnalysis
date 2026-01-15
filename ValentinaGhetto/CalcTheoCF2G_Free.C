#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include <TGraph.h>
#include <TFile.h>

using namespace std;

TGraph *CalcCF(float sourceSize, float Ref0, float Imf0);

// Only radii of the two gaussian; 3 values each radius since we have (mean,up,down)
static const int nsizes = 6;

float SizeBinss[nsizes];
double kmin = 5.;
double kmax = 550.; // 550.01
double kbinWdith = 5.;
int kBins = int((kmax - kmin) / kbinWdith);
// int kBins = 1;

// int kBins = 110; // here the number of bins in the i*5 exportFile loop in mathematica
int rBins = 999; // the more the better, anyway it will stop on its own.

// int Ref0Bins = 10;
// int Imf0Bins = 6;
std::vector<double> VectorValue_Ref0 = {{0.1, 0.2, 0.3, 0.4, 0.5}};
std::vector<double> VectorValue_Imf0 = {{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}};
int Ref0Bins = VectorValue_Ref0.size();
int Imf0Bins = VectorValue_Imf0.size();
// SET THE FUCKING n
//================
int n = 200; // that was for small radii

// Main code
void CalcTheoCF2G_Free()
{
    // set sources WRONG R_Core
    // SizeBinss[0] = 1.28; // 0 1 2, sono mean down up della prima
    // SizeBinss[1] = 1.24;
    // SizeBinss[2] = 1.32; // 3 4 5, della seconda

    // SizeBinss[3] = 3.25;
    // SizeBinss[4] = 3.21;
    // SizeBinss[5] = 3.29;

    SizeBinss[0] = 1.19; // 0 1 2, sono mean down up della prima
    SizeBinss[1] = 1.15;
    SizeBinss[2] = 1.24; // 3 4 5, della seconda

    SizeBinss[3] = 3.16;
    SizeBinss[4] = 3.12;
    SizeBinss[5] = 3.21;

    TObjArray *Og = new TObjArray();
    static const int gmax = 999;
    TGraph *g[gmax];
    int i = 0;
    double Value_Ref0 = 0.;
    double Value_Imf0 = 0.;

    // Setting the output file
    TFile *fout;
    // fout = new TFile("TheoCF_XiPiFree2G_Math.root", "recreate");
    fout = new TFile("TheoCF_XiPiFree2G_Test_NewRCore.root", "recreate");

    for (unsigned iRe = 0; iRe < Ref0Bins; iRe++)
    {

        Value_Ref0 = VectorValue_Ref0[iRe];
        // cout << "Value_Ref0 = " << Value_Ref0 << endl;

        for (unsigned iIm = 0; iIm < Imf0Bins; iIm++)
        {

            Value_Imf0 = VectorValue_Imf0[iIm];

            // cout << "Value_Imf0 = " << Value_Imf0 << endl;

            for (int ii = 0; ii < nsizes; ii++)
            {
                g[ii] = new TGraph();
                g[ii] = CalcCF(SizeBinss[ii], Value_Ref0, Value_Imf0);
                g[ii]->SetName(Form("g%.2f_aRe%.1f_aIm%.1f", SizeBinss[ii], Value_Ref0, Value_Imf0));
                cout << "Graph Name" << g[ii]->GetName() << endl;
                g[ii]->Write();
            } // end loop radii
        }     // end loop Ref0
    }         // end loop Imf0




    fout->Close();
}

//_____________________________
TGraph *CalcCF(float sourceSize, float Ref0, float Imf0)
{
    // SET INPUT FOLDER
    //===============
    char *FolderName;
    // FolderName = "/Users/sartozza/cernbox/Analysis/XiPi_Oton/LednickyCoulombCodes/Free_a0";
    FolderName = "/Users/sartozza/cernbox/Analysis/XiPi_Oton/LednickyCoulombCodes/OutputMiracle";

    bool debug = false;
    FILE *InFile;
    char *FileName;
    char *cdummy = new char[512];
    std::vector<vector<double>> dCkValues;
    dCkValues.resize(0);
    float dCkValues2[kBins][rBins];

    /// uBin starting from 0!!!!!!
    for (unsigned uBin = 0; uBin < kBins; uBin++)
    {
        std::vector<double> drCkValues;
        drCkValues.resize(0);
        // read file for each k*
        // cout << "k bin = " << uBin << endl;

        FileName = Form("%s/Cky_k%d_aRe%.1f_aIm%.1f.dat", FolderName, 5 + (uBin * 5), Ref0, Imf0);
        // cout << "FileName =  " << FileName << endl;

        if (debug)
        {
            std::cout
                << " reading file " << FileName << endl;
        }

        InFile = fopen(FileName, "r");
        int CurrentLine = 0;
        int npoint = 0;
        float rval = 0.0;
        float ckVal = 0.0;
        // int NumDummyLines = nline;
        int rpoint = 0;
        while (!feof(InFile))
        { // line by line
            if (!fgets(cdummy, 511, InFile))
                continue; // get a single line into cdummy
            // if (CurrentLine < NumDummyLines){
            //     CurrentLine++;
            //    continue;
            //  }
            sscanf(cdummy, "%f %f", &rval, &ckVal);
            if (debug)
            {
                cout << " line " << CurrentLine << " read,  rval=" << rval << " ckVal=" << ckVal << endl;
            }
            float dCkval = ckVal * (1.0 / pow(4.0 * M_PI * sourceSize * sourceSize * 0.00506773123 * 0.00506773123, 3. / 2.) * exp(-rval * rval / (4.0 * sourceSize * sourceSize)));
            // 1.0 / pow((4.0 * M_PI * Rad * Rad), 3.0 / 2.0) * exp(-rad * rad / (4.0 * Rad * Rad));

            if (debug)
            {
                cout << "rpoint " << rpoint << endl;
            }
            dCkValues2[uBin][rpoint] = dCkval;
            drCkValues.push_back(dCkval);
            CurrentLine++;
            rpoint++;
        } // while EoF
        dCkValues.push_back(drCkValues);
        fclose(InFile);
    } // end loop in kBin

    TGraph *gCgen = new TGraph();
    gCgen->SetName("gCgen");
    int kpoint = 0;
    if (debug)
        cout << "============> Loop (j) over kBins, up to kBins=" << kBins << endl;
    for (int j = 0; j < kBins; j++)
    {
        if (debug)
            cout << " j=" << j << endl;
        double h = 0.2;
        float sum = 0;
        float integral = 0;
        float y[n + 2];
        if (debug)
            cout << "    ========> Loop (i) over n, up to n=" << n << endl;
        for (int i = 0; i <= n; i++)
        {
            y[i] = dCkValues2[j][i];
            if (debug)
                cout << " i=" << i << " y[i]=" << y[i] << endl;
        }
        if (debug)
            cout << "    ========> Loop (i) over n, up to n=" << n << endl;
        for (int i = 1; i < n; i++)
        {
            sum = sum + h * y[i];
            if (debug)
                cout << " i=" << i << " sum=" << sum << endl;
        }
        integral = h / 2.0 * (y[0] + y[n]) + sum;
        gCgen->SetPoint(kpoint, (j + 1) * 5, integral);
        // std::cout << "k*  " << (j + 1) * 5 << "CkVal  = " << integral << std::endl;
        if (debug)
            std::cout << "k*  " << (j + 1) * 5 << "CkVal  = " << integral << std::endl;
        kpoint++;
    }
    // delete cdummy;
    // return;
    // gCgen->Draw();
    return gCgen;
}
