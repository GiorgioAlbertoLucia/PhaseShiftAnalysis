#include <iostream>

#include <TH1F.h>
#include <TSystem.h>
#include <TROOT.h>

void test()
{
    std::cout << "Hello world" << std::endl;
    TH1F hist("", "", 100, 0, 100);
    std::cout << hist.GetNbinsX() << std::endl;

    gSystem->mkdir("build", kTRUE);
    gSystem->SetBuildDir("build", kTRUE);

    gSystem->SetIncludePath("-isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk");
    
    gSystem->AddIncludePath("-I/opt/homebrew/Cellar/gsl/2.8/include/");
    gSystem->AddIncludePath("-I/Users/glucia/Projects/CATS/phemto/phase_shift/include/");

    gSystem->Load("/opt/homebrew/Cellar/gsl/2.8/lib/libgsl.dylib");
    gSystem->Load("/opt/homebrew/Cellar/gsl/2.8/lib/libgslcblas.dylib");
    
    gSystem->AddLinkedLibs("-L/opt/homebrew/Cellar/gsl/2.8/lib/ -lgsl -lgslcblas");
    gSystem->AddDynamicPath("/opt/homebrew/Cellar/gsl/2.8/lib/");
    
    gROOT->ProcessLine(".L PotentialFromPhaseShiftsDimitar.cpp+");
}
