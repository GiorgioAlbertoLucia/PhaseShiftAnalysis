// Intended only to load all the necessary headers for the library and scripts
// 
// Run only from the root terminal with the following command:
// .x run.cc
// Do NOT load this file in the ROOT interpreter

#include <iostream>
#include "TSystem.h"
#include "TROOT.h"

void load() {

    gSystem->mkdir("build", kTRUE);
    gSystem->SetBuildDir("build", kTRUE);
    
    gSystem->AddIncludePath("-I/opt/homebrew/Cellar/gsl/2.8/include/");
    gSystem->AddIncludePath("-I/Users/glucia/Projects/CATS/phemto/phase_shift/include/");
    
    gSystem->Load("/opt/homebrew/Cellar/gsl/2.8/lib/libgsl.dylib");
    gSystem->Load("/opt/homebrew/Cellar/gsl/2.8/lib/libgslcblas.dylib");
    
    gSystem->AddLinkedLibs("-L/opt/homebrew/Cellar/gsl/2.8/lib/ -lgsl -lgslcblas");
    gSystem->AddDynamicPath("/opt/homebrew/Cellar/gsl/2.8/lib/");
    
    //gROOT->ProcessLine(".L PotentialFromPhaseShiftsDimitar.cpp+");
    gROOT->ProcessLine(".L PlotExpectedWavefunctions.cpp+");

    /*
    gDebug = 3;
    Int_t error = gSystem->CompileMacro(
        "/Users/glucia/Projects/CATS/phemto/phase_shift/PotentialFromPhaseShiftsDimitar.cpp",
        "kOfg"  // k = keep, O = optimize, f = force recompile, g = debug symbols
    );
    
    if (error == 0) {
        std::cout << "Compilation successful! Build artifacts in ./build/" << std::endl;
        gROOT->ProcessLine("PotentialFromPhaseShifts()");
    } else {
        std::cerr << "Compilation failed with error code: " << error << std::endl;
    }
    */
}