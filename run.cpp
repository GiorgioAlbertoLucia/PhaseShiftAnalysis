// Intended only to load all the necessary headers for the library and scripts
// 
// Run only from the root terminal with the following command:
// .x run.cc
// Do NOT load this file in the ROOT interpreter

#include "TSystem.h"
#include "TROOT.h"

void run() {
    
    gSystem->SetBuildDir("build", kTRUE);
    
    // Add include paths
    gSystem->AddIncludePath("-I/opt/homebrew/Cellar/gsl/2.8/include/");
    gSystem->AddIncludePath("-I/Users/glucia/Projects/CATS/phemto/phase_shift/include/");
    
    // Load GSL libraries first
    gSystem->Load("/opt/homebrew/Cellar/gsl/2.8/lib/libgsl.dylib");
    gSystem->Load("/opt/homebrew/Cellar/gsl/2.8/lib/libgslcblas.dylib");
    
    // Compile and load
    gROOT->ProcessLine(".L lednicky_integration.cpp+");
    gROOT->ProcessLine("lednicky_integration()");
}