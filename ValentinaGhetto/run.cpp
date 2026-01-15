// Intended only to load all the necessary headers for the library and scripts
// 
// Run only from the root terminal with the following command:
// .x run.cc
// Do NOT load this file in the ROOT interpreter

#include "TSystem.h"
#include "TROOT.h"
#include <iostream>

void run() {
    gSystem->SetBuildDir("build", kTRUE);
    
    // Load GSL libraries first
    gSystem->Load("/usr/lib/x86_64-linux-gnu/libgsl.so");
    gSystem->Load("/usr/lib/x86_64-linux-gnu/libgslcblas.so");
    
    // Set include path for both compilation and dictionary generation
    gInterpreter->AddIncludePath("/home/galucia/boost_1_90_0");
    gInterpreter->AddIncludePath("/home/galucia/eigen");
    
    // Compile and load

    gROOT->ProcessLine(".L MiracleCoulomb.cpp+");
    gROOT->ProcessLine(".L main.cpp+");
    gROOT->ProcessLine(".L CalcTheoCF2G_Free.C+");
    gROOT->ProcessLine("main(0, nullptr)");
    gROOT->ProcessLine("CalcTheoCF2G_Free()");

    //gROOT->ProcessLine("compare_potentials()");
    //gROOT->ProcessLine("pd_double_gaussian()");
}
