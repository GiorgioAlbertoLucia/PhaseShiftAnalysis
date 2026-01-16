// Intended only to load all the necessary headers for the library and scripts
// 
// Run only from the root terminal with the following command:
// .x run.cc
// Do NOT load this file in the ROOT interpreter

#include "TSystem.h"
#include "TROOT.h"
#include <iostream>

void run() {

    //gSystem->SetFlagsOpt("-O2");
    //gSystem->SetFlagsDebug("-g");
    //gSystem->SetIncludePath("-I/opt/homebrew/include -I/Users/glucia/Projects/CATS/DLM_glucia/install/include");
    //
    //gSystem->SetBuildDir("build", kTRUE);
    //
    //// Let ROOT handle standard paths
    //gSystem->AddDynamicPath("/opt/homebrew/lib");
    //gSystem->AddIncludePath("/opt/homebrew/include");
    //
    //gSystem->AddDynamicPath("/Users/glucia/Projects/CATS/DLM_glucia/install/CMake");
    //gInterpreter->AddIncludePath("/Users/glucia/Projects/CATS/DLM_glucia/install/include");
    //
    //// Load libraries
    //gSystem->Load("libCATS");
    //gSystem->Load("libgsl");
    //gSystem->Load("libgslcblas");
    //gSystem->Load("libflint");
    
    // Compile
    //gROOT->ProcessLine(".L MiracleCoulomb.cpp+");
    //gROOT->ProcessLine(".L main.cpp+");
    gROOT->ProcessLine(".L CalcTheoCF2G_Free.C+");
    //gROOT->ProcessLine("main(0, nullptr)");
    gROOT->ProcessLine("CalcTheoCF2G_Free()");
}