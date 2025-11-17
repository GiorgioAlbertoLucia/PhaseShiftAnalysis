#pragma once

#include <cmath>
#include <string>
#include <TMath.h>

#include "Physics.h"

class Potential {
    public:
        virtual void setParameters(double* parameters) = 0;
        virtual void setParameterErrors(double* parameterErrors) = 0;
        virtual double eval(const double x) const = 0;
        virtual double evalWithCoulomb(const double r, const double Z1 = 0, const double Z2 = 0) const = 0;
        virtual double getParameter(const int iparam) const = 0;
        virtual double getParameterError(const int iparam) const = 0;
        virtual std::string getParameterName(const int iparam) const = 0;
        virtual int getNPars() const = 0;
};

class WoodsSaxonPotential: public Potential {
    public:
        WoodsSaxonPotential() = default;
        explicit WoodsSaxonPotential(const double V0, const double R, const double a):
            mV0(V0), mR(R), ma(a) {}

        void setParameters(double* parameters) override {
            mV0 = parameters[0];
            mR = parameters[1];
            ma = parameters[2];
        }

        void setParameterErrors(double* parameterErrors) override {
            mV0Error = parameterErrors[0];
            mRError = parameterErrors[1];
            maError = parameterErrors[2];
        }

        double eval(const double r) const override {
            return -mV0 / (1.0 + TMath::Exp((r - mR) / ma));
        }

        double getParameter(const int iparam) const override {
            switch (iparam) {
                case 0:     return mV0;
                case 1:     return mR;
                case 2:     return ma;
                default:    return -999.f;
            }
        }

        double getParameterError(const int iparam) const override {
            switch (iparam) {
                case 0:     return mV0Error;
                case 1:     return mRError;
                case 2:     return maError;
                default:    return -999.f;
            }
        }

        std::string getParameterName(const int iparam) const override {
            switch (iparam) {
                case 0:     return std::string("mV0");
                case 1:     return std::string("mR");
                case 2:     return std::string("ma");
                default:    return std::string("");
            }
        }

        int getNPars() const override { return 3; }

        double evalWithCoulomb(const double r, const double Z1 = 0, const double Z2 = 0) const override {
            double V = eval(r);
            V += coulombPotential(r, Z1, Z2);
            return V;
    }
    
    private:
        double mV0;      // Depth (MeV)
        double mR;       // Radius (fm)
        double ma;       // Diffuseness (fm)
        double mV0Error;      // Depth (MeV)
        double mRError;       // Radius (fm)
        double maError;       // Diffuseness (fm)
        
};

class GausPotential: public Potential {
    public:
        GausPotential() = default;
        explicit GausPotential(const double V0, const double sigma):
            mV0(V0), mSigma(sigma) {}

        void setParameters(double* parameters) override {
            mV0 = parameters[0];
            mSigma = parameters[1];
        }

        void setParameterErrors(double* parameterErrors) override {
            mV0Error = parameterErrors[0];
            mSigmaError = parameterErrors[1];
        }

        double eval(const double r) const override {
            return -mV0 * std::exp(- r*r / (2*mSigma*mSigma));
        }

        double getParameter(const int iparam) const override {
            switch (iparam) {
                case 0:     return mV0;
                case 1:     return mSigma;
                default:    return -999.f;
            }
        }

        double getParameterError(const int iparam) const override {
            switch (iparam) {
                case 0:     return mV0Error;
                case 1:     return mSigmaError;
                default:    return -999.f;
            }
        }

        std::string getParameterName(const int iparam) const override {
            switch (iparam) {
                case 0:     return std::string("mV0");
                case 1:     return std::string("mSigma");
                default:    return std::string("");
            }
        }

        int getNPars() const override { return 2; }

        double evalWithCoulomb(const double r, const double Z1 = 0, const double Z2 = 0) const override {
            double V = eval(r);
            V += coulombPotential(r, Z1, Z2);
            return V;
    }
    
    private:
        double mV0;      // Depth (MeV)
        double mSigma;   // Radius (fm)
        double mV0Error;      // Depth (MeV)
        double mSigmaError;   // Radius (fm)
        
};

class DoubleGausPotential: public Potential {
    public:
        DoubleGausPotential() = default;
        explicit DoubleGausPotential(const double V0A, const double sigmaA, const double V0B, const double sigmaB):
            mV0A(V0A), mSigmaA(sigmaA), mV0B(V0B), mSigmaB(sigmaB) {}

        void setParameters(double* parameters) override {
            mV0A = parameters[0];
            mSigmaA = parameters[1];
            mV0B = parameters[2];
            mSigmaB = parameters[3];
        }

        void setParameterErrors(double* parameterErrors) override {
            mV0AError = parameterErrors[0];
            mSigmaAError = parameterErrors[1];
            mV0BError = parameterErrors[2];
            mSigmaBError = parameterErrors[3];
        }

        double eval(const double r) const override {
            //return - mV0A * std::exp(- r*r / (2*mSigmaA*mSigmaA)) - mV0B * std::exp(- r*r / (2*mSigmaB*mSigmaB));
            return - mV0A * std::exp(- r*r / (mSigmaA*mSigmaA)) - mV0B * std::exp(- r*r / (mSigmaB*mSigmaB));
        }

        double getParameter(const int iparam) const override {
            switch (iparam) {
                case 0:     return mV0A;
                case 1:     return mSigmaA;
                case 2:     return mV0B;
                case 3:     return mSigmaB;
                default:    return -999.f;
            }
        }

        double getParameterError(const int iparam) const override {
            switch (iparam) {
                case 0:     return mV0AError;
                case 1:     return mSigmaAError;
                case 2:     return mV0BError;
                case 3:     return mSigmaBError;
                default:    return -999.f;
            }
        }

        std::string getParameterName(const int iparam) const override {
            switch (iparam) {
                case 0:     return std::string("mV0A");
                case 1:     return std::string("mSigmaA");
                case 2:     return std::string("mV0B");
                case 3:     return std::string("mSigmaB");
                default:    return std::string("");
            }
        }

        int getNPars() const override { return 4; }

        double evalWithCoulomb(const double r, const double Z1 = 0, const double Z2 = 0) const override {
            double V = eval(r);
            V += coulombPotential(r, Z1, Z2);
            return V;
    }
    
    private:
        double mV0A;      // Depth (MeV)
        double mSigmaA;   // Radius (fm)
        double mV0B;      // Depth (MeV)
        double mSigmaB;   // Radius (fm)
        double mV0AError;      // Depth (MeV)
        double mSigmaAError;   // Radius (fm)
        double mV0BError;      // Depth (MeV)
        double mSigmaBError;   // Radius (fm)
        
};
