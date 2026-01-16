# Correlation Function Calculator

A refactored C++ application for calculating theoretical correlation functions for particle physics experiments, with support for Coulomb interactions.

## Project Structure

```
.
├── Config.h/cpp              # Configuration and physical constants
├── CoulombWaveFunction.h/cpp # Wave function calculations
├── CorrelationCalculator.h/cpp # Main calculation logic
├── main.cpp                  # Command-line interface
├── CMakeLists.txt           # Build configuration
└── README.md                # This file
```

## Key Improvements

### 1. **Better Separation of Concerns**
- **Config**: All configuration parameters, physical constants, and particle properties
- **CoulombWaveFunction**: Pure physics calculations for wave functions
- **CorrelationCalculator**: High-level calculation orchestration
- **main**: Command-line interface and user interaction

### 2. **Flexible Input Handling**
The program now accepts command-line arguments for customization:

```bash
./correlation_calc --mode both \
                   --output-folder ./my_output/ \
                   --output-root results.root \
                   --particle1-mass 139.57 \
                   --particle1-charge -1.0 \
                   --particle2-mass 1321.71 \
                   --particle2-charge -1.0
```

### 3. **Clean Configuration System**
- Default configurations via `DefaultConfigs::GetXiPiConfig()`
- Easy to add new particle systems
- All parameters centralized in `CalculationConfig` struct

### 4. **Improved Code Quality**
- Descriptive variable names
- Clear function responsibilities
- Proper encapsulation
- Memory management improvements
- Better error handling

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

### Basic Usage (Xi-Pi system with defaults)
```bash
./correlation_calc
```

### Generate only data files
```bash
./correlation_calc --mode generate --output-folder ./data/
```

### Generate only ROOT file (requires data files)
```bash
./correlation_calc --mode root --output-root my_results.root
```

### Custom particle system
```bash
./correlation_calc \
  --particle1-mass 493.677 \
  --particle1-charge 1.0 \
  --particle2-mass 938.272 \
  --particle2-charge 1.0 \
  --output-root KaonProton.root
```

## Configuration

To add a new particle system, edit `Config.cpp`:

```cpp
namespace DefaultConfigs {
    CalculationConfig GetMySystem() {
        CalculationConfig config;
        
        config.particle1 = ParticleProperties(mass1, charge1);
        config.particle2 = ParticleProperties(mass2, charge2);
        
        config.sourceSizes = {...};
        config.realScatteringLengths = {...};
        config.imagScatteringLengths = {...};
        
        config.outputFolder = "./output/";
        config.outputRootFile = "MySystem.root";
        
        return config;
    }
}
```

## Dependencies

- ROOT (6.x or later)
- GSL (GNU Scientific Library)
- FLINT (Fast Library for Number Theory)
- GMP (GNU Multiple Precision Arithmetic Library)
- CATS library
- DLM library

## Output

### Data Files
- Format: `Cky_k<momentum>_aRe<real>_aIm<imag>.dat`
- Content: Tab-separated columns of r (fm) and C(k,r) values

### ROOT File
- Contains TGraph objects for each combination of:
  - Source size
  - Real scattering length
  - Imaginary scattering length
- Graph naming: `g<size>_aRe<real>_aIm<imag>`

## Physics Background

This code calculates correlation functions including:
- Coulomb interactions via wave functions
- Scattering amplitude with effective range expansion
- Gamow penetration factor
- Hypergeometric function evaluation (1F1)

The correlation function is computed as:
```
C(k) = ∫ |ψ(k,r)|² S(r) dr
```

where ψ is the Coulomb wave function and S(r) is the source distribution.