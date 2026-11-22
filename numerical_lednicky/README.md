# Correlation Function Calculator

C++ code that computes theoretical femtoscopic correlation functions for a two-particle system with:

- the exact Coulomb wave function (confluent hypergeometric function, via FLINT/Arb);
- the s-wave strong interaction in the Lednický–Lyuboshitz approach (complex scattering length, effective range);
- optionally, one **resonant partial wave** (e.g. the Δ(1232) in π–N), including its interference with the incident Coulomb wave;
- for π–d, π–t and π–³He, optionally the **Δ(1232) contribution** (Δ → πN with the nucleon then coalescing into the nucleus), as separate scattering and decay terms.

The source is a spherical Gaussian. The calculation is parallelised with OpenMP.

Built-in configurations: Ξ–π, p–³He, π±–d, π±–t, π±–³He, and π±–p / π±–n with the Δ(1232).

---

## Project structure

```
.
├── Config.h/cpp                  # Physical constants, CalculationConfig, built-in configurations
├── CoulombWaveFunction.h/cpp     # Wave function, s-wave amplitude, resonant partial wave
├── CorrelationCalculator.h/cpp   # k/r grids, source integration, I/O
├── DeltaFeedDown.h/cpp           # Delta(1232) term for pi-nucleus systems
├── main.cpp                      # Command-line interface
├── macros/
│   ├── map_piN_to_piA.py         # Python version of the pi-N -> pi-nucleus mapping (for PiN outputs)
│   ├── delta_for_system.py       # Python version of the Delta combination (for PiN outputs)
│   ├── compare_cf.py
│   ├── convert_to_hist_GeV.py
│   └── draw_cf_with_bands.py
├── CMakeLists.txt
└── README.md
```

---

## Building

Dependencies:

- ROOT (6.x or later)
- GSL (GNU Scientific Library)
- FLINT (with Arb), GMP
- CATS / DLM library
- OpenMP

`CMakeLists.txt` contains hard-coded install paths for CATS (`CATS_BASE_DIR`) and FLINT (`FLINT_ROOT`). Edit them to match your machine before building.

```bash
cmake -B build -DGSL_ROOT_DIR=/path/to/your/gsl/installation
cmake --build build
```

The executable is `build/correlation_calc`.

---

## Usage

```bash
OMP_NUM_THREADS=<N_JOBS> ./correlation_calc [options]
```

The program does **not** create output folders. Make them before running (see the folder names in each configuration below).

### Options

| Option | Description |
|---|---|
| `--system <XiPi\|PHe\|PiD\|PiT\|PiHe3\|PiN>` | Built-in configuration (default: `PHe`) |
| `--pion-charge <+1\|-1>` | Pion charge for `PiD`, `PiT`, `PiHe3`, `PiN` (default: −1) |
| `--nucleon-charge <1\|0>` | Nucleon for `PiN`: 1 = proton, 0 = neutron (default: 1) |
| `--mode <generate\|root\|both\|delta>` | Compute the C(k, r) table, build the ROOT file from it, or both (default: `both`). `delta`: only the Δ(1232) file (π–nucleus systems, takes seconds) |
| `--output-folder <path>` | Folder for the intermediate C(k, r) table |
| `--output-root <file>` | Name of the final ROOT file |
| `--dat` | Store the intermediate table as `.dat` files instead of one ROOT file |
| `--particle1-mass`, `--particle1-charge`, `--particle2-mass`, `--particle2-charge` | Override the particles of the chosen configuration (MeV, units of e) |
| `--resonance` | `PiD`, `PiT`, `PiHe3`: also write the Δ(1232) file. `PiN`: add the Δ P₃₃ wave to the pair |
| `--res-regulator <fm>` | Small-r regulator of the resonant wave (Δ scattering term), default 1 fm |
| `--res-decay-regulator <fm>` | `PiD`, `PiT`, `PiHe3`: small-r regulator of the Δ decay term, default 0.2 fm |
| `--delta-decay-radius <fm>` | `PiD`, `PiT`, `PiHe3`: Gaussian R of the Δ decay term, default 0.42 fm |
| `--sigma-q <MeV/c>` | `PiD`, `PiT`, `PiHe3`: internal nucleon momentum per component (defaults 60 for d, 90 for t and ³He) |
| `--w-proton <w>`, `--w-neutron <w>` | `PiD`, `PiT`, `PiHe3`: override the proton/neutron channel weights |
| `--res-decay-only` | `PiN`: keep only the resonance-decay term |
| `--res-elasticity <x>` | `PiN`: override Γ_el/Γ_tot (isospin weight) of the resonance |
| `--help` | Show the help message |

`--system` is read first, so all other options override the chosen configuration.

### Examples

```bash
# p-3He with its default configuration
OMP_NUM_THREADS=8 ./correlation_calc

# pi- d, s-wave strong interaction + Coulomb
mkdir -p D/dat_Pim D/output
OMP_NUM_THREADS=8 ./correlation_calc --system PiD --pion-charge -1

# pi- d, s wave + Coulomb, and the Delta(1232) terms in a second file
OMP_NUM_THREADS=8 ./correlation_calc --system PiD --pion-charge -1 --resonance

# Delta(1232) terms only, e.g. to scan the regulator
OMP_NUM_THREADS=8 ./correlation_calc --system PiHe3 --pion-charge +1 --mode delta --res-regulator 0.5

# pi+ p with the Delta(1232), scattering term
mkdir -p P/dat_Pip P/output
OMP_NUM_THREADS=8 ./correlation_calc --system PiN --pion-charge +1 --nucleon-charge 1 --resonance

# Only rebuild the ROOT file from an existing C(k, r) table
./correlation_calc --system PiD --pion-charge -1 --mode root
```

---

## Built-in configurations

All configurations live in `Config.cpp` under `namespace DefaultConfigs`.

### Pion–nucleus systems: `PiD`, `PiT`, `PiHe3`

- **Source radii:** 0.50–1.20 fm in steps of 0.05 fm, then 1.3–4.0 fm in steps of 0.1 fm (43 values).
- **Momentum grid:** k* = 1–551 MeV/c in steps of 1 MeV/c.
- **Effective range:** zero-range approximation (d₀ = 0) everywhere.
- **Coulomb-only reference:** the first entry of every scattering-length list is f₀ = 0.
- **Output:** `./<D|T|He3>/dat_<Pip|Pim>/` and `./<D|T|He3>/output/TheoCF_<Pip|Pim><D|T|He3>.root`.
- **Δ(1232) term** (with `--resonance` or `--mode delta`): `./<D|T|He3>/output/TheoCF_<Pip|Pim><D|T|He3>_Delta_rsca<..>_rdec<..>_sq<..>.root`. See [Δ(1232) contribution](#δ1232-contribution-to-πnucleus-correlations).

Scattering lengths (fm; f₀ > 0 means attraction):

| System | f₀ entries | Source |
|---|---|---|
| π±d | 0; −0.0353 + 0.0088i (nominal); −0.0358, −0.0348 (±1σ on Re); −0.0382 + 0.0092i | PSI pionic deuterium, Strauch et al., EPJA 47 (2011) 88. The last entry is the value used by ALICE (arXiv:2504.02333). The strong part is the same for π⁺ and π⁻ (isoscalar target). |
| π⁻t | 0; −0.1801; −0.1879; −0.1723 | Theory, arXiv:2606.23331, Table 9.1 (no measurement exists) |
| π⁺t | 0; +0.0919; +0.0841; +0.0997 | Same as above |
| π⁻³He | 0; +0.0918; +0.0840; +0.0996; +0.0580; +0.0792 | Theory as above; the last two are pionic-atom values (Schwanner; Abela) |
| π⁺³He | 0; −0.1803; −0.1881; −0.1725 | Theory as above |

Unit conversion: 1/m_π = ħc / 139.570 MeV = 1.4138 fm.

### Pion–nucleon with the Δ(1232): `PiN`

- **Particles:** π± with a proton (`--nucleon-charge 1`) or a neutron (`--nucleon-charge 0`). Neutral pairs (π–n) are treated without Coulomb.
- **s wave:** off (f₀ = 0). Edit `GetPiNConfig` to add π–N scattering lengths if needed.
- **Δ(1232) in the P₃₃ wave:** Breit–Wigner with M = 1230.55 MeV, Γ = 112.2 MeV, spin weight 2/3 (J = 3/2 out of J = 1/2, 3/2).
- **Isospin weight**, stored as the elasticity: 1 for π⁺p and π⁻n, 1/3 for π⁻p and π⁺n.
- **Output:** `./<P|N>/dat_<Pip|Pim>/` and `./<P|N>/output/TheoCF_<Pip|Pim><P|N>[_Res...].root`.

### Others

- `XiPi`: Ξ–π.
- `PHe`: p–³He. Note that this configuration divides its source sizes by √2.

---

## Resonant partial wave

Enabled with `--resonance`. Because the source is spherical, partial waves with different L do not mix after the angular average. The resonant wave therefore replaces the free Coulomb wave of order L outside the interaction region:

```
psi_L = F_L + T(k) [ G_L + i F_L ]
T(k)  = x * (Gamma(k)/2) / (M - sqrt(s) - i Gamma(k)/2)
Gamma(k) = Gamma_0 * (k/k_R) * B(k)/B(k_R),   B(k) = (k R_B)^2 / (1 + (k R_B)^2)
```

Here F_L and G_L are the regular and irregular Coulomb functions (GSL), x is the elasticity or isospin weight, and R_B = 1 fm is the Blatt–Weisskopf radius. The term added to the angle-integrated |ψ|² is

```
(2L+1) * w_J * ( |psi_L|^2 - F_L^2 ) / rho^2
```

It contains the interference between the incident and the scattered wave, which shifts the correlation peak below the resonance momentum (arXiv:2607.04351).

With `--res-decay-only`, only the resonance-decay term `(2L+1) w_J |T (G_L + i F_L)|^2 / rho^2` is kept. It contains no Coulomb or plane-wave part and is meant to be used with a small source (≈0.4 fm).

**Small-r regulator.** The outer-region form is not valid inside the interaction range, and G_L diverges as r → 0. It is regulated as

```
G_L -> G_L (1 - exp(-r / r_reg))^(2L+1)
```

For the Δ scattering term, r_reg ≈ 1 fm reproduces the ≈140 MeV/c peak of arXiv:2607.04351. Smaller values move the peak towards the Δ momentum (≈227 MeV/c). Treat r_reg as a systematic, and scan it with `--res-regulator`.

Resonance runs get a suffix such as `_Res_x1.00_rreg1.00` or `_ResDecay_x1.00_rreg0.20` on the ROOT file and on the intermediate table, so they never overwrite the s-wave-only results.

---

## Δ(1232) contribution to π–nucleus correlations

In π–nucleus pairs, the Δ peak comes from Δ → πN with the nucleon then coalescing into the nucleus. For `PiD`, `PiT` and `PiHe3` this is computed directly by `DeltaFeedDown`, enabled with `--resonance` or `--mode delta`:

```bash
mkdir -p D/output
OMP_NUM_THREADS=8 ./correlation_calc --system PiD --pion-charge -1 --resonance
```

This writes a second ROOT file next to the usual one. The π–nucleus wave function itself is unchanged: there is no π–nucleus resonance.

### Contents of the Δ file

| Graph | Meaning |
|---|---|
| `gDeltaSca_R<size>` | Δ scattering term mapped to k*_πA, one per source size of the scan |
| `gDeltaDec` | Δ decay term mapped to k*_πA (source R = 0.42 fm) |
| `gPiN_<p\|n>_DeltaSca_R<size>`, `gPiN_<p\|n>_DeltaDec` | The π–p and π–n curves before mapping (vs k*_πN), for checks |

The full model is

```
C(k*) = C_Coulomb+s-wave(k*) + λ_sca · ΔC_sca(k*) + λ_dec · ΔC_dec(k*)
```

where λ_sca and λ_dec are fit parameters.

### What is computed

For each π–N channel, with the P₃₃ wave ψ₁ = F₁ + T (G₁ + i F₁) (see [Resonant partial wave](#resonant-partial-wave)):

```
scattering:  ΔC_sca(k) = ∫ S_R(r)    · 3 w_J (|ψ₁|² − F₁²) / ρ²        (the Coulomb part cancels exactly)
decay:       ΔC_dec(k) = ∫ S_Rdec(r) · 3 w_J |T (G₁ + i F₁)|² / ρ²
```

- **Breit–Wigner:** M = 1230.55 MeV, Γ = 112.2 MeV (arXiv:2607.04351), w_J = 2/3.
- **Isospin weight in T:** 1 for π⁺p and π⁻n, 1/3 for π⁻p and π⁺n.
- **Cost:** only GSL Coulomb functions are needed, so a full radius scan takes seconds.

**Channels and weights**, normalised to 1. You can override them with `--w-proton` and `--w-neutron`.

| | proton taken up | neutron taken up |
|---|---|---|
| π⁺ | Δ⁺⁺ → π⁺p (BR 1) | Δ⁺ → π⁺n (BR 1/3) |
| π⁻ | Δ⁰ → π⁻p (BR 1/3) | Δ⁻ → π⁻n (BR 1) |

- **Decay term:** w = BR × Z/A (proton) or BR × N/A (neutron). This assumes the Δ charge states are produced equally.
- **Scattering term:** w = Z/A or N/A, since the isospin factor is already in T.

**Mapping.** The nucleon carries p_A/A plus an internal momentum q, which is drawn from an isotropic Gaussian with width `--sigma-q` (MeV/c per component, fixed random seed):

```
k*_piN / mu_piN = k*_piA / mu_piA - q / m_N
```

With q = 0 this gives k*_πA ≈ 1.07 k*_πN for d and ≈ 1.10 k*_πN for A = 3. Jacobians and phase-space weights are neglected, so the curves are shapes, and their amplitudes are the λ's.

**Typical result** (R = 1.2 fm, default regulators):

| | scattering term | decay term |
|---|---|---|
| peak | ≈ 148–153 MeV/c | ≈ 221–227 MeV/c |
| dip | ≈ 246–255 MeV/c | — |

The scattering peak position depends on `--res-regulator`. Scan it as a systematic.

### Python macros (π–N outputs)

`macros/delta_for_system.py` and `macros/map_piN_to_piA.py` do the same mapping and combination starting from separate `PiN` runs. They are kept for cross-checks and for cases where you want to change the π–N input by hand (e.g. add a π–N s wave):

```bash
python3 map_piN_to_piA.py input.root output.root --A 2 --sigma-q 60 [--additive]
python3 delta_for_system.py --system d --pion-charge +1 --proton ... --neutron ... \
    [--proton-ref ... --neutron-ref ... | --decay] --sigma-q 60 --output out.root
```

### Combining the interaction and the Δ terms

`macros/combine_cf_delta.py` adds the Δ terms to the Coulomb + s-wave correlation function:

```
C(k*) = 1 + λ_FSI · (C_FSI(k*) − 1) + λ_sca · ΔC_sca(k*) + λ_dec · ΔC_dec(k*)
```

```bash
# one combination
python3 macros/combine_cf_delta.py --cf ../pid/output/piminus_d.root \
    --delta ../pid/output/piminus_d_Delta_rsca1.00_rdec0.20_sq60.root \
    -o ../pid/output/piminus_d_combined.root --lambda-sca 0.5 --lambda-dec 0.5

# grid of combinations, two radii, nominal scattering length only, also as TH1D
python3 macros/combine_cf_delta.py --cf piminus_d.root --delta piminus_d_Delta_rsca1.00_rdec0.20_sq60.root \
    -o grid.root --lambda-sca 0:1:0.25 --lambda-dec 0 0.5 1 --radii 1.2 2.0 --match aRe-0.0353 --hist

# see what the files contain
python3 macros/combine_cf_delta.py --cf piminus_d.root --delta piminus_d_Delta_rsca1.00_rdec0.20_sq60.root --list
```

| Option | Meaning |
|---|---|
| `--cf <file>` | Main ROOT file of the run (`--output-root`) |
| `--delta <file>` | Δ file of the same run (`--resonance` or `--mode delta`) |
| `-o, --output <file>` | Output ROOT file (overwritten) |
| `--lambda-fsi L…` | Strength(s) of C_FSI − 1 (default 1) |
| `--lambda-sca L…` | Strength(s) of the Δ scattering term (default 1) |
| `--lambda-dec L…` | Strength(s) of the Δ decay term (default 1) |
| `--radii R…` | Only these source sizes, as in the graph names |
| `--match TEXT…` | Only CF graphs whose name contains all the strings, e.g. `aRe-0.0353` |
| `--hist` | Also write each curve as a TH1D (`h<R>_aRe…`) |
| `--components` | Also write the input curves to a `components/` directory |
| `--list` | List the graphs in both files and exit |

λ values are listed (`0 0.5 1`) or given as `start:stop:step` ranges (stop included).

- **Output layout:** every combination of the λ values goes to its own directory, `lfsi<λ_FSI>_lsca<λ_sca>_ldec<λ_dec>/`. Inside it, the curves keep the CF graph names `g<R>_aRe<…>_aIm<…>_r<d0>`.
- **Pairing:** each CF graph is paired with `gDeltaSca_R<R>` of the same R. The single `gDeltaDec` is added to all of them.
- **Momentum grid:** the Δ curves are interpolated at the CF k* points.
- **Errors:** the script stops with a message if the files look swapped or if nothing matches the filters. It warns about CF graphs that have no Δ partner.

The Δ curves are shapes with an arbitrary scale, so λ = 1 has no special meaning. To fit, use the three components with free λ's. Use this macro for plots, for templates at fitted values, or for λ scans. Combine files from the same run only, since both depend on the source-radius convention (the √2 conversion in `PionNucleusSourceScan()`).

----------

## Adding a new system

Add a function to `Config.cpp`, declare it in `Config.h`, and add a branch to the `--system` selection in `main.cpp`:

```cpp
namespace DefaultConfigs {
    CalculationConfig GetMySystem() {
        CalculationConfig config;

        config.particle1 = ParticleProperties(mass1, charge1);   // MeV, units of e
        config.particle2 = ParticleProperties(mass2, charge2);

        config.sourceSizes           = {...};   // fm
        config.realScatteringLengths = {...};   // fm, f0 > 0 attractive
        config.imagScatteringLengths = {...};   // fm
        config.effectiveRanges       = {...};   // fm, same length as the lists above

        // optional resonant wave (enabled with --resonance)
        config.resonance.L = 1;
        config.resonance.mass = ...;            // MeV
        config.resonance.width = ...;           // MeV
        config.resonance.elasticity = ...;      // Gamma_el / Gamma_tot or isospin weight
        config.resonance.spinWeight = ...;      // (2J+1) / ((2s1+1)(2s2+1)(2L+1))

        config.outputFolder     = "./output/dat/";
        config.outputRootFolder = "./output/";
        config.outputRootFile   = "MySystem.root";
        return config;
    }
}
```

The three scattering-parameter lists are read entry by entry (entry i of each list forms one parameter set).

---

## Output

### Intermediate C(k, r) table

- **Default:** one ROOT file, `<outputFolder>/Cky_intermediate[_<resonance tag>].root`, with one 2D histogram (k × r) per parameter set, named `hCky_aRe<re>_aIm<im>_r<d0>`.
- **With `--dat`:** text files `Cky_k<k>_aRe<re>_aIm<im>_r<d0>.dat` with tab-separated columns of r (fm) and C(k, r). k is written without decimals, so keep `kBinWidth` ≥ 1 MeV/c with this option.

### ROOT file

`<outputRootFolder>/<outputRootFile>` holds one TGraph per combination of source size and scattering parameters:

```
g<size>_aRe<re>_aIm<im>_r<d0>
```

Here `<d0>` is the effective range. Scattering lengths are written with 4 decimals, and source sizes and effective ranges with 2.

---

## Physics background

```
C(k) = ∫ d³r S(r) |ψ(k, r)|²,     S(r) ∝ exp(-r² / (4R²))
```

The Gaussian above is the code's source convention, so R is the single-particle radius. If your radii refer to exp(−r²/2R²), divide them by √2. The p–³He configuration does this.

ψ is the Coulomb wave function plus the Lednický s-wave scattered wave:

```
f_c(k)^-1 = 1/f0 + d0 k²/2 - (2/a_c) h(η) - i k A_c(η)
```

Here a_c is the Bohr radius of the pair, η = 1/(k a_c), and A_c is the Gamow factor. The integration is done over the angle between k and r (Simpson rule) and over r with the Gaussian weight.

- **Sign convention:** f₀ > 0 means attraction. This is the same sign as the pionic-atom scattering lengths, and the opposite of the nuclear convention used e.g. for p–³He.
- **Validity:** the outer-region (asymptotic) wave functions are not valid inside the interaction range. Results for source radii ≲ 1 fm, and the resonance terms in particular, depend on that approximation.
