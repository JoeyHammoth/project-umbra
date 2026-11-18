# Physics

This document is the reference for the forward model. It mirrors spec §2 and
records the constants and approximations the code must reproduce. Where a number
comes from a paper, the paper wins: transcribe coefficients from the source, never
from memory, and unit-test against its published curves.

## The chain

```text
density field ρ
  → opacity            X = L ρ                     (g cm⁻²)
  → minimum energy     E_min(X) = (a/b)(e^{bX} − 1)
  → integral flux      Φ(>E_min, θ)
  → expected counts    λ_i = T ε A cos θ ΔΩ Φ(>E_min, θ) + b_i
  → observed counts    y_i ~ Poisson(λ_i)
```

A cavity lowers `X`, which lowers `E_min`, which raises `Φ`, which raises the
counts in that direction. That sign chain is the whole experiment.

## Constants (standard rock)

| Symbol | Value | Meaning |
|---|---|---|
| `a` | 2.0 × 10⁻³ GeV cm² g⁻¹ | ionisation term of `−dE/dX = a + bE` |
| `b` | 4.0 × 10⁻⁶ cm² g⁻¹ | radiative term |
| `a/b` | 500 GeV | critical energy |
| `ρ` | 2.65 g cm⁻³ | bulk density |
| `X₀` | 26.5 g cm⁻² | radiation length |

Sanity checks the tests enforce: `E_min(0) = 0`; `E_min(1000 g cm⁻²) ≈ 2.004 GeV`;
`dE_min/dX` matches a central difference to 10⁻³ relative.

## Fidelity levels

| Level | Physics | Why it exists |
|---|---|---|
| **L0** | Beer–Lambert toy: `λ = K Φ₀(θ) e^{−X/Λ} + b` | Fast baseline; classical transmission tomography is exact here |
| **L1** | Gaisser flux + CSDA range–energy | Default; reproduces the ~1% transmission regime |
| **L2** | L1 + Highland multiple scattering + detector angular resolution + soft background | Realistic degradation; small voids get genuinely hard |

## Surface flux

Modified Gaisser parametrisation with the Guan et al. (2015) large-angle
correction, `E` in GeV:

```text
I(E, θ) = 0.14 [ E (1 + 3.64 / (E cos^1.29 θ*)) ]^(−2.7)
              × [ 1/(1 + 1.1 E cosθ*/115) + 0.054/(1 + 1.1 E cosθ*/850) ]

cosθ* = sqrt(cos²θ + P1² + P2 cos^P3 θ + P4 cos^P5 θ) / sqrt(1 + P1² + P2 + P4)
P1..P5 = 0.102573, −0.068287, 0.958633, 0.0407253, 0.817285
```

The bracket **multiplies** the energy (Guan et al. 2015, Eq. 3). Dividing by it
inverts the low-energy spectrum and raises the vertical flux by ~400×; the
70 m⁻² s⁻¹ sr⁻¹ anchor below is a test precisely because that mistake is
easy to make and produces a plausible-looking image rather than an error.

Anchors the tests enforce:

| Quantity | Value |
|---|---|
| Φ(>1 GeV, vertical) | 6.0 × 10⁻³ cm⁻² s⁻¹ sr⁻¹ ≈ 60 m⁻² s⁻¹ sr⁻¹ (literature ≈ 70) |
| I(1 GeV, vertical) | 2.3 × 10⁻³ cm⁻² s⁻¹ sr⁻¹ GeV⁻¹ |
| Transmission through 3.8 m of rock, above 1 GeV | ≈ 70% |
| Transmission through ~75 m, above 1 GeV | ≈ 1% |

`Φ(>E, θ)` is precomputed offline on a `(log10 E, cos θ)` grid and interpolated
bilinearly in log space at runtime. Two coefficients and an effective-angle
definition decide the entire count scale, so the table generator must be
validated against the paper's published curves before any scenario is trusted.

## Derivatives (why gradients are exact)

From the fundamental theorem of calculus:

```text
∂Φ(>E, θ)/∂E = −I(E, θ)
dE_min/dX    = a e^{bX}
∂λ_i/∂X_i    = −K_i I(E_min(X_i), θ_i) a e^{bX_i}
```

These are closed form, so L-BFGS-B, steepest descent, PDHG and the Fisher maps all
run on exact gradients — no finite differences, no autodiff dependency.

## Multiple scattering (L2)

Highland:

```text
θ₀ = (13.6 MeV / (β c p)) √(x/X₀) [1 + 0.038 ln(x/X₀)]
```

Implemented as an energy-weighted Gaussian convolution of expected counts across
neighbouring angular bins. The kernel is separable and symmetric, so its transpose
is itself — which keeps the adjoint exact and cheap.

## Documented approximations

These are known, deliberate and must be repeated in the Methods page:

1. **2-D slice mode** treats rays as lying in the slice plane (a prism); zenith
   dependence is retained, out-of-plane integration is omitted. 3-D removes this.
2. **Point or small detectors**: area enters through `K_i`; finite plate geometry
   optionally via 4–9 jittered sub-rays per bin.
3. **CSDA without range straggling** at L0/L1; stochastic losses only in the
   event-level Monte Carlo (A12).
4. **Homogeneous stone** with optional layered density; real inhomogeneity is a
   nuisance parameter, not a modelled field.
5. **Schematic geometry** for historical monuments — every scenario is labelled as
   such and none predicts any real structure.

## Detector model

`Detector` carries pose, area, efficiency, angular binning, angular resolution and
background rate. Acceptance for one ray cell is `ε A cos θ ΔΩ`; the `K_i/T` part is
stored per ray cell as `k_scale`, so exposure multiplies at evaluation time and
time scrubbing never rebuilds the system.

## References

- T. K. Gaisser, *Cosmic Rays and Particle Physics*.
- M. Guan et al., "A parametrization of the cosmic-ray muon flux at sea-level"
  (2015) — the large-angle correction.
- D. E. Groom, N. V. Mokhov, S. I. Striganov, "Muon stopping power and range
  tables" (2001) — tabulated ranges used by the alternative to CSDA.
- K. Morishima et al. (2017) — the ScanPyramids Big Void.
- S. Procureur et al. (2023) — the North Face Corridor.
