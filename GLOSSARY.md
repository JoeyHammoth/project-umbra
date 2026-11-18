# Glossary

Vocabulary used in code, issues, PRs and the UI. If a term means something
specific here, it is defined here.

| Term | Meaning |
|---|---|
| **Opacity** | `X = Lρ`: density integrated along a line of sight, in g cm⁻². The quantity muography actually measures. |
| **Ray cell** | One (detector, angular bin) pair: a single row of `L`. |
| **Path-length operator** | The sparse CSR matrix `L` of path lengths `L_ij` (cm) through voxel `j` for ray cell `i`. Its adjoint `Lᵀ` is back projection. |
| **Back projection** | `Lᵀ ·`: smearing ray values across the voxels they cross. Used both to reconstruct and to scan for anomalies. |
| **Fidelity level** | `L0` Beer–Lambert toy; `L1` Gaisser flux + CSDA (default); `L2` adds multiple scattering and background. |
| **CSDA** | Continuous slowing-down approximation: `E_min(X) = (a/b)(e^{bX} − 1)` for `−dE/dX = a + bE`. |
| **Void template** | An analytic primitive (box, ellipsoid, capsule) hypothesised at a candidate centre and scale during a detection scan. |
| **Test statistic (TS)** | Twice the log-likelihood ratio between a hypothesised void and the reference structure. |
| **Local significance** | `Z = U/√I` for one candidate, using the score `U` and Fisher information `I`. Not corrected for scanning many candidates. |
| **Global significance** | Significance after correcting for the number of places the void could have been: the look-elsewhere effect. |
| **Look-elsewhere effect (LEE)** | The inflation of false positives when many candidate positions are tested. Corrected by Monte Carlo calibration of `max Z` plus Euler-characteristic extrapolation. |
| **Asimov dataset** | The dataset in which observed counts equal their expectation under `H1`. Its test statistic gives the median expected significance, `Z_A`. |
| **Nuisance parameter** | A quantity that affects counts but is not of interest: bulk density scale, flux normalisation, detector efficiency, alignment. Profiled out, never ignored. |
| **Missing wedge** | The set of directions no detector sees through a voxel. Its complement is the angular coverage rose. It is why limited-angle reconstructions elongate. |
| **PSF** | Point-spread function: how a point of density is blurred by the geometry and the prior. Computed locally per voxel by a few CG iterations. |
| **Reference structure** | The known stone geometry (`ρ_ref`) an anomaly is measured against. Voids only remove material, which is what makes the A4 prior one-sided. |
| **Mystery mode** | The game surface: a hidden chamber the visitor must locate using the same statistics as the Lab. |
| **Exposure** | Observation time `T`, in seconds. Expected counts scale linearly with it. |
| **Coupled Poisson** | Counts for any exposure obtained by binomial thinning between log-spaced checkpoints, so scrubbing the exposure slider is monotone and reversible. |
| **ComputeBus** | The task graph in the browser: content-keyed cache, priorities, cancellation, progressive results. |
