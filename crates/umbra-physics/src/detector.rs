//! Detector geometry, angular binning and acceptance.

/// Placement and orientation of a detector plate.
///
/// `normal` is the outward plate normal (the direction the detector looks along)
/// and `up` fixes the roll about that normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    /// Plate centre, in metres.
    pub position: [f32; 3],
    /// Unit outward normal.
    pub normal: [f32; 3],
    /// Unit in-plane "up" vector, orthogonal to `normal`.
    pub up: [f32; 3],
}

/// How a detector's acceptance is divided into ray cells.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AngularBinning {
    /// Azimuth × zenith rectangular bins, in radians.
    AzZen {
        az_min: f32,
        az_max: f32,
        az_bins: u32,
        zen_min: f32,
        zen_max: f32,
        zen_bins: u32,
    },
    /// A flat list of directions, used for 2-D slice mode.
    Flat2D {
        /// Lower edge of the angular range, in radians.
        min: f32,
        /// Upper edge of the angular range, in radians.
        max: f32,
        /// Number of bins.
        bins: u32,
    },
}

impl AngularBinning {
    /// Number of ray cells contributed by one detector.
    pub fn bins(&self) -> usize {
        match *self {
            AngularBinning::AzZen {
                az_bins, zen_bins, ..
            } => az_bins as usize * zen_bins as usize,
            AngularBinning::Flat2D { bins, .. } => bins as usize,
        }
    }

    /// Mean solid angle of a single bin, in steradians.
    ///
    /// For `AzZen` binning the solid angle of a bin depends on its zenith row
    /// (`Δφ (cos θ_lo − cos θ_hi)`); this returns the mean over rows, which is
    /// what acceptance normalisation uses before per-row corrections. For
    /// `Flat2D`, where the angular range is treated as an in-plane fan, it
    /// returns the angular width of a bin in radians per unit out-of-plane
    /// extent.
    pub fn mean_solid_angle_sr(&self) -> f32 {
        match *self {
            AngularBinning::AzZen {
                az_min,
                az_max,
                az_bins,
                zen_min,
                zen_max,
                zen_bins,
            } => {
                let d_az = (az_max - az_min) / az_bins.max(1) as f32;
                let band = zen_min.cos() - zen_max.cos();
                let n_bins = (az_bins.max(1) * zen_bins.max(1)) as f32;
                (d_az * band.abs()) / n_bins * (az_bins.max(1) as f32)
            }
            AngularBinning::Flat2D { min, max, bins } => (max - min) / bins.max(1) as f32,
        }
    }
}

/// A detector: a small plate with an efficiency and an angular resolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detector {
    /// Position and orientation in the scene.
    pub pose: Pose,
    /// Active area, in m².
    pub area_m2: f32,
    /// Detection efficiency in `[0, 1]`.
    pub eff: f32,
    /// Angular binning.
    pub binning: AngularBinning,
    /// Angular resolution (one sigma), in milliradians. Only used at L2.
    pub sigma_ang_mrad: f32,
    /// Soft-particle background rate, in Hz. Only used at L2.
    pub bg_rate_hz: f32,
}

impl Detector {
    /// Acceptance factor `A_d · ε_d`, in m².
    pub fn acceptance_m2(&self) -> f32 {
        self.area_m2 * self.eff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bin_counts_multiply() {
        let binning = AngularBinning::AzZen {
            az_min: 0.0,
            az_max: std::f32::consts::TAU,
            az_bins: 72,
            zen_min: 0.0,
            zen_max: std::f32::consts::FRAC_PI_2,
            zen_bins: 36,
        };
        assert_eq!(binning.bins(), 2592);
    }

    #[test]
    fn acceptance_combines_area_and_efficiency() {
        let detector = Detector {
            pose: Pose {
                position: [0.0; 3],
                normal: [0.0, 0.0, 1.0],
                up: [0.0, 1.0, 0.0],
            },
            area_m2: 0.25,
            eff: 0.8,
            binning: AngularBinning::Flat2D {
                min: 0.0,
                max: 1.0,
                bins: 10,
            },
            sigma_ang_mrad: 5.0,
            bg_rate_hz: 0.0,
        };
        assert!((detector.acceptance_m2() - 0.2).abs() < 1e-6);
        assert_eq!(detector.binning.bins(), 10);
    }
}
