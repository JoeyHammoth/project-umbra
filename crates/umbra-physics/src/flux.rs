//! Tabulated integral and differential muon intensity.
//!
//! `Φ(>E, θ)` is precomputed offline on a `(log10 E, cos θ)` grid by the table
//! generator (spec §6.2) and interpolated bilinearly in log space at runtime.
//! The tables are versioned assets, so this type is deliberately plain data.
//!
//! # Table layout
//!
//! `integral` and `diff` are row-major with `log10_e_min` as the slow index and
//! `cos_theta` as the fast index, so a single ray's lookup touches two adjacent
//! cache lines.

/// A precomputed muon intensity table.
///
/// No accessor is implemented yet: the interpolation kernel and its derivative
/// arrival with the table generator ticket, and their values are validated
/// against the published Gaisser/Guan curves before use (spec §2.3).
#[derive(Debug, Clone, PartialEq)]
pub struct FluxTable {
    /// Version string of the generating table, e.g. `gaisser-guan-2015.v1`.
    pub version: String,
    /// Ascending `log10(E_min / GeV)` axis.
    pub log10_e_min: Vec<f32>,
    /// Ascending `cos θ` axis.
    pub cos_theta: Vec<f32>,
    /// `Φ(>E, θ)` in cm⁻² s⁻¹ sr⁻¹.
    pub integral: Vec<f32>,
    /// `I(E, θ) = -∂Φ/∂E` in cm⁻² s⁻¹ sr⁻¹ GeV⁻¹.
    pub diff: Vec<f32>,
}

impl FluxTable {
    /// Number of grid entries.
    pub fn len(&self) -> usize {
        self.integral.len()
    }

    /// Whether the table holds no values.
    pub fn is_empty(&self) -> bool {
        self.integral.is_empty()
    }

    /// Whether the axes and value buffers are mutually consistent.
    pub fn is_consistent(&self) -> bool {
        let expected = self.log10_e_min.len() * self.cos_theta.len();
        expected != 0 && self.integral.len() == expected && self.diff.len() == expected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consistency_requires_matching_shapes() {
        let table = FluxTable {
            version: "test".into(),
            log10_e_min: vec![0.0, 1.0],
            cos_theta: vec![0.5, 0.75, 1.0],
            integral: vec![0.0; 6],
            diff: vec![0.0; 6],
        };
        assert!(table.is_consistent());
        assert_eq!(table.len(), 6);
    }

    #[test]
    fn empty_table_is_not_consistent() {
        let table = FluxTable {
            version: "test".into(),
            log10_e_min: vec![],
            cos_theta: vec![],
            integral: vec![],
            diff: vec![],
        };
        assert!(table.is_empty());
        assert!(!table.is_consistent());
    }
}
