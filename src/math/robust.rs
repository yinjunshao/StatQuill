use nalgebra::DMatrix;

/// Result of Winsorization on a dataset
#[derive(Debug, Clone)]
pub struct WinsorizationResult {
    /// The Winsorized data matrix (extreme values capped to quantile thresholds)
    pub winsorized_data: DMatrix<f64>,
    /// Per-column lower threshold (the value at the lower quantile)
    pub lower_thresholds: Vec<f64>,
    /// Per-column upper threshold (the value at the upper quantile)
    pub upper_thresholds: Vec<f64>,
    /// Per-column count of values that were capped (either lower or upper)
    pub capped_counts: Vec<usize>,
    /// Per-row flag: true if this row had any capped values
    pub row_has_outliers: Vec<bool>,
}

/// Result of robust scaling (median & IQR-based)
#[derive(Debug, Clone)]
pub struct RobustScaleResult {
    /// The scaled data (centered by median, scaled by IQR)
    pub scaled_data: DMatrix<f64>,
    /// Per-column median (location)
    pub column_medians: Vec<f64>,
    /// Per-column IQR (inter-quartile range = Q3 - Q1)
    pub column_iqrs: Vec<f64>,
}

/// Winsorization: cap extreme values at specified quantile thresholds.
pub fn winsorize(data: &DMatrix<f64>, lower_quantile: f64, upper_quantile: f64) -> WinsorizationResult {
    let (nrows, ncols) = data.shape();
    if nrows == 0 || ncols == 0 {
        return WinsorizationResult {
            winsorized_data: data.clone(),
            lower_thresholds: vec![0.0; ncols],
            upper_thresholds: vec![0.0; ncols],
            capped_counts: vec![0; ncols],
            row_has_outliers: vec![false; nrows],
        };
    }

    let mut winsorized = DMatrix::zeros(nrows, ncols);
    let mut lower_thresholds = Vec::with_capacity(ncols);
    let mut upper_thresholds = Vec::with_capacity(ncols);
    let mut capped_counts = Vec::with_capacity(ncols);
    let mut row_has_outliers = vec![false; nrows];

    for j in 0..ncols {
        let mut sorted: Vec<f64> = data
            .column(j)
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let (lo, hi) = if sorted.is_empty() {
            (0.0, 0.0)
        } else {
            (
                quantile_of_sorted(&sorted, lower_quantile),
                quantile_of_sorted(&sorted, upper_quantile),
            )
        };

        lower_thresholds.push(lo);
        upper_thresholds.push(hi);

        let mut capped = 0usize;
        for i in 0..nrows {
            let val = data[(i, j)];
            if !val.is_finite() {
                let median = quantile_of_sorted(&sorted, 0.5);
                winsorized[(i, j)] = median;
                capped += 1;
                row_has_outliers[i] = true;
            } else if val < lo {
                winsorized[(i, j)] = lo;
                capped += 1;
                row_has_outliers[i] = true;
            } else if val > hi {
                winsorized[(i, j)] = hi;
                capped += 1;
                row_has_outliers[i] = true;
            } else {
                winsorized[(i, j)] = val;
            }
        }
        capped_counts.push(capped);
    }

    WinsorizationResult {
        winsorized_data: winsorized,
        lower_thresholds,
        upper_thresholds,
        capped_counts,
        row_has_outliers,
    }
}

/// Compute a quantile from an already-sorted slice of f64 values.
fn quantile_of_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let q = q.clamp(0.0, 1.0);
    let idx = q * (n - 1) as f64;
    let lo_idx = idx.floor() as usize;
    let hi_idx = idx.ceil() as usize;
    let frac = idx - lo_idx as f64;
    if lo_idx >= n - 1 {
        sorted[n - 1]
    } else {
        sorted[lo_idx] + frac * (sorted[hi_idx] - sorted[lo_idx])
    }
}

/// Robust scaling: center each column by its median and scale by its IQR.
pub fn robust_scale(data: &DMatrix<f64>) -> RobustScaleResult {
    let (nrows, ncols) = data.shape();
    if nrows == 0 || ncols == 0 {
        return RobustScaleResult {
            scaled_data: data.clone(),
            column_medians: vec![0.0; ncols],
            column_iqrs: vec![1.0; ncols],
        };
    }

    let mut scaled = DMatrix::zeros(nrows, ncols);
    let mut medians = Vec::with_capacity(ncols);
    let mut iqrs = Vec::with_capacity(ncols);

    for j in 0..ncols {
        let mut sorted: Vec<f64> = data
            .column(j)
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = quantile_of_sorted(&sorted, 0.5);
        let q1 = quantile_of_sorted(&sorted, 0.25);
        let q3 = quantile_of_sorted(&sorted, 0.75);
        let iqr = (q3 - q1).max(1e-10);

        medians.push(median);
        iqrs.push(iqr);

        for i in 0..nrows {
            let val = data[(i, j)];
            if val.is_finite() {
                scaled[(i, j)] = (val - median) / iqr;
            } else {
                scaled[(i, j)] = 0.0;
            }
        }
    }

    RobustScaleResult {
        scaled_data: scaled,
        column_medians: medians,
        column_iqrs: iqrs,
    }
}

/// Full Robust Preprocessing Pipeline
#[derive(Debug, Clone)]
pub struct RobustPreprocessingResult {
    pub preprocessed: DMatrix<f64>,
    pub winsor_lower: Vec<f64>,
    pub winsor_upper: Vec<f64>,
    pub capped_counts: Vec<usize>,
    pub column_medians: Vec<f64>,
    pub column_iqrs: Vec<f64>,
}

pub fn robust_preprocess(data: &DMatrix<f64>) -> RobustPreprocessingResult {
    let winsor_result = winsorize(data, 0.05, 0.95);
    let scale_result = robust_scale(&winsor_result.winsorized_data);
    RobustPreprocessingResult {
        preprocessed: scale_result.scaled_data,
        winsor_lower: winsor_result.lower_thresholds,
        winsor_upper: winsor_result.upper_thresholds,
        capped_counts: winsor_result.capped_counts,
        column_medians: scale_result.column_medians,
        column_iqrs: scale_result.column_iqrs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_winsorize_no_outliers() {
        let data = DMatrix::from_row_slice(20, 1, &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
        ]);
        let result = winsorize(&data, 0.05, 0.95);
        // With 20 points at 5%/95% quantiles, the min/max should be capped
        assert!(result.capped_counts[0] <= 2);
        // All values should be finite and within [approx q5, approx q95]
        for i in 0..20 {
            let val = result.winsorized_data[(i, 0)];
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_robust_scale() {
        let data = DMatrix::from_row_slice(5, 1, &[1.0, 2.0, 3.0, 4.0, 100.0]);
        let result = robust_scale(&data);
        assert!((result.column_medians[0] - 3.0).abs() < 1e-10);
        assert!((result.column_iqrs[0] - 2.0).abs() < 1e-10);
        assert!((result.scaled_data[(4, 0)] - 48.5).abs() < 1e-10);
    }

    #[test]
    fn test_robust_preprocess_pipeline() {
        let data = DMatrix::from_row_slice(10, 1, &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0,
        ]);
        let result = robust_preprocess(&data);
        assert!(result.capped_counts[0] >= 1);
        for i in 0..10 {
            assert!(result.preprocessed[(i, 0)].is_finite());
        }
    }
}
