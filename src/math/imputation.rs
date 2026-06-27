use nalgebra::DMatrix;

/// Result of imputation on a dataset
#[derive(Debug, Clone)]
pub struct ImputationResult {
    /// The imputed data matrix (rows × columns, all f64, no NaN)
    pub imputed_data: DMatrix<f64>,
    /// Per-column median values used for imputation
    pub column_medians: Vec<f64>,
    /// Per-column missingness counts (how many cells were imputed)
    pub missing_counts: Vec<usize>,
    /// Missingness indicator matrix: 1.0 where original was missing, 0.0 otherwise
    /// Same shape as imputed_data
    pub missingness_indicators: DMatrix<f64>,
    /// Per-row missingness flag: true if this row had any imputed values
    pub row_has_missing: Vec<bool>,
}

/// Performs median imputation on a numeric data matrix.
///
/// Strategy:
/// 1. For each column, compute the median of non-NaN values
/// 2. Replace NaN/f64::NAN with the column median
/// 3. If an entire column is NaN, fill with 0.0
/// 4. Generate "missingness indicator" columns — a binary flag for each original
///    column marking whether that cell was imputed. This preserves the information
///    that a value *was* missing, which is crucial for downstream modeling.
///
/// This is superior to mean imputation because the median is robust to outliers.
/// A single extreme value won't drag the imputed value away from the central tendency.
pub fn median_imputation(data: &DMatrix<f64>) -> ImputationResult {
    let (nrows, ncols) = data.shape();
    if nrows == 0 || ncols == 0 {
        return ImputationResult {
            imputed_data: data.clone(),
            column_medians: vec![0.0; ncols],
            missing_counts: vec![0; ncols],
            missingness_indicators: DMatrix::zeros(nrows, ncols),
            row_has_missing: vec![false; nrows],
        };
    }

    let mut imputed = DMatrix::zeros(nrows, ncols);
    let mut missing_indicators = DMatrix::zeros(nrows, ncols);
    let mut col_medians = Vec::with_capacity(ncols);
    let mut missing_counts = Vec::with_capacity(ncols);
    let mut row_has_missing = vec![false; nrows];

    for j in 0..ncols {
        // Collect non-NaN values from this column
        let mut sorted_vals: Vec<f64> = data
            .column(j)
            .iter()
            .copied()
            .filter(|v| !v.is_nan() && !v.is_infinite())
            .collect();
        sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if sorted_vals.is_empty() {
            0.0
        } else {
            median_of_sorted(&sorted_vals)
        };

        col_medians.push(median);

        let mut missing = 0usize;
        for i in 0..nrows {
            let val = data[(i, j)];
            if val.is_nan() || val.is_infinite() {
                imputed[(i, j)] = median;
                missing_indicators[(i, j)] = 1.0;
                missing += 1;
                row_has_missing[i] = true;
            } else {
                imputed[(i, j)] = val;
                missing_indicators[(i, j)] = 0.0;
            }
        }
        missing_counts.push(missing);
    }

    ImputationResult {
        imputed_data: imputed,
        column_medians: col_medians,
        missing_counts,
        missingness_indicators: missing_indicators,
        row_has_missing,
    }
}

/// Compute the median of an already-sorted slice of f64 values.
fn median_of_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Augment the imputed data matrix with missingness indicator columns.
///
/// Returns a new DMatrix where the first `ncols` columns are the imputed data
/// and the remaining columns are binary flags (0.0 or 1.0) indicating whether
/// each original cell was missing.
///
/// Only columns that actually had missing values get an indicator column.
/// This avoids blowing up the feature space with useless zero-columns.
pub fn augment_with_missingness(
    imputed: &DMatrix<f64>,
    indicators: &DMatrix<f64>,
) -> DMatrix<f64> {
    let (nrows, ncols) = imputed.shape();
    if nrows == 0 {
        return DMatrix::zeros(0, 0);
    }

    // Count how many indicator columns are non-trivial (had any missing)
    let mut indicator_cols: Vec<usize> = Vec::new();
    for j in 0..ncols {
        let has_any = (0..nrows).any(|i| indicators[(i, j)] > 0.5);
        if has_any {
            indicator_cols.push(j);
        }
    }

    if indicator_cols.is_empty() {
        return imputed.clone();
    }

    let total_cols = ncols + indicator_cols.len();
    let mut augmented = DMatrix::zeros(nrows, total_cols);

    // Copy original imputed columns
    for j in 0..ncols {
        for i in 0..nrows {
            augmented[(i, j)] = imputed[(i, j)];
        }
    }

    // Copy indicator columns
    for (k, &orig_col) in indicator_cols.iter().enumerate() {
        let dest_col = ncols + k;
        for i in 0..nrows {
            augmented[(i, dest_col)] = indicators[(i, orig_col)];
        }
    }

    augmented
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_median_imputation_no_missing() {
        let data = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = median_imputation(&data);
        assert_eq!(result.missing_counts, vec![0, 0]);
        assert!(!result.row_has_missing.iter().any(|&x| x));
        // Data should be unchanged
        for i in 0..3 {
            for j in 0..2 {
                assert!((result.imputed_data[(i, j)] - data[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_median_imputation_with_nan() {
        let data = DMatrix::from_row_slice(4, 1, &[1.0, f64::NAN, 3.0, f64::NAN]);
        let result = median_imputation(&data);
        // Median of [1, 3] = 2
        assert!((result.column_medians[0] - 2.0).abs() < 1e-10);
        assert_eq!(result.missing_counts[0], 2);
        assert!((result.imputed_data[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((result.imputed_data[(1, 0)] - 2.0).abs() < 1e-10);
        assert!((result.imputed_data[(2, 0)] - 3.0).abs() < 1e-10);
        assert!((result.imputed_data[(3, 0)] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_augment_with_missingness() {
        let imputed = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let indicators = DMatrix::from_row_slice(3, 2, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let augmented = augment_with_missingness(&imputed, &indicators);
        // Should have 2 original + 1 indicator column (only col 1 had missing)
        assert_eq!(augmented.ncols(), 3);
        assert_eq!(augmented.nrows(), 3);
        assert!((augmented[(0, 2)] - 1.0).abs() < 1e-10);
        assert!((augmented[(1, 2)] - 0.0).abs() < 1e-10);
        assert!((augmented[(2, 2)] - 1.0).abs() < 1e-10);
    }
}
