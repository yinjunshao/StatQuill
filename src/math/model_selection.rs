use nalgebra::DMatrix;

/// The type of model recommended for a dataset
#[derive(Debug, Clone, PartialEq)]
pub enum ModelType {
    /// Auto-Regressive (AR) model for time series
    /// Indicates a single numeric target with sufficient temporal structure
    AutoRegressive {
        /// Name of the target column
        target_column: String,
        /// Differencing order applied (0, 1, or 2)
        differencing_order: usize,
        /// Whether the series was found stationary after differencing
        is_stationary: bool,
        /// ADF test statistic on the final (differenced) series
        adf_statistic: f64,
        /// Number of observations in the series
        series_length: usize,
    },
    /// Seasonal model (Holt-Winters) for time series with clear seasonal patterns
    Seasonal {
        /// Name of the target column
        target_column: String,
        /// Detected seasonal period
        seasonal_period: usize,
        /// Confidence in the seasonal detection (0-1)
        seasonal_confidence: f64,
        /// Whether seasonality is multiplicative
        multiplicative: bool,
        /// Series length
        series_length: usize,
    },
    /// Multivariate Regression for cross-sectional / multi-axis tables
    /// Multiple numeric columns with at least one feature and one target
    MatrixRegression {
        /// Whether multicollinearity was detected among features
        has_multicollinearity: bool,
        /// Number of numeric feature columns available
        feature_count: usize,
        /// Whether the data suggests a prediction scenario (one unknown among knowns)
        is_prediction_scenario: bool,
        /// Number of observations
        sample_count: usize,
    },
    /// Classification for categorical target columns
    Classification {
        /// Name of the target column
        target_column: String,
        /// Number of classes
        num_classes: usize,
        /// Whether binary classification
        is_binary: bool,
        /// Class labels
        classes: Vec<String>,
    },
    /// Fallback: simple statistics only (no predictive model is appropriate)
    SafeFallback {
        /// Why no predictive model was selected
        reason: String,
    },
}

/// Diagnostic summary used by the model selection router
#[derive(Debug, Clone)]
pub struct DataDiagnostics {
    /// Number of observations
    pub n_samples: usize,
    /// Number of numeric columns
    pub n_numeric_cols: usize,
    /// Whether a time (datetime) column was detected
    pub has_time_column: bool,
    /// Name of the time column, if any
    pub time_column_name: Option<String>,
    /// Number of columns with > 10% missing values
    pub high_missing_cols: usize,
    /// Total count of missing cells in the numeric matrix
    pub total_missing_cells: usize,
    /// Whether multicollinearity was detected (any pairwise r > 0.9)
    pub has_multicollinearity: bool,
    /// Count of highly collinear pairs (r > 0.9)
    pub collinear_pair_count: usize,
    /// Whether extreme outliers were detected (any value beyond [Q1-3*IQR, Q3+3*IQR])
    pub has_extreme_outliers: bool,
    /// Count of outlier cells detected
    pub outlier_count: usize,
    /// Whether the numeric matrix is square-ish (suggests covariance/correlation matrix)
    pub is_square_matrix: bool,
    /// Maximum pairwise correlation magnitude
    pub max_correlation: f64,
}

impl DataDiagnostics {
    /// Create diagnostics from a raw numeric matrix and parser metadata
    ///
    /// `numeric_df`: the raw numeric data matrix (rows × columns)
    /// `has_time_column`: whether the parser detected a datetime column
    /// `time_column_name`: the name of the time column
    /// `numeric_column_names`: names of the numeric columns
    /// `corr_matrix`: optional pre-computed correlation matrix
    pub fn from_data(
        numeric_df: &DMatrix<f64>,
        has_time_column: bool,
        time_column_name: Option<String>,
        corr_matrix: Option<&DMatrix<f64>>,
    ) -> Self {
        let (n_samples, n_numeric_cols) = numeric_df.shape();

        // Count missing values (NaN or Inf)
        let total_missing = count_missing(numeric_df);
        let high_missing_cols = count_high_missing_columns(numeric_df, 0.10);

        // Detect outliers: any value outside [Q1 - 3*IQR, Q3 + 3*IQR]
        let (has_outliers, outlier_count) = detect_extreme_outliers(numeric_df);

        // Multicollinearity check from correlation matrix
        let (has_mc, mc_count, max_corr) = if let Some(corr) = corr_matrix {
            let n = corr.nrows();
            let mut pairs = 0usize;
            let mut max_abs = 0.0f64;
            for i in 0..n {
                for j in (i + 1)..n {
                    let r = corr[(i, j)].abs();
                    if r > max_abs {
                        max_abs = r;
                    }
                    if r > 0.9 {
                        pairs += 1;
                    }
                }
            }
            (pairs > 0, pairs, max_abs)
        } else {
            // Quick calculation for small matrices
            let mut pairs = 0usize;
            let mut max_abs = 0.0f64;
            if n_numeric_cols >= 2 {
                for i in 0..n_numeric_cols {
                    for j in (i + 1)..n_numeric_cols {
                        if let Some(r) = pearson_r(numeric_df, i, j) {
                            let abs_r = r.abs();
                            if abs_r > max_abs {
                                max_abs = abs_r;
                            }
                            if abs_r > 0.9 {
                                pairs += 1;
                            }
                        }
                    }
                }
            }
            (pairs > 0, pairs, max_abs)
        };

        DataDiagnostics {
            n_samples,
            n_numeric_cols,
            has_time_column,
            time_column_name,
            high_missing_cols,
            total_missing_cells: total_missing,
            has_multicollinearity: has_mc,
            collinear_pair_count: mc_count,
            has_extreme_outliers: has_outliers,
            outlier_count,
            is_square_matrix: n_samples == n_numeric_cols && n_samples > 1,
            max_correlation: max_corr,
        }
    }
}

/// The Model Selection Router
///
/// Examines the data diagnostics and decides which model pipeline to run.
/// This is the "automated router block" that replaces manual user decisions
/// about whether to run AR, Matrix Regression, or fall back to summary stats.
///
/// ## Decision Logic
///
/// 1. **Insufficient Data** → SafeFallback
///    - < 5 rows or 0 numeric columns or < 2 numeric columns with ≥ 2 needed
///
/// 2. **Time Series → AR Model** (if data has a time column and sufficient structure)
///    - Has datetime column + at least 1 numeric column
///    - At least 10 observations for meaningful AR fitting
///    - Not a square matrix (then it's probably a covariance table, not a series)
///
/// 3. **Cross-Sectional → Matrix Regression**
///    - ≥ 2 numeric columns (at least one feature + one target)
///    - At least 5 observations
///
/// 4. **Fallback → Simple Statistics**
///    - Everything else
pub fn select_model(diagnostics: &DataDiagnostics) -> ModelType {
    // Guard: insufficient data for any modeling
    if diagnostics.n_samples < 3 {
        return ModelType::SafeFallback {
            reason: format!(
                "Only {} observations — need at least 3 for any analysis.",
                diagnostics.n_samples
            ),
        };
    }

    if diagnostics.n_numeric_cols == 0 {
        return ModelType::SafeFallback {
            reason: "No numeric columns detected. Cannot perform quantitative modeling."
                .to_string(),
        };
    }

    // Check if this looks like a time series
    let has_time = diagnostics.has_time_column && diagnostics.time_column_name.is_some();

    if has_time && diagnostics.n_samples >= 10 && diagnostics.n_numeric_cols >= 1 {
        // It has a time axis — prefer AR modeling on the time series targets
        // But check: if it's a square matrix, it's probably a correlation/covariance table
        if diagnostics.is_square_matrix && diagnostics.max_correlation < 0.3 {
            return ModelType::SafeFallback {
                reason: "Square matrix with low correlations — appears to be an identity/random matrix, not time series data.".to_string(),
            };
        }

        // Good candidate for AR
        return ModelType::AutoRegressive {
            target_column: diagnostics
                .time_column_name
                .clone()
                .unwrap_or_else(|| "value".to_string()),
            differencing_order: 0, // will be determined at fit time
            is_stationary: false,  // will be determined by ADF
            adf_statistic: 0.0,    // will be determined by ADF
            series_length: diagnostics.n_samples,
        };
    }

    // Cross-sectional / tabular data: Matrix Regression
    if diagnostics.n_numeric_cols >= 2 && diagnostics.n_samples >= 5 {
        // Warn about multicollinearity but still proceed
        let has_mc = diagnostics.has_multicollinearity;
        return ModelType::MatrixRegression {
            has_multicollinearity: has_mc,
            feature_count: diagnostics.n_numeric_cols - 1, // one column is the implicit target
            is_prediction_scenario: diagnostics.n_numeric_cols >= 2,
            sample_count: diagnostics.n_samples,
        };
    }

    // Single numeric column with enough data — could still do basic stats
    if diagnostics.n_numeric_cols == 1 && diagnostics.n_samples >= 5 {
        return ModelType::SafeFallback {
            reason: "Only one numeric column — need at least two for regression. Consider using the time-series mode if this is temporal data.".to_string(),
        };
    }

    ModelType::SafeFallback {
        reason: "Data does not meet the requirements for AR modeling or matrix regression. Check that you have enough numeric columns and observations.".to_string(),
    }
}

/// Generate a human-readable recommendation string from the model selection
pub fn explain_selection(model: &ModelType, diagnostics: &DataDiagnostics) -> String {
    let mut explanation = String::new();

    explanation.push_str(&format!(
        "Data: {} rows × {} numeric columns",
        diagnostics.n_samples, diagnostics.n_numeric_cols
    ));

    if diagnostics.has_time_column {
        if let Some(ref tc) = diagnostics.time_column_name {
            explanation.push_str(&format!("\nTime column detected: '{}'", tc));
        }
    }

    if diagnostics.total_missing_cells > 0 {
        explanation.push_str(&format!(
            "\nMissing values: {} cells imputed via median ({:.1}% of data)",
            diagnostics.total_missing_cells,
            100.0 * diagnostics.total_missing_cells as f64
                / (diagnostics.n_samples.max(1) * diagnostics.n_numeric_cols.max(1)) as f64
        ));
    }

    if diagnostics.has_extreme_outliers {
        explanation.push_str(&format!(
            "\nOutliers detected: {} values Winsorized at 5th/95th percentile",
            diagnostics.outlier_count
        ));
    }

    if diagnostics.has_multicollinearity {
        explanation.push_str(&format!(
            "\n⚠ Multicollinearity: {} highly correlated pairs (r > 0.9) — regression coefficients may be unstable",
            diagnostics.collinear_pair_count
        ));
    }

    match model {
        ModelType::AutoRegressive { differencing_order, is_stationary, adf_statistic, series_length, .. } => {
            explanation.push_str("\n\n→ Selected: Auto-Regressive (AR) Time Series Model");
            explanation.push_str(&format!("\n  Series length: {}", series_length));

            if *differencing_order > 0 {
                explanation.push_str(&format!("\n  Differencing: d={} applied to achieve stationarity", differencing_order));
            }
            if *is_stationary {
                explanation.push_str(&format!("\n  ADF test: stationary (τ = {:.3}, p < 0.05)", adf_statistic));
            } else {
                explanation.push_str(&format!("\n  ADF test: non-stationary (τ = {:.3}) — predictions may be unreliable", adf_statistic));
            }
        }
        ModelType::Seasonal { seasonal_period, seasonal_confidence, multiplicative, series_length, .. } => {
            explanation.push_str("\n\n→ Selected: Holt-Winters Seasonal Model");
            explanation.push_str(&format!("\n  Series length: {}", series_length));
            explanation.push_str(&format!("\n  Seasonal period: {} observations", seasonal_period));
            explanation.push_str(&format!("\n  Seasonality confidence: {:.0}%", seasonal_confidence * 100.0));
            explanation.push_str(&format!("\n  Type: {}", if *multiplicative { "Multiplicative" } else { "Additive" }));
        }
        ModelType::Classification { num_classes, is_binary, target_column, classes, .. } => {
            explanation.push_str("\n\n→ Selected: Multinomial Classification");
            explanation.push_str(&format!("\n  Target: '{}'", target_column));
            if *is_binary {
                explanation.push_str(&format!("\n  Type: Binary classification ({} classes)", num_classes));
            } else {
                explanation.push_str(&format!("\n  Type: Multi-class classification ({} classes: {})", num_classes, classes.join(", ")));
            }
        }
        ModelType::MatrixRegression { has_multicollinearity, feature_count, sample_count, .. } => {
            explanation.push_str("\n\n→ Selected: Multivariate Matrix Regression");
            explanation.push_str(&format!("\n  Features: {} numeric columns as predictors", feature_count));
            explanation.push_str(&format!("\n  Samples: {}", sample_count));
            if *has_multicollinearity {
                explanation.push_str("\n  Ridge regularization applied to stabilize estimates");
            }
        }
        ModelType::SafeFallback { reason } => {
            explanation.push_str("\n\n→ Selected: Safe Fallback (no predictive model)");
            explanation.push_str(&format!("\n  Reason: {}", reason));
        }
    }

    explanation
}

// ── Diagnostic helpers ──

fn count_missing(data: &DMatrix<f64>) -> usize {
    data.iter().filter(|v| !v.is_finite()).count()
}

fn count_high_missing_columns(data: &DMatrix<f64>, threshold: f64) -> usize {
    let (_nrows, ncols) = data.shape();
    let nrows = data.nrows();
    if nrows == 0 {
        return 0;
    }
    let mut count = 0usize;
    for j in 0..ncols {
        let missing = data.column(j).iter().filter(|v| !v.is_finite()).count();
        if missing as f64 / nrows as f64 > threshold {
            count += 1;
        }
    }
    count
}

fn detect_extreme_outliers(data: &DMatrix<f64>) -> (bool, usize) {
    let (_nrows, ncols) = data.shape();
    let nrows = data.nrows();
    if nrows < 4 {
        return (false, 0);
    }

    let mut total_outliers = 0usize;

    for j in 0..ncols {
        let mut sorted: Vec<f64> = data
            .column(j)
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if sorted.len() < 4 {
            continue;
        }

        let q1 = quantile_sorted(&sorted, 0.25);
        let q3 = quantile_sorted(&sorted, 0.75);
        let iqr = q3 - q1;
        if iqr < 1e-15 {
            continue; // constant column
        }

        let lower_fence = q1 - 3.0 * iqr;
        let upper_fence = q3 + 3.0 * iqr;

        for i in 0..nrows {
            let val = data[(i, j)];
            if val.is_finite() && (val < lower_fence || val > upper_fence) {
                total_outliers += 1;
            }
        }
    }

    (total_outliers > 0, total_outliers)
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
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

/// Compute Pearson correlation between two columns of a DMatrix
fn pearson_r(data: &DMatrix<f64>, col_a: usize, col_b: usize) -> Option<f64> {
    let n = data.nrows();
    if n < 2 {
        return None;
    }

    let mut sum_a = 0.0;
    let mut sum_b = 0.0;
    let mut count = 0usize;

    for i in 0..n {
        let a = data[(i, col_a)];
        let b = data[(i, col_b)];
        if a.is_finite() && b.is_finite() {
            sum_a += a;
            sum_b += b;
            count += 1;
        }
    }

    if count < 2 {
        return None;
    }

    let mean_a = sum_a / count as f64;
    let mean_b = sum_b / count as f64;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for i in 0..n {
        let a = data[(i, col_a)];
        let b = data[(i, col_b)];
        if a.is_finite() && b.is_finite() {
            let da = a - mean_a;
            let db = b - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-15 {
        None
    } else {
        Some(cov / denom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    fn make_diag(n_samples: usize, n_cols: usize) -> DataDiagnostics {
        let data = DMatrix::from_element(n_samples, n_cols, 1.0);
        DataDiagnostics::from_data(&data, false, None, None)
    }

    #[test]
    fn test_select_model_insufficient_data() {
        let diag = make_diag(2, 2);
        let model = select_model(&diag);
        match model {
            ModelType::SafeFallback { ref reason } => {
                assert!(reason.contains("3"), "Expected '3' in reason, got: {}", reason);
            }
            _ => panic!("Expected SafeFallback"),
        }
    }

    #[test]
    fn test_select_model_matrix_regression() {
        let diag = make_diag(100, 3);
        let model = select_model(&diag);
        match model {
            ModelType::MatrixRegression { .. } => {}
            _ => panic!("Expected MatrixRegression for 100×3 data"),
        }
    }

    #[test]
    fn test_select_model_single_column() {
        let diag = make_diag(50, 1);
        let model = select_model(&diag);
        match model {
            ModelType::SafeFallback { .. } => {}
            _ => panic!("Expected SafeFallback for single column"),
        }
    }

    #[test]
    fn test_data_diagnostics_creates_valid() {
        let data = DMatrix::from_row_slice(5, 3, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
            13.0, 14.0, 15.0,
        ]);
        let diag = DataDiagnostics::from_data(&data, true, Some("date".to_string()), None);
        assert_eq!(diag.n_samples, 5);
        assert_eq!(diag.n_numeric_cols, 3);
        assert!(diag.has_time_column);
        assert_eq!(diag.time_column_name, Some("date".to_string()));
        assert_eq!(diag.total_missing_cells, 0);
    }
}
