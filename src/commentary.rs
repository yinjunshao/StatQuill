use crate::math;
use crate::parser;
use nalgebra::DMatrix;

/// Generate local commentary when OpenRouter is unavailable or not configured.
///
/// This provides meaningful, data-driven insights without requiring an external API.
/// Commentary is generated from:
///   - Strongest correlations between numeric columns
///   - Outlier counts and distributions
///   - Model selection diagnostics
///   - Prediction intervals and confidence levels
pub struct LocalCommentary;

impl LocalCommentary {
    /// Generate a complete commentary report as markdown-formatted text.
    pub fn generate(
        parser: &parser::DataParser,
        numeric_df: &DMatrix<f64>,
        diagnostics: &math::model_selection::DataDiagnostics,
        selected_model: &math::model_selection::ModelType,
    ) -> String {
        let mut report = String::new();

        // ── Executive Summary ──
        report.push_str("## Executive Summary\n\n");
        let n_numeric = parser.numeric_columns.len();
        let n_categorical = parser.columns.iter()
            .filter(|c| c.dtype == parser::ColumnType::Categorical)
            .count();

        report.push_str(&format!(
            "This dataset contains **{} rows** across **{} columns** ({} numeric, {} categorical).\n\n",
            parser.rows.len(),
            parser.headers.len(),
            n_numeric,
            n_categorical,
        ));

        let missing_total: usize = parser.columns.iter().map(|c| c.missing_count).sum();
        if missing_total > 0 {
            let pct = 100.0 * missing_total as f64 / (parser.rows.len() * parser.headers.len()) as f64;
            report.push_str(&format!(
                "**Data quality**: {}% of cells are empty. ",
                format!("{:.1}", pct)
            ));
            if pct > 10.0 {
                report.push_str("Missingness is high — predictions should be treated with caution.\n\n");
            } else {
                report.push_str("Missingness is moderate and has been imputed.\n\n");
            }
        } else {
            report.push_str("**Data quality**: No missing values detected.\n\n");
        }

        // ── Model Selection Summary ──
        report.push_str("## Model Selection\n\n");
        match selected_model {
            math::model_selection::ModelType::AutoRegressive { target_column, differencing_order, is_stationary, adf_statistic, .. } => {
                report.push_str("**Selected**: Auto-Regressive (AR) Time Series Model\n\n");
                report.push_str(&format!("- Target: `{}`\n", target_column));
                report.push_str(&format!("- Differencing: d = {}\n", differencing_order));
                if *is_stationary {
                    report.push_str(&format!("- Stationarity: ✓ Stationary (ADF τ = {:.3}, p < 0.05)\n", adf_statistic));
                } else {
                    report.push_str(&format!("- Stationarity: ✗ Non-stationary (ADF τ = {:.3}) — use with caution\n", adf_statistic));
                }
            }
            math::model_selection::ModelType::Seasonal { target_column, seasonal_period, seasonal_confidence, multiplicative, .. } => {
                report.push_str("**Selected**: Holt-Winters Seasonal Model\n\n");
                report.push_str(&format!("- Target: `{}`\n", target_column));
                report.push_str(&format!("- Seasonal period: {} observations\n", seasonal_period));
                report.push_str(&format!("- Seasonality confidence: {:.0}%\n", seasonal_confidence * 100.0));
                report.push_str(&format!("- Type: {}\n", if *multiplicative { "Multiplicative" } else { "Additive" }));
            }
            math::model_selection::ModelType::Classification { target_column, num_classes, is_binary, classes, .. } => {
                report.push_str("**Selected**: Multinomial Classification\n\n");
                report.push_str(&format!("- Target: `{}`\n", target_column));
                if *is_binary {
                    report.push_str(&format!("- Type: Binary classification ({} classes)\n", num_classes));
                } else {
                    report.push_str(&format!("- Type: Multi-class ({} classes: {})\n", num_classes, classes.join(", ")));
                }
            }
            math::model_selection::ModelType::MatrixRegression { has_multicollinearity, feature_count, sample_count, .. } => {
                report.push_str("**Selected**: Multivariate Linear Regression\n\n");
                report.push_str(&format!("- Features: {} numeric predictors\n", feature_count));
                report.push_str(&format!("- Samples: {} observations\n", sample_count));
                if *has_multicollinearity {
                    report.push_str("- ⚠ Multicollinearity detected — coefficients may be unstable; ridge regularization applied.\n");
                } else {
                    report.push_str("- ✓ No multicollinearity issues detected.\n");
                }
            }
            math::model_selection::ModelType::SafeFallback { reason } => {
                report.push_str("**Status**: No predictive model selected\n\n");
                report.push_str(&format!("- Reason: {}\n", reason));
            }
        }
        report.push('\n');

        // ── Key Correlations ──
        report.push_str("## Key Correlations\n\n");
        if numeric_df.ncols() >= 2 {
            let (cov, cols, _corr) = math::covariance::CovarianceEngine::compute(numeric_df);
            let n = cols.len();
            if n > 0 {
                let sd: Vec<f64> = (0..n).map(|j| cov[(j, j)].sqrt()).collect();
                let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
                for i in 0..n {
                    for j in (i + 1)..n {
                        if sd[i] * sd[j] > 1e-15 {
                            let r = cov[(i, j)] / (sd[i] * sd[j]);
                            if r.abs() > 0.3 {
                                pairs.push((i, j, r));
                            }
                        }
                    }
                }
                pairs.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());

                if pairs.is_empty() {
                    report.push_str("No strong correlations (|r| > 0.3) found between numeric columns.\n\n");
                } else {
                    report.push_str("| Variable A | Variable B | Correlation | Strength |\n");
                    report.push_str("|---|---|---|---|\n");
                    for (i, j, r) in pairs.iter().take(10) {
                        let a = parser.numeric_columns.get(*i).cloned()
                            .unwrap_or_else(|| format!("col{}", i));
                        let b = parser.numeric_columns.get(*j).cloned()
                            .unwrap_or_else(|| format!("col{}", j));
                        let strength = if r.abs() > 0.9 { "Very Strong" }
                            else if r.abs() > 0.7 { "Strong" }
                            else if r.abs() > 0.5 { "Moderate" }
                            else { "Weak" };
                        let direction = if *r > 0.0 { "↑" } else { "↓" };
                        report.push_str(&format!(
                            "| `{}` | `{}` | {:.3} {} | {} |\n",
                            crate::display::trunc(&a, 15),
                            crate::display::trunc(&b, 15),
                            r.abs(),
                            direction,
                            strength,
                        ));
                    }
                    report.push('\n');

                    if pairs.iter().any(|(_, _, r)| r.abs() > 0.9) {
                        let count = pairs.iter().filter(|(_, _, r)| r.abs() > 0.9).count();
                        report.push_str(&format!(
                            "**⚠ {} highly correlated pairs detected** (|r| > 0.9). "
                        , count));
                        report.push_str("These columns carry nearly identical information. Consider removing one from each pair to improve model stability.\n\n");
                    }
                }
            }
        } else {
            report.push_str("Not enough numeric columns for correlation analysis.\n\n");
        }

        // ── Outlier Analysis ──
        report.push_str("## Outlier Analysis\n\n");
        if diagnostics.has_extreme_outliers {
            report.push_str(&format!(
                "**{} extreme outliers detected** across all numeric columns.\n\n",
                diagnostics.outlier_count
            ));
            report.push_str("Outliers were identified using the [Q1 − 3×IQR, Q3 + 3×IQR] criterion and capped at the 5th/95th percentiles during preprocessing.\n\n");
            report.push_str("**Impact**: These extreme values can pull regression lines and inflate error estimates. Winsorization has been applied to reduce their influence, but predictions near these extremes will be less reliable than those within the central 90% of the data.\n\n");
        } else {
            report.push_str("✓ No extreme outliers detected. The data is well-behaved within the normal range.\n\n");
        }

        // ── Data Shape Assessment ──
        report.push_str("## Data Shape Assessment\n\n");

        // Check for time series indicators
        if parser.has_time_data() || diagnostics.has_time_column {
            report.push_str("- **Time dimension present**: The data contains a time axis. Consider exploring time-series patterns (trends, seasonality) beyond point predictions.\n");
        }

        // Sample size
        if diagnostics.n_samples < 30 {
            report.push_str(&format!(
                "- **Small sample size** (n = {}): Statistical estimates have wider confidence intervals. "
            , diagnostics.n_samples));
            report.push_str("Collecting more data would improve reliability.\n");
        } else if diagnostics.n_samples > 1000 {
            report.push_str(&format!(
                "- **Large sample size** (n = {}): Statistical estimates are stable and reliable.\n",
                diagnostics.n_samples
            ));
        }

        // Categorical columns
        if n_categorical > 0 {
            report.push_str(&format!(
                "- **{} categorical column(s)** present: These have been encoded and included as features, "
            , n_categorical));
            report.push_str("improving the model's ability to capture group-level differences.\n");
        }

        report.push('\n');

        // ── Practical Recommendations ──
        report.push_str("## Recommendations\n\n");

        match selected_model {
            math::model_selection::ModelType::AutoRegressive { .. } => {
                report.push_str("1. **Validate with hold-out data**: Reserve the last 20% of observations and compare predicted vs actual.\n");
                report.push_str("2. **Consider seasonality**: If your data has weekly/monthly cycles, a seasonal model (Holt-Winters) may outperform plain AR.\n");
                report.push_str("3. **Monitor residual autocorrelation**: If residuals show patterns, increase the AR order or add differencing.\n");
            }
            math::model_selection::ModelType::Seasonal { .. } => {
                report.push_str("1. **Validate seasonal pattern**: Check that the detected period matches domain knowledge (e.g., 12 for monthly, 4 for quarterly).\n");
                report.push_str("2. **Monitor residual patterns**: If seasonality persists in residuals, try multiplicative instead of additive.\n");
                report.push_str("3. **Use short forecast horizons**: Seasonal models are most accurate for 1-2 periods ahead.\n");
            }
            math::model_selection::ModelType::Classification { .. } => {
                report.push_str("1. **Check class balance**: If one class dominates, consider class weights or resampling.\n");
                report.push_str("2. **Evaluate with accuracy + per-class precision**: Overall accuracy can be misleading for imbalanced data.\n");
                report.push_str("3. **Use probability thresholds**: The predicted class is the most likely; review all probabilities for borderline cases.\n");
            }
            math::model_selection::ModelType::MatrixRegression { has_multicollinearity, .. } => {
                report.push_str("1. **Check feature importance**: Focus on predictors with the largest absolute coefficients.\n");
                if *has_multicollinearity {
                    report.push_str("2. **Reduce multicollinearity**: Remove one variable from each highly correlated pair to stabilize coefficients.\n");
                    report.push_str("3. **Try ridge regression** (already applied): The regularization mitigates but doesn't eliminate collinearity issues.\n");
                } else {
                    report.push_str("2. **Validate predictions**: Compare against a held-out subset to estimate real-world accuracy.\n");
                }
            }
            math::model_selection::ModelType::SafeFallback { .. } => {
                report.push_str("1. **Add more data**: The current dataset is insufficient for predictive modeling.\n");
                report.push_str("2. **Reformat data**: Ensure the file has at least 2 numeric columns and 10+ rows.\n");
                report.push_str("3. **Check column types**: Make sure numeric columns are formatted as numbers, not text.\n");
            }
        }

        report.push('\n');
        report.push_str("---\n");
        report.push_str("*Generated locally by StatQuill. No external AI was used for this analysis.*\n");

        report
    }

    /// Generate a short status-line commentary (1-2 lines).
    pub fn generate_short(
        diagnostics: &math::model_selection::DataDiagnostics,
        selected_model: &math::model_selection::ModelType,
    ) -> String {
        match selected_model {
            math::model_selection::ModelType::AutoRegressive { .. } => {
                if diagnostics.has_time_column {
                    "Time series detected: using AR model with stationarity checks.".to_string()
                } else {
                    "Auto-regressive model fitted; check residual diagnostics.".to_string()
                }
            }
            math::model_selection::ModelType::Seasonal { seasonal_period, seasonal_confidence, .. } => {
                format!(
                    "Seasonal model (Holt-Winters) selected with period={} (confidence: {:.0}%).",
                    seasonal_period, seasonal_confidence * 100.0
                )
            }
            math::model_selection::ModelType::Classification { num_classes, is_binary, .. } => {
                if *is_binary {
                    "Binary classification model selected (logistic regression).".to_string()
                } else {
                    format!("Multi-class classification model selected ({} classes, softmax + cross entropy).", num_classes)
                }
            }
            math::model_selection::ModelType::MatrixRegression { has_multicollinearity, .. } => {
                if *has_multicollinearity {
                    format!(
                        "Multivariate regression fitted with ridge regularization ({} collinear pairs detected).",
                        diagnostics.collinear_pair_count
                    )
                } else {
                    "Multivariate regression fitted successfully — no multicollinearity issues.".to_string()
                }
            }
            math::model_selection::ModelType::SafeFallback { reason } => {
                format!("No predictive model available — {}", reason)
            }
        }
    }
}
