use nalgebra::DMatrix;

/// Plausibility score for a fully filled row under the fitted model.
#[derive(Debug, Clone)]
pub struct RowPlausibility {
    /// Joint log-likelihood (higher = more plausible)
    pub log_likelihood: f64,
    /// Calibrated percentile from conformal or empirical distribution (0–100)
    pub calibrated_percentile: f64,
    /// Which threshold was crossed (if any)
    /// < 5% → anomaly, < 1% → severe anomaly
    pub anomaly_flag: AnomalyLevel,
    /// Per-field contribution to the anomaly score (positive = surprising)
    pub field_scores: Vec<(String, f64)>,
    /// The most suspicious fields (most negative contribution to log-likelihood)
    pub suspicious_fields: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyLevel {
    Normal,
    Suspicious,   // 1% ≤ percentile < 5%
    Anomaly,      // 0.1% ≤ percentile < 1%
    SevereAnomaly, // percentile < 0.1%
}

impl std::fmt::Display for AnomalyLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnomalyLevel::Normal => write!(f, "Normal"),
            AnomalyLevel::Suspicious => write!(f, "Suspicious (p < 5%)"),
            AnomalyLevel::Anomaly => write!(f, "Anomaly (p < 1%)"),
            AnomalyLevel::SevereAnomaly => write!(f, "Severe Anomaly (p < 0.1%)"),
        }
    }
}

/// Computes row plausibility for numeric targets using Gaussian residual likelihood.
///
/// For each numeric column, we fit a simple model:
///   - Estimate mean and standard deviation from the training data
///   - Compute the log-likelihood of the observed value under N(mean, std²)
///   - Sum across all numeric columns for the joint log-likelihood
///
/// The calibrated percentile is computed by comparing this row's total likelihood
/// against the empirical distribution of likelihoods from a held-out set.
pub struct PlausibilityEngine;

impl PlausibilityEngine {
    /// Compute plausibility for a single row against training data.
    ///
    /// * `row_values` - the values for this row (one per numeric column)
    /// * `column_names` - names of the numeric columns
    /// * `column_means` - mean of each column from training data
    /// * `column_stds` - standard deviation of each column from training data
    /// * `reference_log_liks` - log-likelihoods from a reference/held-out set (for calibration)
    pub fn score_row(
        row_values: &[f64],
        column_names: &[String],
        column_means: &[f64],
        column_stds: &[f64],
        reference_log_liks: &[f64],
    ) -> RowPlausibility {
        let n = row_values.len().min(column_names.len()).min(column_means.len()).min(column_stds.len());
        if n == 0 {
            return RowPlausibility {
                log_likelihood: 0.0,
                calibrated_percentile: 50.0,
                anomaly_flag: AnomalyLevel::Normal,
                field_scores: Vec::new(),
                suspicious_fields: Vec::new(),
            };
        }

        // Compute per-field z-scores and contributions to log-likelihood
        let mut field_scores: Vec<(String, f64)> = Vec::new();
        let mut total_loglik = 0.0;

        for j in 0..n {
            let mean = column_means[j];
            let std = column_stds[j].max(1e-10);
            let val = row_values[j];
            let z = (val - mean) / std;

            // Gaussian log-likelihood: -0.5 * ln(2π) - ln(σ) - 0.5 * ((x-μ)/σ)²
            let field_ll = -0.5 * (2.0 * std::f64::consts::PI).ln() - std.ln() - 0.5 * z * z;
            total_loglik += field_ll;

            field_scores.push((column_names[j].clone(), z.abs()));
        }

        // Calibrated percentile: what fraction of reference rows have LOWER log-likelihood?
        let percentile = if reference_log_liks.is_empty() {
            50.0
        } else {
            let count_below = reference_log_liks
                .iter()
                .filter(|&&ll| ll < total_loglik)
                .count();
            100.0 * count_below as f64 / reference_log_liks.len() as f64
        };

        let anomaly_flag = if percentile < 0.1 {
            AnomalyLevel::SevereAnomaly
        } else if percentile < 1.0 {
            AnomalyLevel::Anomaly
        } else if percentile < 5.0 {
            AnomalyLevel::Suspicious
        } else {
            AnomalyLevel::Normal
        };

        // Most suspicious fields (top 3 by absolute z-score)
        let mut sorted_fields = field_scores.clone();
        sorted_fields.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let suspicious_fields: Vec<String> = sorted_fields
            .iter()
            .take(3)
            .filter(|(_, z)| *z > 2.0)
            .map(|(name, _)| name.clone())
            .collect();

        RowPlausibility {
            log_likelihood: total_loglik,
            calibrated_percentile: percentile,
            anomaly_flag,
            field_scores,
            suspicious_fields,
        }
    }

    /// Compute reference log-likelihoods from training data for calibration.
    ///
    /// Returns a vector of log-likelihoods, one per row in the training set.
    /// These are used to calibrate the percentile of new rows.
    pub fn compute_reference_log_liks(
        data: &DMatrix<f64>,
        column_means: &[f64],
        column_stds: &[f64],
    ) -> Vec<f64> {
        let (nrows, ncols) = data.shape();
        let mut log_liks = Vec::with_capacity(nrows);

        for i in 0..nrows {
            let mut row_ll = 0.0;
            for j in 0..ncols {
                let mean = column_means[j];
                let std = column_stds[j].max(1e-10);
                let val = data[(i, j)];
                let z = (val - mean) / std;
                let field_ll = -0.5 * (2.0 * std::f64::consts::PI).ln() - std.ln() - 0.5 * z * z;
                row_ll += field_ll;
            }
            log_liks.push(row_ll);
        }

        log_liks
    }

    /// Compute column means and standard deviations from a data matrix.
    pub fn compute_column_stats(data: &DMatrix<f64>) -> (Vec<f64>, Vec<f64>) {
        let (nrows, ncols) = data.shape();
        if nrows == 0 {
            return (vec![0.0; ncols], vec![1.0; ncols]);
        }

        let means: Vec<f64> = (0..ncols).map(|j| data.column(j).mean()).collect();
        let stds: Vec<f64> = (0..ncols)
            .map(|j| {
                let mean = means[j];
                let var: f64 = (0..nrows)
                    .map(|i| {
                        let d = data[(i, j)] - mean;
                        d * d
                    })
                    .sum::<f64>()
                    / nrows as f64;
                var.sqrt().max(1e-10)
            })
            .collect();
        (means, stds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plausibility_normal_row() {
        let means = vec![10.0, 20.0];
        let stds = vec![2.0, 3.0];
        // Reference log-likelihoods from extreme/anomalous rows
        let ref_lls = vec![-100.0, -50.0, -80.0, -40.0, -60.0];

        let result = PlausibilityEngine::score_row(
            &[10.5, 19.0],
            &["x".to_string(), "y".to_string()],
            &means,
            &stds,
            &ref_lls,
        );

        // Row is close to the mean → should be plausible (high percentile)
        assert_eq!(result.anomaly_flag, AnomalyLevel::Normal);
        assert!(result.calibrated_percentile > 50.0);
    }

    #[test]
    fn test_plausibility_anomaly_row() {
        let means = vec![10.0, 20.0];
        let stds = vec![2.0, 3.0];
        // Reference log-likelihoods: all around -3 to -4
        let ref_lls: Vec<f64> = (-5..5).map(|_| -3.5).collect();

        let result = PlausibilityEngine::score_row(
            &[50.0, 20.0], // x is 20σ away
            &["x".to_string(), "y".to_string()],
            &means,
            &stds,
            &ref_lls,
        );

        assert!(result.calibrated_percentile < 1.0);
        assert!(matches!(result.anomaly_flag, AnomalyLevel::Anomaly | AnomalyLevel::SevereAnomaly));
        assert!(!result.suspicious_fields.is_empty());
    }
}
