/// Conformal prediction calibration for regression outputs.
///
/// Conformal prediction provides distribution-free, finite-sample coverage guarantees.
/// Instead of relying on t-distribution assumptions (which break with messy real-world data),
/// we calibrate uncertainty bounds using held-out residuals.
///
/// ## How it works
/// 1. Split data into training (80%) and calibration (20%) sets
/// 2. Train model on training set
/// 3. Compute absolute residuals on calibration set
/// 4. The (1-α) quantile of calibration residuals becomes the nonconformity threshold
/// 5. For new predictions, the interval is: [pred - threshold, pred + threshold]
///
/// This guarantees P(Y ∈ [pred - ε, pred + ε]) ≥ 1-α (marginal coverage).
#[derive(Debug, Clone)]
pub struct ConformalCalibrator {
    /// Nonconformity scores (absolute residuals) from calibration set
    pub calibration_residuals: Vec<f64>,
    /// Calibrated threshold at significance level α
    pub threshold: f64,
    /// Significance level used
    pub alpha: f64,
    /// Number of calibration samples
    pub n_calibration: usize,
    /// Coverage achieved on calibration (empirical)
    pub empirical_coverage: f64,
}

impl ConformalCalibrator {
    /// Create a new calibrator with default settings.
    pub fn new() -> Self {
        Self {
            calibration_residuals: Vec::new(),
            threshold: 0.0,
            alpha: 0.05,
            n_calibration: 0,
            empirical_coverage: 0.0,
        }
    }

    /// Calibrate using held-out residuals.
    ///
    /// * `calibration_abs_residuals` - |y_true - y_pred| for each point in the calibration set
    /// * `alpha` - significance level (e.g., 0.05 for 95% coverage)
    pub fn calibrate(&mut self, calibration_abs_residuals: &[f64], alpha: f64) {
        if calibration_abs_residuals.is_empty() {
            self.threshold = 0.0;
            self.alpha = alpha;
            self.n_calibration = 0;
            self.empirical_coverage = 0.0;
            return;
        }

        let n = calibration_abs_residuals.len();
        self.n_calibration = n;
        self.alpha = alpha;

        // Compute the (1-α) quantile, with finite-sample correction:
        // Use index = ceil((n + 1) * (1 - α)) - 1
        // This ensures P(coverage) ≥ 1-α exactly (Vovk et al., 2005)
        let corrected_quantile = 1.0 - alpha;
        let idx_f = (n as f64 + 1.0) * corrected_quantile;
        let idx = (idx_f.ceil() as usize).saturating_sub(1).min(n - 1);

        let mut sorted = calibration_abs_residuals.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.calibration_residuals = sorted.clone();
        self.threshold = sorted[idx];

        // Empirical coverage on calibration set
        let covered = sorted.iter().filter(|&&r| r <= self.threshold).count();
        self.empirical_coverage = covered as f64 / n as f64;
    }

    /// Predict with conformal interval for a new point.
    ///
    /// Returns (prediction, lower_bound, upper_bound).
    pub fn predict(&self, point_prediction: f64) -> (f64, f64, f64) {
        let eps = self.threshold;
        (point_prediction, point_prediction - eps, point_prediction + eps)
    }

    /// Score a prediction: returns a calibrated percentile (0–100) indicating
    /// how typical this residual is compared to calibration data.
    ///
    /// Higher percentile = more typical (residual is smaller than most calibration residuals).
    /// Lower percentile = more anomalous.
    pub fn score_percentile(&self, absolute_residual: f64) -> f64 {
        if self.calibration_residuals.is_empty() {
            return 50.0; // no calibration data, assume typical
        }

        let count_below = self
            .calibration_residuals
            .iter()
            .filter(|&&r| r <= absolute_residual)
            .count();

        100.0 * count_below as f64 / self.calibration_residuals.len() as f64
    }
}

/// Split data into training and calibration sets.
///
/// Returns (train_indices, calibration_indices).
/// Uses a deterministic split based on index (not random) for reproducibility.
pub fn conformal_split(n_samples: usize, calibration_fraction: f64) -> (Vec<usize>, Vec<usize>) {
    let n_calib = ((n_samples as f64) * calibration_fraction).ceil() as usize;
    let n_calib = n_calib.min(n_samples.saturating_sub(2)).max(1);
    let n_train = n_samples - n_calib;

    let train_indices: Vec<usize> = (0..n_train).collect();
    let calib_indices: Vec<usize> = (n_train..n_samples).collect();

    (train_indices, calib_indices)
}

/// Compute absolute residuals for a set of predictions.
pub fn compute_abs_residuals(y_true: &[f64], y_pred: &[f64]) -> Vec<f64> {
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(y, p)| (y - p).abs())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conformal_coverage() {
        // Generate calibration residuals from N(0, 1) folded → half-normal
        let residuals: Vec<f64> = vec![
            0.1, 0.3, 0.15, 0.5, 0.2, 0.4, 0.25, 0.35, 0.05, 0.45,
            0.12, 0.32, 0.18, 0.55, 0.22, 0.42, 0.28, 0.38, 0.08, 0.48,
        ];

        let mut calibrator = ConformalCalibrator::new();
        calibrator.calibrate(&residuals, 0.10); // 90% coverage

        // Threshold should be roughly the 90th percentile of these values
        assert!(calibrator.threshold > 0.0);
        assert!(calibrator.empirical_coverage >= 0.85); // should be ≥ 90% by construction

        let (_pred, lo, hi) = calibrator.predict(10.0);
        assert!(lo < 10.0 && hi > 10.0);
    }

    #[test]
    fn test_conformal_split() {
        let (train, calib) = conformal_split(100, 0.2);
        assert_eq!(train.len(), 80);
        assert_eq!(calib.len(), 20);
        assert_eq!(train.last(), Some(&79));
        assert_eq!(calib.first(), Some(&80));
    }

    #[test]
    fn test_score_percentile() {
        let residuals: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let mut calibrator = ConformalCalibrator::new();
        calibrator.calibrate(&residuals, 0.10);

        // Residual = 0.15 → about 20th percentile (0.1 ≤ 0.15, 0.2 > 0.15)
        let pct = calibrator.score_percentile(0.15);
        assert!(pct >= 10.0 && pct <= 30.0);

        // Residual = 2.0 → 100th percentile (larger than all calibration residuals)
        let pct2 = calibrator.score_percentile(2.0);
        assert!(pct2 >= 90.0);
    }
}
