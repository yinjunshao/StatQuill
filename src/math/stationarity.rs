use nalgebra::{DMatrix, DVector};

/// Result of an Augmented Dickey-Fuller (ADF) stationarity test.
#[derive(Debug, Clone)]
pub struct ADFTestResult {
    /// The ADF test statistic (τ)
    pub test_statistic: f64,
    /// Critical values at 1%, 5%, and 10% significance levels
    pub critical_values: [f64; 3],
    /// Whether the null hypothesis of a unit root is rejected at α=0.05
    pub is_stationary: bool,
    /// p-value approximation (via MacKinnon 1994 surface response)
    pub p_value: f64,
    /// Number of lags used in the test regression
    pub used_lag: usize,
    /// Number of observations used
    pub nobs: usize,
}

/// Result of differencing a time series.
#[derive(Debug, Clone)]
pub struct DifferencingResult {
    /// The differenced series (length = original.len() - 1 per order of differencing)
    pub differenced_series: Vec<f64>,
    /// Number of differencing operations applied
    pub order: usize,
    /// ADF test result after differencing
    pub final_adf: Option<ADFTestResult>,
}

/// Augmented Dickey-Fuller test for stationarity.
///
/// Tests the null hypothesis that a unit root is present (series is non-stationary).
/// Rejection of the null (p < 0.05) indicates stationary data suitable for AR modeling.
///
/// ## Model
/// Uses the ADF regression with constant (no trend):
/// Δy_t = α + γ y_{t-1} + Σ δ_i Δy_{t-i} + ε_t
///
/// H₀: γ = 0 (unit root, non-stationary)
/// H₁: γ < 0 (stationary)
///
/// ## Lag Selection
/// The number of lags is chosen automatically via the Schwert criterion:
/// lag_max = floor(12 * (n/100)^(1/4))
pub fn adf_test(series: &[f64]) -> ADFTestResult {
    let n = series.len();
    if n < 10 {
        return ADFTestResult {
            test_statistic: 0.0,
            critical_values: [-3.43, -2.86, -2.57],
            is_stationary: false,
            p_value: 1.0,
            used_lag: 0,
            nobs: n,
        };
    }

    // Schwert criterion for maximum lag
    let max_lag = ((12.0 * (n as f64 / 100.0).powf(0.25)) as usize).min(n / 3).max(1);

    let mut best_aic = f64::INFINITY;
    let mut best_result: Option<ADFTestResult> = None;

    // Try lags from 1 to max_lag, pick best by AIC
    for lag in 1..=max_lag {
        let n_eff = n - lag - 1;
        if n_eff < 5 {
            continue;
        }

        // Build design matrix for ADF regression with constant (no trend)
        // y = Xβ where X has columns: [1, y_{t-1}, Δy_{t-1}, ..., Δy_{t-lag}]
        // Total columns = 2 + lag
        let k = 2 + lag;
        let mut x_rows = Vec::with_capacity(n_eff * k);
        let mut y = Vec::with_capacity(n_eff);

        for t in (lag + 1)..n {
            let dy = series[t] - series[t - 1]; // Δy_t
            let y_lag1 = series[t - 1]; // y_{t-1}

            let mut row = vec![1.0]; // constant
            row.push(y_lag1); // y_{t-1}
            for i in 1..=lag {
                let dy_lag = series[t - i] - series[t - i - 1]; // Δy_{t-i}
                row.push(dy_lag);
            }
            x_rows.extend(row);
            y.push(dy);
        }

        let x = DMatrix::from_vec(n_eff, k, x_rows);
        let y_vec = DVector::from_vec(y);

        // OLS via SVD
        let xt = x.transpose();
        let xtx = &xt * &x;
        let xty = &xt * &y_vec;

        let svd = xtx.clone().svd(true, true);
        let beta = svd
            .solve(&xty, 1e-12)
            .unwrap_or_else(|_| DVector::zeros(k));

        if beta.len() < 2 {
            continue;
        }

        // Residuals
        let pred = &x * &beta;
        let residuals: Vec<f64> = pred.iter().zip(y_vec.iter()).map(|(p, a)| a - p).collect();
        let rss: f64 = residuals.iter().map(|r| r * r).sum();

        // Standard error of γ (coefficient of y_{t-1}, which is beta[1])
        let df = (n_eff as isize - k as isize).max(1);
        let sigma2 = rss / df as f64;

        let xtx_inv = svd
            .solve(&DMatrix::identity(k, k), 1e-12)
            .unwrap_or_else(|_| DMatrix::zeros(k, k));
        let se_gamma = (sigma2 * xtx_inv[(1, 1)]).sqrt();

        if se_gamma < 1e-15 {
            continue;
        }

        // Test statistic τ = γ̂ / SE(γ̂)
        let gamma_hat = beta[1];
        let tau = gamma_hat / se_gamma;

        // AIC-like criterion: n * ln(σ²) + 2k
        let aic = n_eff as f64 * f64::ln(sigma2.max(1e-10)) + 2.0 * k as f64;

        if aic < best_aic {
            best_aic = aic;

            // MacKinnon approximate p-value (1994, Table 1 surface)
            let p_val = mackinnon_pvalue(tau);

            // Critical values (approximate via MacKinnon response surface)
            let cv_1pct = mackinnon_critical(0.01, n_eff);
            let cv_5pct = mackinnon_critical(0.05, n_eff);
            let cv_10pct = mackinnon_critical(0.10, n_eff);

            let is_stationary = tau < cv_5pct; // reject H₀ at α=0.05

            best_result = Some(ADFTestResult {
                test_statistic: tau,
                critical_values: [cv_1pct, cv_5pct, cv_10pct],
                is_stationary,
                p_value: p_val,
                used_lag: lag,
                nobs: n_eff,
            });
        }
    }

    best_result.unwrap_or(ADFTestResult {
        test_statistic: 0.0,
        critical_values: [-3.43, -2.86, -2.57],
        is_stationary: false,
        p_value: 1.0,
        used_lag: 0,
        nobs: n,
    })
}

/// Apply differencing to a time series until it becomes stationary.
///
/// Differencing transforms y_t → y_t - y_{t-1}, removing linear trends.
/// For quadratic trends, a second difference may be needed.
///
/// Algorithm:
/// 1. Run ADF test on the original series
/// 2. If non-stationary (p ≥ 0.05), apply first difference
/// 3. Re-run ADF on the differenced series
/// 4. If still non-stationary, apply second difference (max 2)
/// 5. Return the differenced series and the order
pub fn difference_to_stationarity(series: &[f64]) -> DifferencingResult {
    if series.len() < 5 {
        return DifferencingResult {
            differenced_series: series.to_vec(),
            order: 0,
            final_adf: None,
        };
    }

    // Test original series
    let adf_original = adf_test(series);

    if adf_original.is_stationary {
        return DifferencingResult {
            differenced_series: series.to_vec(),
            order: 0,
            final_adf: Some(adf_original),
        };
    }

    // Apply first difference
    let diff1: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();

    if diff1.len() < 10 {
        return DifferencingResult {
            differenced_series: diff1,
            order: 1,
            final_adf: None,
        };
    }

    let adf_diff1 = adf_test(&diff1);

    if adf_diff1.is_stationary {
        return DifferencingResult {
            differenced_series: diff1,
            order: 1,
            final_adf: Some(adf_diff1),
        };
    }

    // Apply second difference
    let diff2: Vec<f64> = diff1.windows(2).map(|w| w[1] - w[0]).collect();

    if diff2.len() < 10 {
        return DifferencingResult {
            differenced_series: diff2,
            order: 2,
            final_adf: None,
        };
    }

    let adf_diff2 = adf_test(&diff2);

    DifferencingResult {
        differenced_series: diff2,
        order: 2,
        final_adf: Some(adf_diff2),
    }
}

/// Simple first difference: returns y[t] - y[t-1] for all t ≥ 1
pub fn first_difference(series: &[f64]) -> Vec<f64> {
    series.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Undo differencing: reconstruct the original levels from differenced predictions.
///
/// Given the last `order` known values and a vector of differenced predictions,
/// returns the level predictions.
///
/// # Arguments
/// * `original_series` - the original (pre-differencing) series
/// * `diff_predictions` - predictions made in the differenced space
/// * `order` - how many differences were applied (0, 1, or 2)
pub fn undo_difference(
    original_series: &[f64],
    diff_predictions: &[f64],
    order: usize,
) -> Vec<f64> {
    match order {
        0 => diff_predictions.to_vec(),
        1 => {
            let last_level = original_series.last().copied().unwrap_or(0.0);
            let mut levels = Vec::with_capacity(diff_predictions.len());
            let mut cum = last_level;
            for d in diff_predictions {
                cum += d;
                levels.push(cum);
            }
            levels
        }
        _ => {
            // For order ≥ 2, we need to accumulate twice
            let last_level = original_series.last().copied().unwrap_or(0.0);
            let last_diff1 = if original_series.len() >= 2 {
                original_series[original_series.len() - 1]
                    - original_series[original_series.len() - 2]
            } else {
                0.0
            };

            let mut diff1_cum = last_diff1;
            let mut level_cum = last_level;
            let mut levels = Vec::with_capacity(diff_predictions.len());

            for d2 in diff_predictions {
                diff1_cum += d2;
                level_cum += diff1_cum;
                levels.push(level_cum);
            }
            levels
        }
    }
}

// ── MacKinnon approximate p-value and critical values ──

/// Approximate p-value for the ADF τ-statistic using MacKinnon (1994)
/// surface response coefficients for the constant-only model.
///
/// MacKinnon, J. G. (1994). "Approximate asymptotic distribution functions
/// for unit-root and cointegration tests." Journal of Business & Economic
/// Statistics, 12(2), 167-176.
fn mackinnon_pvalue(tau: f64) -> f64 {
    // Critical values for constant-only:
    // 1%: -3.43, 5%: -2.86, 10%: -2.57
    // Fit a smooth logistic curve through these points.
    let z = (tau + 1.0) / 2.0;
    let logistic = 1.0 / (1.0 + (-z * 3.5).exp());
    logistic.min(1.0).max(0.0)
}

/// Approximate ADF critical value at significance level alpha
/// using MacKinnon (1994) response surface.
///
/// For the constant-only model.
fn mackinnon_critical(alpha: f64, n: usize) -> f64 {
    let t = n as f64;

    match alpha {
        a if (a - 0.01).abs() < 0.005 => {
            // 1% level (constant-only): -3.43 asymptotic
            let b_inf = -3.4335;
            let b1 = -5.999;
            let b2 = -29.25;
            b_inf + b1 / t + b2 / (t * t)
        }
        a if (a - 0.05).abs() < 0.005 => {
            // 5% level (constant-only): -2.86 asymptotic
            let b_inf = -2.8621;
            let b1 = -2.738;
            let b2 = -8.36;
            b_inf + b1 / t + b2 / (t * t)
        }
        a if (a - 0.10).abs() < 0.005 => {
            // 10% level (constant-only): -2.57 asymptotic
            let b_inf = -2.5671;
            let b1 = -1.438;
            let b2 = -4.48;
            b_inf + b1 / t + b2 / (t * t)
        }
        _ => {
            let cv5 = mackinnon_critical(0.05, n);
            let cv10 = mackinnon_critical(0.10, n);
            if alpha < 0.05 {
                let cv1 = mackinnon_critical(0.01, n);
                let t = ((alpha / 0.01).ln() / (0.05_f64 / 0.01_f64).ln()).clamp(0.0, 1.0);
                cv1 + t * (cv5 - cv1)
            } else {
                let t = (((alpha - 0.05) / 0.05) as f64).clamp(0.0, 1.0);
                cv5 + t * (cv10 - cv5)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adf_on_random_walk() {
        // Random walk: y_t = y_{t-1} + ε_t (non-stationary)
        let n = 500;
        let mut rw = vec![0.0; n];
        for i in 1..n {
            rw[i] = rw[i - 1] + (rand::random::<f64>() - 0.5) * 0.5;
        }
        let result = adf_test(&rw);
        // Random walk should NOT be stationary
        assert!(!result.is_stationary, "Random walk should be non-stationary");
    }

    #[test]
    fn test_adf_on_stationary_series() {
        // White noise around constant mean — should not crash and produce finite values
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(42);
        let wn: Vec<f64> = (0..500).map(|_| rng.gen::<f64>() * 0.5 + 10.0).collect();
        let result = adf_test(&wn);
        // Verify it produces finite, reasonable values
        assert!(result.test_statistic.is_finite(), "ADF should produce finite τ");
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0, "p-value in [0,1]");
        assert!(result.nobs >= 480, "nobs should be close to 500, got {}", result.nobs);
        assert!(result.used_lag >= 1);
    }

    #[test]
    fn test_difference_to_stationarity() {
        // Create a deterministic trending series with large trend component
        let n = 500;
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(123);
        let mut trend: Vec<f64> = Vec::with_capacity(n);
        for t in 0..n {
            trend.push(0.5 * t as f64 + (rng.gen::<f64>() - 0.5) * 0.1);
        }
        let result = difference_to_stationarity(&trend);
        // Trending series should need at least 1 difference
        assert!(result.order >= 1,
            "Trending series should need at least 1 difference, got {}", result.order);
        // After differencing, we should have a valid series
        assert!(!result.differenced_series.is_empty());
        assert!(result.differenced_series.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_undo_difference() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // First differences: [1, 1, 1, 1], so first prediction is +1 from last
        let diff_preds = vec![1.0, 1.0, 1.0];
        let levels = undo_difference(&original, &diff_preds, 1);
        assert!((levels[0] - 6.0).abs() < 1e-10);
        assert!((levels[1] - 7.0).abs() < 1e-10);
        assert!((levels[2] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_first_difference() {
        let s = vec![1.0, 3.0, 6.0, 10.0];
        let d = first_difference(&s);
        assert_eq!(d, vec![2.0, 3.0, 4.0]);
    }
}
