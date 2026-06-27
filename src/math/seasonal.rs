/// Holt-Winters triple exponential smoothing for time series with trend and seasonality.
///
/// This is a widely-used local forecasting method that handles:
///   - Level (baseline)
///   - Trend (additive slope)
///   - Seasonality (additive or multiplicative pattern)
///
/// Much better than plain AR for data with clear seasonal patterns.
#[derive(Debug, Clone)]
pub struct HoltWinters {
    /// Smoothing parameter for level (0 < α < 1)
    pub alpha: f64,
    /// Smoothing parameter for trend (0 < β < 1)
    pub beta: f64,
    /// Smoothing parameter for seasonality (0 < γ < 1)
    pub gamma: f64,
    /// Seasonal period (e.g., 4 for quarterly, 12 for monthly)
    pub period: usize,
    /// Whether to use multiplicative (true) or additive (false) seasonality
    pub multiplicative: bool,
    /// Fitted level component (last value)
    pub level: f64,
    /// Fitted trend component (last value)
    pub trend: f64,
    /// Fitted seasonal components (one per period)
    pub seasonal: Vec<f64>,
    /// Residual variance (σ²)
    pub sigma2: f64,
    /// Whether the model has been fitted
    pub fitted: bool,
}

impl HoltWinters {
    /// Create a new Holt-Winters model with given smoothing parameters.
    ///
    /// * `period` - number of observations per seasonal cycle (e.g., 12 for monthly data with yearly season)
    /// * `multiplicative` - true for multiplicative seasonality (amplitude grows with level)
    pub fn new(alpha: f64, beta: f64, gamma: f64, period: usize, multiplicative: bool) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            beta: beta.clamp(0.0, 1.0),
            gamma: gamma.clamp(0.0, 1.0),
            period: period.max(2),
            multiplicative,
            level: 0.0,
            trend: 0.0,
            seasonal: Vec::new(),
            sigma2: 0.0,
            fitted: false,
        }
    }

    /// Auto-tune smoothing parameters by grid search over the validation MSE.
    ///
    /// Returns a new HoltWinters with optimized parameters.
    pub fn auto_tune(series: &[f64], period: usize) -> Self {
        let n = series.len();
        if n < 2 * period + 2 {
            // Not enough data for tuning — use defaults
            return Self::new(0.3, 0.1, 0.1, period, false);
        }

        let alphas = [0.1, 0.3, 0.5, 0.7];
        let betas = [0.05, 0.1, 0.2, 0.3];
        let gammas = [0.05, 0.1, 0.2, 0.3];

        // Use last 20% as validation
        let val_start = n - (n / 5);
        let train_len = val_start;

        let mut best_mse = f64::INFINITY;
        let mut best_model = None;

        for &alpha in &alphas {
            for &beta in &betas {
                for &gamma in &gammas {
                    let mut model = Self::new(alpha, beta, gamma, period, false);
                    if let Ok(_) = model.fit(&series[..train_len]) {
                        let preds = model.predict(series, n - val_start);
                        let mse: f64 = preds
                            .iter()
                            .zip(&series[val_start..])
                            .map(|(p, a)| (p - a).powi(2))
                            .sum::<f64>()
                            / preds.len() as f64;
                        if mse < best_mse {
                            best_mse = mse;
                            best_model = Some(model);
                        }
                    }
                }
            }
        }

        match best_model {
            Some(m) => m,
            None => {
                // Fallback: fit with defaults on full data
                let mut m = Self::new(0.3, 0.1, 0.1, period, false);
                let _ = m.fit(series);
                m
            }
        }
    }

    /// Fit the Holt-Winters model to a time series.
    pub fn fit(&mut self, series: &[f64]) -> Result<(), String> {
        let n = series.len();
        if n < 2 * self.period {
            return Err(format!(
                "Series too short ({} observations) for period {}",
                n, self.period
            ));
        }

        // Initialize level, trend, and seasonal components from first two periods
        let mut level = series.iter().take(self.period).sum::<f64>() / self.period as f64;
        let mut trend = (series.iter().skip(self.period).take(self.period).sum::<f64>()
            - series.iter().take(self.period).sum::<f64>())
            / (self.period * self.period) as f64;

        let mut seasonal: Vec<f64> = if self.multiplicative {
            series
                .iter()
                .take(self.period)
                .map(|&y| if level.abs() > 1e-10 { y / level } else { 1.0 })
                .collect()
        } else {
            series
                .iter()
                .take(self.period)
                .map(|&y| y - level)
                .collect()
        };

        let mut residuals = Vec::new();

        for t in self.period..n {
            let y_t = series[t];
            let s_idx = t % self.period;
            let s_t = seasonal[s_idx];

            // Forecast one step ahead
            let forecast = if self.multiplicative {
                (level + trend) * s_t
            } else {
                level + trend + s_t
            };

            let residual = y_t - forecast;
            residuals.push(residual);

            // Update components
            let new_level = if self.multiplicative && s_t.abs() > 1e-10 {
                self.alpha * (y_t / s_t) + (1.0 - self.alpha) * (level + trend)
            } else if !self.multiplicative {
                self.alpha * (y_t - s_t) + (1.0 - self.alpha) * (level + trend)
            } else {
                level
            };

            let new_trend = self.beta * (new_level - level) + (1.0 - self.beta) * trend;

            let new_season = if self.multiplicative && new_level.abs() > 1e-10 {
                self.gamma * (y_t / new_level) + (1.0 - self.gamma) * s_t
            } else if !self.multiplicative {
                self.gamma * (y_t - new_level) + (1.0 - self.gamma) * s_t
            } else {
                s_t
            };

            level = new_level;
            trend = new_trend;
            seasonal[s_idx] = new_season;
        }

        self.level = level;
        self.trend = trend;
        self.seasonal = seasonal;

        // Compute residual variance
        if !residuals.is_empty() {
            let n_res = residuals.len() as f64;
            self.sigma2 = residuals.iter().map(|r| r * r).sum::<f64>() / n_res;
        }

        self.fitted = true;
        Ok(())
    }

    /// Predict the next `steps` values.
    ///
    /// * `series` - the original series (needed for seasonal indexing)
    pub fn predict(&self, series: &[f64], steps: usize) -> Vec<f64> {
        if !self.fitted {
            return Vec::new();
        }

        let n = series.len();
        let mut preds = Vec::with_capacity(steps);

        for h in 1..=steps {
            let t = n + h - 1;
            let s_idx = t % self.period;
            let s = self.seasonal.get(s_idx).copied().unwrap_or(if self.multiplicative { 1.0 } else { 0.0 });

            let pred = if self.multiplicative {
                (self.level + h as f64 * self.trend) * s
            } else {
                self.level + h as f64 * self.trend + s
            };
            preds.push(pred);
        }

        preds
    }

    /// Predict with confidence intervals using Gaussian approximation.
    pub fn predict_with_intervals(
        &self,
        series: &[f64],
        steps: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let preds = self.predict(series, steps);
        let se = self.sigma2.sqrt();
        // Wider intervals for multi-step forecasts
        let lower: Vec<f64> = preds
            .iter()
            .enumerate()
            .map(|(h, &p)| p - 1.96 * se * ((h + 1) as f64).sqrt())
            .collect();
        let upper: Vec<f64> = preds
            .iter()
            .enumerate()
            .map(|(h, &p)| p + 1.96 * se * ((h + 1) as f64).sqrt())
            .collect();
        (preds, lower, upper)
    }
}

/// Detect the dominant seasonal period from autocorrelation of the differenced series.
///
/// Returns (period, confidence) where period is the estimated seasonal length
/// and confidence is a heuristic 0–1 score of how clear the seasonality is.
pub fn detect_seasonal_period(series: &[f64], max_period: usize) -> (usize, f64) {
    let n = series.len();
    let max_p = max_period.min(n / 3).max(2);

    if n < 2 * max_p {
        return (1, 0.0);
    }

    // Detrend: take first difference
    let diff: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();
    let diff_mean = diff.iter().sum::<f64>() / diff.len() as f64;

    // Compute autocorrelations at each lag
    let mut best_period = 1usize;
    let mut best_acf = 0.0f64;
    let mut acfs = Vec::new();

    for lag in 2..=max_p {
        let mut num = 0.0;
        let mut den1 = 0.0;
        let mut den2 = 0.0;

        for t in lag..diff.len() {
            let d1 = diff[t] - diff_mean;
            let d2 = diff[t - lag] - diff_mean;
            num += d1 * d2;
            den1 += d1 * d1;
            den2 += d2 * d2;
        }

        if den1 > 0.0 && den2 > 0.0 {
            let acf = num / (den1 * den2).sqrt();
            acfs.push((lag, acf));
            if acf.abs() > best_acf.abs() {
                best_acf = acf.abs();
                best_period = lag;
            }
        }
    }

    // Also check harmonics: if period/2 has similar ACF, period might be the real one
    let harmonic_acf = acfs
        .iter()
        .find(|&&(lag, _)| lag == best_period / 2)
        .map(|&(_, acf)| acf.abs())
        .unwrap_or(0.0);

    let confidence = best_acf.min(1.0).max(0.0);
    let is_harmonic = harmonic_acf > 0.5 * best_acf;

    if is_harmonic && best_period / 2 >= 2 {
        (best_period / 2, confidence * 0.9)
    } else {
        (best_period, confidence)
    }
}

/// Seasonal differencing: y_t - y_{t-s} where s is the seasonal period.
pub fn seasonal_difference(series: &[f64], period: usize) -> Vec<f64> {
    if period == 0 || period >= series.len() {
        return series.to_vec();
    }
    series.windows(period + 1).map(|w| w[period] - w[0]).collect()
}

/// Apply both seasonal and first differencing.
/// Returns (differenced_series, seasonal_order, first_order)
pub fn full_difference(series: &[f64], seasonal_period: usize) -> (Vec<f64>, usize, usize) {
    if seasonal_period <= 1 || seasonal_period >= series.len() {
        // Just first difference
        let d1 = super::stationarity::first_difference(series);
        let order = if series.len() > 1 { 1 } else { 0 };
        return (d1, 0, order);
    }

    // Seasonal difference first, then check stationarity, then first difference if needed
    let sdiff = seasonal_difference(series, seasonal_period);
    if sdiff.len() < 5 {
        return (sdiff, 1, 0);
    }

    let adf = super::stationarity::adf_test(&sdiff);
    if adf.is_stationary {
        (sdiff, 1, 0)
    } else {
        let d1 = super::stationarity::first_difference(&sdiff);
        (d1, 1, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_holt_winters_fit_predict() {
        // Generate seasonal series: level + trend + sin seasonality
        let n = 60;
        let mut series = vec![0.0; n];
        for t in 0..n {
            series[t] = 10.0 + 0.5 * t as f64 + 5.0 * ((t as f64 * 2.0 * std::f64::consts::PI) / 12.0).sin();
        }

        let mut model = HoltWinters::new(0.3, 0.1, 0.1, 12, false);
        assert!(model.fit(&series).is_ok());
        assert!(model.fitted);

        let preds = model.predict(&series, 6);
        assert_eq!(preds.len(), 6);
        assert!(preds.iter().all(|&p| p.is_finite()));
    }

    #[test]
    fn test_detect_seasonal_period() {
        // Clear 12-period seasonality
        let mut series = vec![0.0; 120];
        for t in 0..120 {
            series[t] = ((t as f64 * 2.0 * std::f64::consts::PI) / 12.0).sin();
        }
        let (period, conf) = detect_seasonal_period(&series, 24);
        // Should detect 12 or 6 (harmonic)
        assert!(period == 12 || period == 6);
        assert!(conf > 0.3);
    }

    #[test]
    fn test_seasonal_difference() {
        let s = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let d = seasonal_difference(&s, 4);
        assert_eq!(d, vec![4.0, 4.0, 4.0, 4.0]); // s[4]-s[0]=5-1=4, etc.
    }
}
