use anyhow::Result;
use nalgebra::DVector;
use statrs::distribution::{ContinuousCDF, StudentsT};

#[derive(Debug, Clone)]
pub struct AutoregressiveModel {
    pub max_lag: usize,
    pub phi: Vec<f64>,
    pub c: f64,
    pub sigma2: f64,
    pub p: usize,
}

#[derive(Debug, Clone)]
pub struct ARResult {
    pub predictions: Vec<f64>,
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub rss: f64,
    pub sigma2: f64,
    pub prediction_variance: Vec<f64>,
    pub standard_error: Vec<f64>,
    pub prediction_interval_lower: Vec<f64>,
    pub prediction_interval_upper: Vec<f64>,
    pub cv: Vec<f64>,
}

impl AutoregressiveModel {
    pub fn new(max_lag: usize) -> Self {
        Self {
            max_lag,
            phi: Vec::new(),
            c: 0.0,
            sigma2: 0.0,
            p: 0,
        }
    }

    /// Fit AR(p) model with AIC-based order selection
    pub fn fit(&mut self, series: &[f64]) -> Result<ARResult> {
        let n = series.len();
        let max_p = self.max_lag.min(n / 3);

        let mut best_aic = f64::INFINITY;
        let mut best_result: Option<ARResult> = None;

        for p in 1..=max_p {
            let n_eff = n - p;
            if n_eff <= p + 1 {
                continue;
            }

            let mut y = DVector::zeros(n_eff);
            let mut x_vals = Vec::with_capacity(n_eff * (p + 1));

            for t in 0..n_eff {
                y[t] = series[t + p];
                x_vals.push(1.0); // intercept
                for lag in 1..=p {
                    x_vals.push(series[t + p - lag]);
                }
            }

            let x = nalgebra::DMatrix::from_vec(n_eff, p + 1, x_vals);

            // Solve OLS: β = (XᵀX)⁻¹ Xᵀ y using SVD
            let x_t_x = x.transpose() * &x;
            let x_t_y = x.transpose() * &y;

            // SVD for pseudo-inverse (nalgebra 0.33 returns SVD directly)
            let svd = x_t_x.clone().svd(true, true);
            let beta = svd.solve(&x_t_y, 1e-10).unwrap_or_else(|_| DVector::zeros(p + 1));

            if beta.is_empty() {
                continue;
            }

            let c = beta[0];
            let phi: Vec<f64> = beta.iter().skip(1).copied().collect();

            let y_pred = &x * &beta;
            let residuals: Vec<f64> = y_pred.iter().zip(y.iter()).map(|(yp, yt)| yt - yp).collect();
            let rss: f64 = residuals.iter().map(|r| r * r).sum();

            let df = n_eff - (p + 1);
            let sigma2 = if df > 0 { rss / df as f64 } else { rss };

            let aic = n_eff as f64 * f64::ln(sigma2.max(1e-10)) + 2.0 * (p + 1) as f64;

            if aic < best_aic {
                best_aic = aic;

                // Compute (XᵀX)⁻¹ for prediction variance
                let xtx_inv = svd.solve(
                    &nalgebra::DMatrix::identity(p + 1, p + 1),
                    1e-10,
                ).unwrap_or(nalgebra::DMatrix::zeros(p + 1, p + 1));

                let se_items: Vec<(f64, f64)> = {
                    let mut items = Vec::with_capacity(n_eff);
                    for i in 0..n_eff {
                        let xr = x.row(i);
                        let xr_vec = DVector::from_iterator(p + 1, xr.iter().copied());
                        let var = (xr_vec.dot(&(&xtx_inv * &xr_vec))) * sigma2;
                        let var = f64::max(var, 0.0);
                        let se = var.sqrt();
                        items.push((var, se));
                    }
                    items
                };

                let t_crit = Self::t_critical(0.975, df.max(1) as f64);

                let pred_var: Vec<f64> = se_items.iter().map(|(v, _)| *v).collect();
                let se: Vec<f64> = se_items.iter().map(|(_, s)| *s).collect();
                let lower: Vec<f64> = y_pred.iter().zip(se.iter()).map(|(yp, s)| yp - t_crit * s).collect();
                let upper: Vec<f64> = y_pred.iter().zip(se.iter()).map(|(yp, s)| yp + t_crit * s).collect();
                let cv: Vec<f64> = se.iter()
                    .zip(y_pred.iter())
                    .map(|(s, yp)| s / (yp.abs() + 1e-10))
                    .collect();

                best_result = Some(ARResult {
                    predictions: y_pred.iter().copied().collect(),
                    coefficients: phi,
                    intercept: c,
                    rss,
                    sigma2,
                    prediction_variance: pred_var,
                    standard_error: se,
                    prediction_interval_lower: lower,
                    prediction_interval_upper: upper,
                    cv,
                });
            }
        }

        match best_result {
            Some(result) => {
                self.p = 1; // placeholder, we don't track p through result struct
                self.c = result.intercept;
                self.phi = result.coefficients.clone();
                self.sigma2 = result.sigma2;
                Ok(result)
            }
            None => Err(anyhow::anyhow!("Could not fit AR model to data")),
        }
    }

    /// Predict the next `steps` values
    pub fn predict_next(&self, series: &[f64], steps: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let p = self.phi.len();
        if p == 0 {
            return Err(anyhow::anyhow!("Model not fitted"));
        }

        let n = series.len();
        let mut history: Vec<f64> = series[n.saturating_sub(p)..].to_vec();
        let mut predictions = Vec::with_capacity(steps);
        let mut lower_bounds = Vec::with_capacity(steps);
        let mut upper_bounds = Vec::with_capacity(steps);

        for _ in 0..steps {
            let mut pred = self.c;
            for i in 0..p {
                let idx = history.len() - 1 - i;
                pred += self.phi[i] * history[idx];
            }

            let variance = self.sigma2 * (1.0 + self.phi.iter().map(|phi| phi * phi).sum::<f64>());
            let se = variance.sqrt();

            let df = (n - p - 1).max(1) as f64;
            let t_crit = Self::t_critical(0.975, df);

            predictions.push(pred);
            lower_bounds.push(pred - t_crit * se);
            upper_bounds.push(pred + t_crit * se);
            history.push(pred);
        }

        Ok((predictions, lower_bounds, upper_bounds))
    }

    fn t_critical(prob: f64, df: f64) -> f64 {
        if df < 1.0 {
            return 1.96;
        }
        match StudentsT::new(0.0, 1.0, df) {
            Ok(dist) => dist.inverse_cdf(prob),
            Err(_) => 1.96,
        }
    }
}
