use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, StudentsT};

#[derive(Debug, Clone)]
pub struct MatrixLinearRegression {
    pub reg: f64,
    pub beta: Vec<f64>,  // includes intercept as beta[0]
    pub xtx_inv: DMatrix<f64>,
    pub sigma2: f64,
    pub feature_cols: Vec<String>,
    pub target_col: String,
    pub x_mean: Vec<f64>,
    pub x_std: Vec<f64>,
    pub y_mean: f64,
    pub y_std: f64,
    pub n_samples: usize,
    pub k_features: usize,
}

#[derive(Debug, Clone)]
pub struct RegressionResult {
    pub predictions: Vec<f64>,
    pub coefficients: Vec<f64>,       // without intercept
    pub intercept: f64,
    pub rss: f64,
    pub sigma2: f64,
    pub prediction_variance: Vec<f64>,
    pub standard_error: Vec<f64>,
    pub prediction_interval_lower: Vec<f64>,
    pub prediction_interval_upper: Vec<f64>,
    pub cv: Vec<f64>,
    pub target_column: String,
    pub feature_columns: Vec<String>,
}

impl MatrixLinearRegression {
    pub fn new(regularization: f64) -> Self {
        Self {
            reg: regularization,
            beta: Vec::new(),
            xtx_inv: DMatrix::zeros(0, 0),
            sigma2: 0.0,
            feature_cols: Vec::new(),
            target_col: String::new(),
            x_mean: Vec::new(),
            x_std: Vec::new(),
            y_mean: 0.0,
            y_std: 1.0,
            n_samples: 0,
            k_features: 0,
        }
    }

    /// Fit multivariate linear regression
    /// `data`: rows × cols matrix (all numeric)
    /// `target_idx`: column index of the target
    /// `feature_indices`: column indices of features
    pub fn fit(
        &mut self,
        data: &DMatrix<f64>,
        target_idx: usize,
        feature_indices: &[usize],
        feature_names: &[String],
        target_name: &str,
    ) -> RegressionResult {
        let (nrows, _) = data.shape();

        self.target_col = target_name.to_string();
        self.feature_cols = feature_indices.iter().map(|&i| feature_names[i].clone()).collect();
        self.k_features = feature_indices.len();
        self.n_samples = nrows;

        // Extract X (features) and y (target)
        let mut x_raw = DMatrix::zeros(nrows, feature_indices.len());
        let mut y = DVector::zeros(nrows);

        for i in 0..nrows {
            y[i] = data[(i, target_idx)];
            for (j, &fi) in feature_indices.iter().enumerate() {
                x_raw[(i, j)] = data[(i, fi)];
            }
        }

        // Standardize
        self.x_mean = (0..feature_indices.len())
            .map(|j| x_raw.column(j).mean())
            .collect();
        self.x_std = (0..feature_indices.len())
            .map(|j| {
                let col = x_raw.column(j);
                let mean = self.x_mean[j];
                let var: f64 = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / nrows as f64;
                let std = var.sqrt();
                if std < 1e-10 { 1.0 } else { std }
            })
            .collect();

        self.y_mean = y.mean();
        let y_var: f64 = y.iter().map(|v| (v - self.y_mean).powi(2)).sum::<f64>() / nrows as f64;
        self.y_std = y_var.sqrt().max(1e-10);

        // Standardized matrices
        let mut x = x_raw.clone();
        for j in 0..feature_indices.len() {
            let mean = self.x_mean[j];
            let std = self.x_std[j];
            for i in 0..nrows {
                x[(i, j)] = (x_raw[(i, j)] - mean) / std;
            }
        }

        let mut y_norm = DVector::zeros(nrows);
        for i in 0..nrows {
            y_norm[i] = (y[i] - self.y_mean) / self.y_std;
        }

        // Add intercept column to design matrix
        let mut x_design = DMatrix::zeros(nrows, feature_indices.len() + 1);
        for i in 0..nrows {
            x_design[(i, 0)] = 1.0;
            for j in 0..feature_indices.len() {
                x_design[(i, j + 1)] = x[(i, j)];
            }
        }

        // Ridge-regularized normal equation
        let mut xtx = &x_design.transpose() * &x_design;
        for i in 0..xtx.nrows() {
            xtx[(i, i)] += self.reg;
        }

        let xty = x_design.transpose() * &y_norm;

        // Solve via SVD pseudo-inverse
        let svd = xtx.clone().svd(true, true);
        self.xtx_inv = svd
            .solve(&DMatrix::identity(xtx.nrows(), xtx.ncols()), 1e-10)
            .unwrap_or(DMatrix::zeros(xtx.nrows(), xtx.ncols()));

        self.beta = (&self.xtx_inv * &xty).iter().copied().collect();

        // Predictions (in normalized space, then un-normalized)
        let y_pred_norm = &x_design * &DVector::from_iterator(self.beta.len(), self.beta.iter().copied());
        let mut y_pred = DVector::zeros(nrows);
        for i in 0..nrows {
            y_pred[i] = y_pred_norm[i] * self.y_std + self.y_mean;
        }

        // Residuals and variance
        let residuals: Vec<f64> = y_pred.iter().zip(y.iter()).map(|(yp, yt)| yt - yp).collect();
        let rss: f64 = residuals.iter().map(|r| r * r).sum();

        let df = (nrows - feature_indices.len() - 1).max(1);
        self.sigma2 = rss / df as f64;

        // Prediction variance for training points
        let n_features_plus_1 = feature_indices.len() + 1;
        let mut pred_var = Vec::with_capacity(nrows);
        let mut se = Vec::with_capacity(nrows);
        let mut lower = Vec::with_capacity(nrows);
        let mut upper = Vec::with_capacity(nrows);
        let mut cv = Vec::with_capacity(nrows);

        let t_crit = Self::t_critical(0.975, df as f64);

        for i in 0..nrows {
            let xr = x_design.row(i);
            let xr_vec = DVector::from_iterator(n_features_plus_1, xr.iter().copied());
            let var = (xr_vec.dot(&(&self.xtx_inv * &xr_vec))) * self.sigma2 * (self.y_std * self.y_std);
            let var = var.max(0.0);
            let s = var.sqrt();

            pred_var.push(var);
            se.push(s);
            lower.push(y_pred[i] - t_crit * s);
            upper.push(y_pred[i] + t_crit * s);
            cv.push(s / (y_pred[i].abs() + 1e-10));
        }

        RegressionResult {
            predictions: y_pred.iter().copied().collect(),
            coefficients: self.beta.iter().skip(1).copied().collect(), // exclude intercept
            intercept: self.beta.first().copied().unwrap_or(0.0) * self.y_std + self.y_mean,
            rss,
            sigma2: self.sigma2,
            prediction_variance: pred_var,
            standard_error: se,
            prediction_interval_lower: lower,
            prediction_interval_upper: upper,
            cv,
            target_column: self.target_col.clone(),
            feature_columns: self.feature_cols.clone(),
        }
    }

    /// Predict for a single new observation
    /// Returns (prediction, lower, upper, cv)
    pub fn predict(&self, x_new: &[f64]) -> (f64, f64, f64, f64) {
        let k = self.k_features;
        if k == 0 {
            return (0.0, 0.0, 0.0, 0.0);
        }

        // Standardize the input
        let mut x_norm = DVector::zeros(k + 1);
        x_norm[0] = 1.0; // intercept
        for j in 0..k {
            x_norm[j + 1] = (x_new[j] - self.x_mean[j]) / self.x_std[j];
        }

        // Predict in normalized space
        let beta_vec = DVector::from_iterator(k + 1, self.beta.iter().copied());
        let pred_norm = x_norm.dot(&beta_vec);
        let pred = pred_norm * self.y_std + self.y_mean;

        // Prediction variance
        let var = (x_norm.dot(&(&self.xtx_inv * &x_norm))) * self.sigma2 * (self.y_std * self.y_std);
        let var = var.max(0.0);
        let se = var.sqrt();

        let df = (self.n_samples - self.k_features - 1).max(1) as f64;
        let t_crit = Self::t_critical(0.975, df);

        let cv_val = se / (pred.abs() + 1e-10);

        (pred, pred - t_crit * se, pred + t_crit * se, cv_val)
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
