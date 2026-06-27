use nalgebra::DMatrix;

pub struct CovarianceEngine;

impl CovarianceEngine {
    /// Compute covariance and correlation matrices from a numeric data matrix
    /// `data`: rows × columns (observations × variables)
    pub fn compute(data: &DMatrix<f64>) -> (DMatrix<f64>, Vec<String>, DMatrix<f64>) {
        let (nrows, ncols) = data.shape();
        if nrows < 2 || ncols == 0 {
            return (
                DMatrix::zeros(0, 0),
                vec![],
                DMatrix::zeros(0, 0),
            );
        }

        // Center the data (subtract column means)
        let means = data.row_mean();
        let mut centered = data.clone();
        for j in 0..ncols {
            let mean = means[j];
            for i in 0..nrows {
                centered[(i, j)] -= mean;
            }
        }

        // Covariance: (Xᵀ X) / (n-1)
        let cov = (centered.transpose() * &centered) / ((nrows - 1) as f64);

        // Correlation: cov / (std_i * std_j)
        let std_devs: Vec<f64> = (0..ncols)
            .map(|j| cov[(j, j)].sqrt())
            .collect();

        let mut corr = DMatrix::zeros(ncols, ncols);
        for i in 0..ncols {
            for j in 0..ncols {
                let denom = std_devs[i] * std_devs[j];
                if denom > 1e-15 {
                    corr[(i, j)] = cov[(i, j)] / denom;
                } else {
                    corr[(i, j)] = 0.0;
                }
            }
        }

        // Generate numbered column labels
        let cols: Vec<String> = (0..ncols).map(|i| format!("col{}", i)).collect();

        (cov, cols, corr)
    }

    /// Find highly correlated pairs from a correlation matrix
    pub fn find_collinear_pairs(
        corr: &DMatrix<f64>,
        col_names: &[String],
        threshold: f64,
    ) -> Vec<(String, String, f64)> {
        let n = corr.nrows();
        let mut pairs = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let r = corr[(i, j)];
                if r.abs() > threshold {
                    pairs.push((col_names[i].clone(), col_names[j].clone(), r));
                }
            }
        }

        // Sort by absolute correlation descending
        pairs.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());
        pairs
    }
}
