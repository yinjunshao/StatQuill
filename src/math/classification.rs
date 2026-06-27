use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// Label encoder: maps string categories to integer indices and back.
#[derive(Debug, Clone)]
pub struct LabelEncoder {
    pub classes: Vec<String>,
    /// Map from class name → index
    pub class_to_index: HashMap<String, usize>,
}

impl LabelEncoder {
    /// Create encoder from a list of target values (one per row).
    pub fn fit(values: &[String]) -> Self {
        let mut seen = HashMap::new();
        let mut classes = Vec::new();
        for v in values {
            let v = v.trim().to_string();
            if !v.is_empty() && !seen.contains_key(&v) {
                seen.insert(v.clone(), classes.len());
                classes.push(v);
            }
        }
        // Sort for deterministic output
        classes.sort();
        let class_to_index: HashMap<String, usize> = classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();
        LabelEncoder {
            classes,
            class_to_index,
        }
    }

    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    pub fn is_binary(&self) -> bool {
        self.classes.len() == 2
    }

    /// Encode a single label to its index.
    pub fn encode(&self, label: &str) -> Option<usize> {
        self.class_to_index.get(label.trim()).copied()
    }

    /// Decode an index back to the class label.
    pub fn decode(&self, index: usize) -> Option<&str> {
        self.classes.get(index).map(|s| s.as_str())
    }

    /// Convert a vector of string labels to integer indices.
    pub fn transform(&self, values: &[String]) -> Vec<usize> {
        values
            .iter()
            .filter_map(|v| self.encode(v))
            .collect()
    }
}

/// Result of a classification prediction for a single row.
#[derive(Debug, Clone)]
pub struct ClassificationPrediction {
    /// Predicted class label
    pub predicted_class: String,
    /// Probability for each class (sums to 1, in class order)
    pub probabilities: Vec<f64>,
    /// Confidence in the predicted class (its probability)
    pub confidence: f64,
    /// The index of the predicted class
    pub predicted_index: usize,
}

/// Multinomial (Softmax) Logistic Regression for multi-class classification.
///
/// For binary targets: uses a single weight vector with sigmoid (logistic regression).
/// For K > 2 targets: uses K weight vectors with softmax + cross-entropy.
///
/// This is a local-only model — no external API needed.
#[derive(Debug, Clone)]
pub struct MultinomialLogisticRegression {
    /// Weight matrix: (num_classes × num_features)
    /// For binary: shape is (1 × num_features)
    pub weights: DMatrix<f64>,
    /// Intercept (bias) for each class
    pub intercepts: Vec<f64>,
    /// Number of classes
    pub num_classes: usize,
    /// Number of features
    pub num_features: usize,
    /// Class labels in order
    pub classes: Vec<String>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Feature means (for standardization)
    pub x_mean: Vec<f64>,
    /// Feature standard deviations
    pub x_std: Vec<f64>,
    /// Regularization strength (L2)
    pub lambda: f64,
    /// Whether the model has been fitted
    pub fitted: bool,
}

impl MultinomialLogisticRegression {
    pub fn new(lambda: f64) -> Self {
        Self {
            weights: DMatrix::zeros(0, 0),
            intercepts: Vec::new(),
            num_classes: 0,
            num_features: 0,
            classes: Vec::new(),
            feature_names: Vec::new(),
            x_mean: Vec::new(),
            x_std: Vec::new(),
            lambda,
            fitted: false,
        }
    }

    /// Fit the model using gradient descent with softmax + cross-entropy loss.
    ///
    /// * `features` - design matrix: rows × features (should be standardized)
    /// * `labels` - integer target labels: one per row, 0 ≤ label < num_classes
    /// * `feature_names` - column names for the feature matrix
    /// * `classes` - string labels for each class index
    pub fn fit(
        &mut self,
        features: &DMatrix<f64>,
        labels: &[usize],
        feature_names: &[String],
        classes: &[String],
    ) {
        let (nrows, ncols) = features.shape();
        if nrows == 0 || ncols == 0 || labels.len() != nrows || classes.is_empty() {
            return;
        }

        self.num_features = ncols;
        self.num_classes = classes.len();
        self.classes = classes.to_vec();
        self.feature_names = feature_names.to_vec();

        // Standardize features
        self.x_mean = (0..ncols)
            .map(|j| features.column(j).mean())
            .collect();
        self.x_std = (0..ncols)
            .map(|j| {
                let col = features.column(j);
                let mean = self.x_mean[j];
                let var: f64 = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / nrows as f64;
                let std = var.sqrt();
                if std < 1e-10 { 1.0 } else { std }
            })
            .collect();

        let mut x_norm = features.clone();
        for j in 0..ncols {
            for i in 0..nrows {
                x_norm[(i, j)] = (features[(i, j)] - self.x_mean[j]) / self.x_std[j];
            }
        }

        // One-hot encode labels
        let y_onehot = DMatrix::from_fn(nrows, self.num_classes, |i, j| {
            if labels.get(i) == Some(&j) { 1.0 } else { 0.0 }
        });

        // Initialize weights
        let mut w = DMatrix::zeros(self.num_classes, ncols);
        let mut b = vec![0.0; self.num_classes];

        // Gradient descent with softmax cross-entropy
        let learning_rate = 0.1;
        let max_iter = 500;
        let tol = 1e-6;

        let mut prev_loss = f64::INFINITY;

        for _iter in 0..max_iter {
            // Forward pass: compute scores = X * W^T + b
            let scores = &x_norm * &w.transpose();
            // Softmax
            let probs = softmax_rows(&scores, &b);

            // Compute cross-entropy loss
            let mut loss = 0.0;
            for i in 0..nrows {
                for k in 0..self.num_classes {
                    let y_ik = y_onehot[(i, k)];
                    let p_ik = probs[(i, k)].max(1e-15);
                    loss -= y_ik * f64::ln(p_ik);
                }
            }
            loss /= nrows as f64;
            // Add L2 regularization
            let reg_term: f64 = w.iter().map(|w_ij| w_ij * w_ij).sum::<f64>() * self.lambda / (2.0 * nrows as f64);
            loss += reg_term;

            // Check convergence
            if (prev_loss - loss).abs() < tol {
                break;
            }
            prev_loss = loss;

            // Gradient
            let error = &probs - &y_onehot; // (nrows × num_classes)
            let dw = &error.transpose() * &x_norm; // (num_classes × ncols)
            let db: Vec<f64> = (0..self.num_classes)
                .map(|k| error.column(k).sum())
                .collect();

            // Update with L2 regularization
            for j in 0..ncols {
                for k in 0..self.num_classes {
                    w[(k, j)] -= learning_rate * (dw[(k, j)] / nrows as f64 + self.lambda * w[(k, j)] / nrows as f64);
                }
            }
            for k in 0..self.num_classes {
                b[k] -= learning_rate * (db[k] / nrows as f64);
            }
        }

        self.weights = w;
        self.intercepts = b;
        self.fitted = true;
    }

    /// Predict class probabilities for a single row.
    pub fn predict_proba(&self, x: &[f64]) -> Vec<f64> {
        if !self.fitted || x.len() != self.num_features {
            return vec![0.0; self.num_classes.max(1)];
        }

        // Standardize input
        let x_norm: DVector<f64> = DVector::from_iterator(
            self.num_features,
            (0..self.num_features).map(|j| {
                (x[j] - self.x_mean[j]) / self.x_std[j]
            }),
        );

        // Scores = W * x_norm + b
        let scores_vec = &self.weights * &x_norm;
        let mut scores: Vec<f64> = (0..self.num_classes)
            .map(|k| scores_vec[k] + self.intercepts[k])
            .collect();

        // Softmax
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = scores.iter().map(|s| (s - max_score).exp()).sum();
        if exp_sum == 0.0 {
            let uniform = 1.0 / self.num_classes as f64;
            return vec![uniform; self.num_classes];
        }
        for s in &mut scores {
            *s = ((*s - max_score).exp()) / exp_sum;
        }
        scores
    }

    /// Predict class for a single row.
    pub fn predict(&self, x: &[f64]) -> ClassificationPrediction {
        let probs = self.predict_proba(x);
        let mut best_idx = 0;
        let mut best_prob = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            if p > best_prob {
                best_prob = p;
                best_idx = i;
            }
        }
        ClassificationPrediction {
            predicted_class: self.classes.get(best_idx).cloned().unwrap_or_default(),
            probabilities: probs,
            confidence: best_prob,
            predicted_index: best_idx,
        }
    }
}

/// Compute softmax across each row of the scores matrix.
fn softmax_rows(scores: &DMatrix<f64>, biases: &[f64]) -> DMatrix<f64> {
    let (nrows, ncols) = scores.shape();
    let mut probs = DMatrix::zeros(nrows, ncols);
    for i in 0..nrows {
        let row_max = (0..ncols)
            .map(|j| scores[(i, j)] + biases[j])
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = (0..ncols)
            .map(|j| (scores[(i, j)] + biases[j] - row_max).exp())
            .sum();
        if exp_sum > 0.0 {
            for j in 0..ncols {
                probs[(i, j)] = (scores[(i, j)] + biases[j] - row_max).exp() / exp_sum;
            }
        }
    }
    probs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_encoder_basic() {
        let values: Vec<String> = ["cat", "dog", "cat", "bird", "dog"]
            .iter().map(|s| s.to_string()).collect();
        let encoder = LabelEncoder::fit(&values);
        assert_eq!(encoder.num_classes(), 3);
        // After sorting alphabetically: bird (0), cat (1), dog (2)
        assert_eq!(encoder.encode("bird"), Some(0));
        assert_eq!(encoder.encode("cat"), Some(1));
        assert_eq!(encoder.encode("dog"), Some(2));
        assert_eq!(encoder.decode(0), Some("bird"));
    }

    #[test]
    fn test_label_encoder_binary() {
        let values: Vec<String> = ["yes", "no", "yes", "yes"]
            .iter().map(|s| s.to_string()).collect();
        let encoder = LabelEncoder::fit(&values);
        assert!(encoder.is_binary());
        assert_eq!(encoder.num_classes(), 2);
    }

    #[test]
    fn test_multinomial_fit_predict() {
        use nalgebra::DMatrix;

        // Simple 2-class problem with clear separation
        let features = DMatrix::from_row_slice(6, 2, &[
            1.0, 1.0, // class 0
            1.5, 1.2, // class 0
            0.8, 0.9, // class 0
            5.0, 5.0, // class 1
            5.5, 4.8, // class 1
            4.5, 5.2, // class 1
        ]);
        let labels = vec![0usize, 0, 0, 1, 1, 1];
        let feature_names = vec!["x".to_string(), "y".to_string()];
        let classes = vec!["low".to_string(), "high".to_string()];

        let mut model = MultinomialLogisticRegression::new(0.01);
        model.fit(&features, &labels, &feature_names, &classes);
        assert!(model.fitted);

        let pred = model.predict(&[5.0, 5.1]);
        assert_eq!(pred.predicted_class, "high");
        assert!(pred.confidence > 0.5);
    }

    #[test]
    fn test_softmax_properties() {
        let scores = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
        let b = vec![0.0, 0.0, 0.0];
        let probs = softmax_rows(&scores, &b);
        for i in 0..2 {
            let row_sum: f64 = (0..3).map(|j| probs[(i, j)]).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }
}
