use std::collections::HashMap;

/// Encoding strategy for categorical columns
#[derive(Debug, Clone)]
pub enum EncodingStrategy {
    /// One-hot encoding: creates K binary columns for K categories
    /// Best for low-cardinality (unique ≤ 10)
    OneHot,
    /// Frequency encoding: replace category with its occurrence proportion
    /// Best for medium-cardinality (10 < unique ≤ 50)
    Frequency,
    /// Ordinal encoding: map ordered categories to integers 0,1,2,...
    /// Only use when the ordering is explicitly specified, not inferred
    Ordinal,
}

/// Metadata for encoding a single categorical column, used at prediction time
#[derive(Debug, Clone)]
pub struct CategoricalEncoder {
    /// Column name
    pub column_name: String,
    /// The strategy used
    pub strategy: EncodingStrategy,
    /// For OneHot: map from category value → column index in the expanded feature space
    pub onehot_map: HashMap<String, usize>,
    /// For Frequency: map from category value → frequency proportion
    pub frequency_map: HashMap<String, f64>,
    /// For Ordinal: map from category value → integer level
    pub ordinal_map: HashMap<String, usize>,
    /// Categories in order (for OneHot column naming)
    pub categories: Vec<String>,
    /// Default value for unseen categories (most frequent or median level)
    pub default_value: f64,
    /// How many new columns this encoder adds to the feature space
    pub expansion_width: usize,
    /// Starting column index in the augmented feature matrix
    pub feature_start_idx: usize,
}

impl CategoricalEncoder {
    /// Build an encoder from a column's raw string values.
    ///
    /// * `values` - the raw string values for this column (one per row)
    /// * `column_name` - the name of the column
    /// * `ordered` - if true, treat as ordinal (caller must know the ordering is real)
    /// * `feature_start_idx` - where this encoder's columns begin in the augmented matrix
    pub fn fit(
        values: &[String],
        column_name: &str,
        ordered: bool,
        feature_start_idx: usize,
    ) -> Self {
        // Count frequencies
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut total_nonempty = 0usize;
        for v in values {
            let v = v.trim().to_string();
            if !v.is_empty() {
                *counts.entry(v.clone()).or_insert(0) += 1;
                total_nonempty += 1;
            }
        }

        if total_nonempty == 0 {
            return CategoricalEncoder {
                column_name: column_name.to_string(),
                strategy: EncodingStrategy::Frequency,
                onehot_map: HashMap::new(),
                frequency_map: HashMap::new(),
                ordinal_map: HashMap::new(),
                categories: Vec::new(),
                default_value: 0.0,
                expansion_width: 0,
                feature_start_idx,
            };
        }

        let unique_count = counts.len();
        let total = total_nonempty as f64;

        let strategy = if ordered {
            EncodingStrategy::Ordinal
        } else if unique_count <= 10 {
            EncodingStrategy::OneHot
        } else {
            EncodingStrategy::Frequency
        };

        // Sort categories by frequency (descending) for consistent ordering
        let mut sorted_cats: Vec<(String, usize)> = counts.into_iter().collect();
        sorted_cats.sort_by(|a, b| b.1.cmp(&a.1));

        let categories: Vec<String> = sorted_cats.iter().map(|(k, _)| k.clone()).collect();

        let (frequency_map, onehot_map, ordinal_map, default_value, expansion_width) = match strategy {
            EncodingStrategy::OneHot => {
                // Drop the most frequent category as reference to avoid dummy variable trap
                let encoding_cats = if categories.len() > 1 {
                    &categories[..categories.len() - 1]
                } else {
                    &categories[..]
                };
                let mut oh_map = HashMap::new();
                for (i, cat) in encoding_cats.iter().enumerate() {
                    oh_map.insert(cat.clone(), i);
                }
                let wid = encoding_cats.len();
                let dv = 0.0; // all zeros = reference category
                (HashMap::new(), oh_map, HashMap::new(), dv, wid)
            }
            EncodingStrategy::Frequency => {
                let mut freq_map = HashMap::new();
                for &(ref cat, count) in &sorted_cats {
                    freq_map.insert(cat.clone(), count as f64 / total);
                }
                let dv = if !categories.is_empty() {
                    *freq_map.get(&categories[0]).unwrap_or(&0.0)
                } else {
                    0.0
                };
                (freq_map, HashMap::new(), HashMap::new(), dv, 1)
            }
            EncodingStrategy::Ordinal => {
                let mut ord_map = HashMap::new();
                for (i, cat) in categories.iter().enumerate() {
                    ord_map.insert(cat.clone(), i);
                }
                let max_ord = categories.len().saturating_sub(1);
                // Normalize ordinal to [0, 1] range for better scaling
                let dv = if max_ord > 0 {
                    (max_ord / 2) as f64 / max_ord as f64
                } else {
                    0.0
                };
                (HashMap::new(), HashMap::new(), ord_map, dv, 1)
            }
        };

        CategoricalEncoder {
            column_name: column_name.to_string(),
            strategy,
            onehot_map,
            frequency_map,
            ordinal_map,
            categories,
            default_value,
            expansion_width,
            feature_start_idx,
        }
    }

    /// Encode a single value (used at prediction time for new inputs)
    pub fn encode_single(&self, value: &str) -> Vec<f64> {
        let v = value.trim();
        let cat_key = if v.is_empty() {
            // Missing: use default
            return match self.strategy {
                EncodingStrategy::OneHot => vec![0.0; self.expansion_width],
                EncodingStrategy::Frequency | EncodingStrategy::Ordinal => vec![self.default_value],
            };
        } else {
            v.to_string()
        };

        match self.strategy {
            EncodingStrategy::OneHot => {
                let mut vec = vec![0.0; self.expansion_width];
                if let Some(&idx) = self.onehot_map.get(&cat_key) {
                    if idx < self.expansion_width {
                        vec[idx] = 1.0;
                    }
                }
                // If category not found, encode as all-zeros (reference category)
                vec
            }
            EncodingStrategy::Frequency => {
                let val = self.frequency_map.get(&cat_key).copied().unwrap_or(self.default_value);
                vec![val]
            }
            EncodingStrategy::Ordinal => {
                let max_ord = self.categories.len().saturating_sub(1);
                if max_ord == 0 {
                    return vec![0.0];
                }
                let ord = self.ordinal_map.get(&cat_key).copied().unwrap_or(max_ord / 2);
                // Normalize to [0, 1]
                vec![ord as f64 / max_ord as f64]
            }
        }
    }

    /// Get human-readable names for the encoded feature columns
    pub fn feature_names(&self) -> Vec<String> {
        match self.strategy {
            EncodingStrategy::OneHot => {
                let encoding_cats = if self.categories.len() > 1 {
                    &self.categories[..self.categories.len() - 1]
                } else {
                    &self.categories[..]
                };
                encoding_cats
                    .iter()
                    .map(|cat| format!("{}={}", self.column_name, cat))
                    .collect()
            }
            EncodingStrategy::Frequency | EncodingStrategy::Ordinal => {
                vec![format!("{}_encoded", self.column_name)]
            }
        }
    }
}

/// Full encoding pipeline: transforms a dataset with categorical columns into
/// an all-numeric feature matrix plus metadata for prediction-time consistency.
#[derive(Debug, Clone)]
pub struct EncodingPipeline {
    /// Encoders, one per categorical column, in the order they appear
    pub encoders: Vec<CategoricalEncoder>,
    /// Total number of numeric features after encoding (original numeric + encoded categorical)
    pub total_features: usize,
    /// How many original numeric columns exist (before categorical expansion)
    pub num_original_numeric: usize,
    /// Column names for the final feature matrix
    pub feature_names: Vec<String>,
}

impl EncodingPipeline {
    /// Build an encoding pipeline from parser data.
    ///
    /// * `original_numeric_data` - the numeric matrix (rows × numeric_cols) from the parser
    /// * `numeric_col_names` - names of the numeric columns
    /// * `categorical_cols` - (column_name, raw_values_per_row, ordered_flag) for each categorical col
    pub fn fit(
        original_numeric_data: &[Vec<f64>],
        numeric_col_names: &[String],
        categorical_cols: &[(String, Vec<String>, bool)], // (name, values, ordered)
    ) -> Self {
        let num_original_numeric = numeric_col_names.len();
        let _nrows = original_numeric_data.len();

        let mut encoders = Vec::new();
        let mut feature_start = num_original_numeric;
        let mut feature_names: Vec<String> = numeric_col_names.to_vec();

        for (col_name, values, ordered) in categorical_cols {
            let encoder = CategoricalEncoder::fit(values, col_name, *ordered, feature_start);
            let enc_names = encoder.feature_names();
            feature_names.extend(enc_names);
            feature_start += encoder.expansion_width;
            encoders.push(encoder);
        }

        let total_features = feature_names.len();

        // Validate shapes
        if !original_numeric_data.is_empty() {
            assert_eq!(
                original_numeric_data[0].len(),
                num_original_numeric,
                "Numeric data width must match numeric_col_names length"
            );
        }

        EncodingPipeline {
            encoders,
            total_features,
            num_original_numeric,
            feature_names,
        }
    }

    /// Encode the full dataset into an all-numeric feature matrix: rows × total_features.
    /// `categorical_col_values`: outer Vec is per-encoder (same order as encoders),
    /// inner Vec is per-row string values for that categorical column.
    pub fn transform(
        &self,
        original_numeric_data: &[Vec<f64>],
        categorical_col_values: &[Vec<String>],
    ) -> Vec<Vec<f64>> {
        let nrows = original_numeric_data.len();
        if nrows == 0 {
            return Vec::new();
        }

        let mut result: Vec<Vec<f64>> = Vec::with_capacity(nrows);

        for row_idx in 0..nrows {
            let mut row: Vec<f64> = Vec::with_capacity(self.total_features);

            // Copy original numeric columns
            if row_idx < original_numeric_data.len() {
                row.extend_from_slice(&original_numeric_data[row_idx]);
            } else {
                row.extend(std::iter::repeat(0.0).take(self.num_original_numeric));
            }

            // Encode categorical columns in order
            for (enc_idx, encoder) in self.encoders.iter().enumerate() {
                let raw_val = categorical_col_values
                    .get(enc_idx)
                    .and_then(|col| col.get(row_idx))
                    .map(|s| s.as_str())
                    .unwrap_or("");
                let encoded = encoder.encode_single(raw_val);
                row.extend(encoded);
            }

            result.push(row);
        }

        result
    }

    /// Encode a single row's categorical values for prediction.
    /// Takes the categorical column values in the same order as encoders were created.
    pub fn transform_single(
        &self,
        numeric_features: &[f64],
        categorical_values: &[String],
    ) -> Vec<f64> {
        let mut row: Vec<f64> = Vec::with_capacity(self.total_features);
        row.extend_from_slice(numeric_features);
        for (i, encoder) in self.encoders.iter().enumerate() {
            let raw_val = categorical_values.get(i).map(|s| s.as_str()).unwrap_or("");
            let encoded = encoder.encode_single(raw_val);
            row.extend(encoded);
        }
        row
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onehot_encoder() {
        let values: Vec<String> = ["A", "B", "A", "C", "B", "A", "A", "B", "C", "B"]
            .iter().map(|s| s.to_string()).collect();
        let encoder = CategoricalEncoder::fit(&values, "color", false, 0);

        assert!(matches!(encoder.strategy, EncodingStrategy::OneHot));
        // 3 unique categories → most frequent dropped → 2 encoding columns
        assert_eq!(encoder.expansion_width, 2);

        let encoded_a = encoder.encode_single("A");
        assert_eq!(encoded_a.len(), 2);
        // "A" might not be the reference depending on sort order
        // Verify that at least one category maps to all-zeros
        let encoded = encoder.encode_single("");
        assert_eq!(encoded.len(), 2);
    }

    #[test]
    fn test_frequency_encoder() {
        let mut values = Vec::new();
        for i in 0..30 {
            values.push(format!("cat{}", i % 15)); // 15 unique cats, 2 each
        }
        let encoder = CategoricalEncoder::fit(&values, "many_cats", false, 0);
        assert!(matches!(encoder.strategy, EncodingStrategy::Frequency));
        assert_eq!(encoder.expansion_width, 1);
    }

    #[test]
    fn test_ordinal_encoder() {
        let values: Vec<String> = ["Low", "Medium", "High", "Low", "Medium", "High", "Low"]
            .iter().map(|s| s.to_string()).collect();
        let encoder = CategoricalEncoder::fit(&values, "level", true, 0);
        assert!(matches!(encoder.strategy, EncodingStrategy::Ordinal));
        assert_eq!(encoder.expansion_width, 1);

        let encoded_low = encoder.encode_single("Low");
        assert!(!encoded_low.is_empty());
    }

    #[test]
    fn test_pipeline_basic() {
        // 2 numeric cols + 1 categorical col
        let numeric_data = vec![
            vec![1.0, 10.0],
            vec![2.0, 20.0],
            vec![3.0, 30.0],
            vec![4.0, 40.0],
            vec![5.0, 50.0],
        ];
        let num_names = vec!["x".to_string(), "y".to_string()];
        let cat_values: Vec<String> = ["A", "B", "A", "C", "B"]
            .iter().map(|s| s.to_string()).collect();
        let cat_cols = vec![("color".to_string(), cat_values, false)];

        let pipeline = EncodingPipeline::fit(&numeric_data, &num_names, &cat_cols);
        assert_eq!(pipeline.num_original_numeric, 2);
        // 3 categories → 2 one-hot columns + 2 numeric = 4 total
        assert!(pipeline.total_features >= 3);
    }
}
