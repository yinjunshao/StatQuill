use crate::display::{DisplayEngine, PredictionDisplay};
use crate::math::autoregressive::AutoregressiveModel;
pub use crate::math::linear_regression::MatrixLinearRegression;
use crate::math::stationarity;
use crate::parser::DataParser;
use anyhow::Result;
use nalgebra::DMatrix;
use std::collections::HashMap;

pub struct PredictionLoop;

impl PredictionLoop {
    pub fn run(
        parser: &DataParser,
        display: &mut DisplayEngine,
        numeric_df: &DMatrix<f64>,
    ) -> Result<()> {
        let (_nrows, ncols) = numeric_df.shape();
        if ncols < 2 {
            display.print_warning("Need at least 2 numeric columns for prediction.");
            return Ok(());
        }

        display.print_status("Preprocessing data (median imputation + Winsorization + robust scaling)...");

        // ── Robust Preprocessing Pipeline ──
        // Step 1: Median imputation for missing values
        let imputation = crate::math::imputation::median_imputation(numeric_df);
        if imputation.missing_counts.iter().any(|&c| c > 0) {
            display.print_info(&format!(
                "  ✓ Median imputation: {} missing values filled across {} columns",
                imputation.missing_counts.iter().sum::<usize>(),
                imputation.missing_counts.iter().filter(|&&c| c > 0).count()
            ));
        }

        // Step 2: Winsorization + robust scaling
        let preprocessed = crate::math::robust::robust_preprocess(&imputation.imputed_data);
        if preprocessed.capped_counts.iter().any(|&c| c > 0) {
            display.print_info(&format!(
                "  ✓ Winsorization: {} extreme values capped at [5%, 95%]",
                preprocessed.capped_counts.iter().sum::<usize>()
            ));
        }
        display.print_status("Training models on preprocessed data...");

        let numeric_cols = &parser.numeric_columns;

        // Train models: one model per target column using all other columns as features
        let mut models: HashMap<String, MatrixLinearRegression> = HashMap::new();

        for (target_idx, target_name) in numeric_cols.iter().enumerate() {
            let feature_indices: Vec<usize> = (0..ncols).filter(|&i| i != target_idx).collect();

            let mut model = MatrixLinearRegression::new(1e-5);
            model.fit(
                &preprocessed.preprocessed,
                target_idx,
                &feature_indices,
                numeric_cols,
                target_name,
            );
            display.print_text(&format!(
                "  ✓ Model trained: {} ← {:?}",
                target_name, model.feature_cols
            ));
            models.insert(target_name.clone(), model);
        }

        // Interactive prediction loop
        loop {
            display.print_separator();
            println!("\x1b[1mPREDICTION MODE\x1b[0m");
            println!("Enter known values. Leave blank to predict. Type 'q' to quit.");
            println!("{}", "═".repeat(60));

            println!(
                "\n\x1b[2mAvailable columns: {}\x1b[0m",
                numeric_cols.join(", ")
            );
            println!("\x1b[2mData range: {} rows\x1b[0m", _nrows);

            // Collect user inputs
            let mut user_inputs: HashMap<String, f64> = HashMap::new();
            let mut quit = false;

            for col in numeric_cols {
                let prompt = format!("  {}", col);
                let val = display.prompt_input(&prompt)?;

                if val.to_lowercase() == "q" {
                    quit = true;
                    break;
                }

                if !val.is_empty() {
                    match val.parse::<f64>() {
                        Ok(n) => {
                            user_inputs.insert(col.clone(), n);
                        }
                        Err(_) => {
                            display.print_warning(&format!(
                                "Invalid number for {}, skipping.",
                                col
                            ));
                        }
                    }
                }
            }

            if quit {
                break;
            }

            if user_inputs.is_empty() {
                display
                    .print_warning("No values entered. Please provide at least one known value.");
                continue;
            }

            // Predict missing columns
            let mut predictions: Vec<PredictionDisplay> = Vec::new();

            for target_name in numeric_cols {
                if user_inputs.contains_key(target_name) {
                    continue; // User already provided this value
                }

                if let Some(model) = models.get(target_name) {
                    // Build feature vector from known values (use column mean if unknown)
                    let feature_values: Vec<f64> = model
                        .feature_cols
                        .iter()
                        .map(|feat| {
                            user_inputs.get(feat).copied().unwrap_or_else(|| {
                                // Find column mean by matching the feature name
                                if let Some(col_meta) =
                                    parser.columns.iter().find(|c| &c.name == feat)
                                {
                                    col_meta.mean.unwrap_or(0.0)
                                } else {
                                    0.0
                                }
                            })
                        })
                        .collect();

                    let (pred, lower, upper, cv) = model.predict(&feature_values);

                    predictions.push(PredictionDisplay {
                        target: target_name.clone(),
                        value: pred,
                        lower,
                        upper,
                        cv,
                    });
                }
            }

            if predictions.is_empty() {
                display.print_warning("No predictions could be generated.");
                display.print_text("All columns either provided or no models available.");
            } else {
                display.show_predictions(&predictions);
            }

            if !display.confirm("Make another prediction?", true) {
                break;
            }
        }

        Ok(())
    }
}

pub struct TimeSeriesLoop;

impl TimeSeriesLoop {
    pub fn run(parser: &DataParser, display: &mut DisplayEngine) -> Result<()> {
        let time_col = parser
            .time_column
            .as_ref()
            .cloned()
            .unwrap_or_else(|| "time".to_string());

        let numeric_cols = &parser.numeric_columns;

        display.print_info(&format!("Time Series Mode (Time column: {})", time_col));

        // Filter targets (exclude time column if it's also numeric)
        let targets: Vec<&String> = numeric_cols
            .iter()
            .filter(|c| *c != &time_col)
            .collect();

        if targets.is_empty() {
            display.print_warning("No valid target columns found.");
            return Ok(());
        }

        println!("\nAvailable targets:");
        for (i, t) in targets.iter().enumerate() {
            println!("  {}. {}", i + 1, t);
        }

        let choice = display.prompt_input(&format!(
            "Select target column (number or name, default: {})",
            targets[0]
        ))?;

        let target_name: String = if choice.is_empty() {
            targets[0].clone().to_string()
        } else if let Ok(idx) = choice.parse::<usize>() {
            if idx >= 1 && idx <= targets.len() {
                targets[idx - 1].clone().to_string()
            } else {
                targets
                    .iter()
                    .find(|t| t.as_str() == choice)
                    .map(|&t| t.clone())
                    .unwrap_or_else(|| targets[0].clone().to_string())
            }
        } else {
            targets
                .iter()
                .find(|t| t.as_str() == choice)
                .map(|&t| t.clone())
                .unwrap_or_else(|| targets[0].clone().to_string())
        };

        display.print_status(&format!(
            "Extracting time series data for '{}'...",
            target_name
        ));

        // Get the numeric data values for the target
        let numeric_data = parser.numeric_data_values();
        if numeric_data.is_empty() {
            display.print_warning("No numeric data available.");
            return Ok(());
        }

        // Find the index of the target in numeric_columns
        let target_idx = numeric_cols.iter().position(|c| c == &target_name);
        let target_idx = match target_idx {
            Some(i) => i,
            None => {
                display.print_warning(&format!(
                    "Column '{}' not found in numeric data.",
                    target_name
                ));
                return Ok(());
            }
        };

        let series: Vec<f64> = numeric_data.iter().map(|row| row[target_idx]).collect();

        // ── Stationarity Testing & Differencing ──
        display.print_status("Testing for stationarity (Augmented Dickey-Fuller)...");
        let adf = stationarity::adf_test(&series);
        display.print_info(&format!(
            "  ADF test: τ = {:.3}, p ≈ {:.3}, lag = {}, {}",
            adf.test_statistic,
            adf.p_value,
            adf.used_lag,
            if adf.is_stationary {
                "stationary ✓"
            } else {
                "non-stationary — will difference"
            }
        ));

        let (working_series, diff_order, original_for_undo) = if adf.is_stationary {
            (series.clone(), 0usize, series.clone())
        } else {
            display.print_status("Differencing series to achieve stationarity...");
            let diff_result = stationarity::difference_to_stationarity(&series);
            display.print_info(&format!(
                "  Differencing order: d={} ({} → {} observations)",
                diff_result.order,
                series.len(),
                diff_result.differenced_series.len()
            ));
            if let Some(ref final_adf) = diff_result.final_adf {
                display.print_info(&format!(
                    "  After differencing: τ = {:.3}, {}",
                    final_adf.test_statistic,
                    if final_adf.is_stationary {
                        "stationary ✓"
                    } else {
                        "still non-stationary (proceeding with caution)"
                    }
                ));
            }
            (diff_result.differenced_series, diff_result.order, series.clone())
        };

        display.print_status(&format!(
            "Fitting AR model to {} observations...",
            working_series.len()
        ));

        let max_lag = (10.min(working_series.len() / 4)).max(1);
        let mut ar_model = AutoregressiveModel::new(max_lag);

        match ar_model.fit(&working_series) {
            Ok(result) => {
                display.show_ar_results(&result, ar_model.p, &ar_model.phi, ar_model.c);

                let steps_str = display.prompt_input("How many future steps to predict?")?;
                let steps: usize = steps_str.parse().unwrap_or(1);

                match ar_model.predict_next(&working_series, steps) {
                    Ok((preds, lower, upper)) => {
                        // Undo differencing to get predictions in original scale
                        let (level_preds, level_lower, level_upper) = if diff_order > 0 {
                            let lp = stationarity::undo_difference(&original_for_undo, &preds, diff_order);
                            let ll = stationarity::undo_difference(&original_for_undo, &lower, diff_order);
                            let lu = stationarity::undo_difference(&original_for_undo, &upper, diff_order);
                            (lp, ll, lu)
                        } else {
                            (preds, lower, upper)
                        };
                        if diff_order > 0 {
                            display.print_info(&format!(
                                "  Predictions restored to original scale (d={})",
                                diff_order
                            ));
                        }
                        display.show_future_predictions(&target_name, &level_preds, &level_lower, &level_upper);
                    }
                    Err(e) => {
                        display.print_warning(&format!("Prediction failed: {}", e));
                    }
                }
            }
            Err(e) => {
                display.print_warning(&format!("AR model failed: {}", e));
                display.print_info("Falling back to Matrix Regression...");

                // Fallback: use Matrix Regression
                let numeric_df = parser.get_numeric_matrix();
                PredictionLoop::run(parser, display, &numeric_df)?;
            }
        }

        Ok(())
    }
}
