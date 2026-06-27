mod input;

use crate::ai;
use crate::commentary;
use crate::config::ConfigManager;
use crate::display::{self};
use crate::math;
use crate::math::model_selection::{self, DataDiagnostics};
use crate::parser;
use crate::prediction;
use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use nalgebra::DMatrix;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, Paragraph, Tabs, Wrap},
    Frame,
};
use std::collections::HashMap;

// ── Application State ──
pub enum AppState {
    Banner,
    SetupApiKey,
    SetupModel,
    FileInput,
    ContextInput,
    Analyzing,
    Results,
    PredictionMode,
    SigLevelInput,
    Done,
}

#[derive(PartialEq)]
pub(crate) enum ResultsTab {
    Overview,
    Correlations,
    AIAnalysis,
    Diagnostics,
}

pub struct App {
    pub state: AppState,
    // Input buffers
    pub api_key_input: String,
    pub api_key_cursor: usize,
    pub model_input: String,
    pub model_cursor: usize,
    pub file_path_input: String,
    pub file_path_cursor: usize,
    pub context_input: String,
    pub context_cursor: usize,
    // Config
    pub cfg: ConfigManager,
    pub model_override: Option<String>,
    // Analysis results
    pub analysis_messages: Vec<String>,
    pub ai_commentary: Option<String>,
    pub data_summary_lines: Vec<String>,
    pub correlation_lines: Vec<String>,
    pub parser: Option<parser::DataParser>,
    pub numeric_df: Option<DMatrix<f64>>,
    pub selected_model: Option<model_selection::ModelType>,
    pub data_diagnostics: Option<DataDiagnostics>,
    // Results tabs
    pub active_tab: ResultsTab,
    // Prediction mode
    pub prediction_inputs: HashMap<String, String>,
    pub prediction_focus: usize,
    pub prediction_results: Vec<display::PredictionDisplay>,
    pub prediction_message: String,
    // Scroll offset for Results tabs
    pub scroll_offset: usize,
    // Anomaly diagnostics
    pub diagnostic_lines: Vec<String>,
    // Significance level input
    pub sig_level_input: String,
    pub sig_level_cursor: usize,
    // Status
    pub status_message: String,
    pub should_exit: bool,
    // Chat state for AI Analysis tab
    pub chat_input: String,
    pub chat_cursor: usize,
    pub chat_history: Vec<ai::ChatMessage>, // user + assistant exchanges (after initial commentary)
    pub chat_waiting: bool,                 // true while waiting for AI response
    pub chat_status: String,                // status message shown above chat input
    pub chat_focused: bool,                 // true when chat textbox is active for typing
}

impl App {
    pub fn new(cfg: ConfigManager, model_override: Option<String>) -> Self {
        let default_model = cfg.model().to_string();
        Self {
            state: AppState::Banner,
            api_key_input: String::new(),
            api_key_cursor: 0,
            model_input: default_model,
            model_cursor: 0,
            file_path_input: String::new(),
            file_path_cursor: 0,
            context_input: String::new(),
            context_cursor: 0,
            cfg,
            model_override,
            analysis_messages: Vec::new(),
            ai_commentary: None,
            data_summary_lines: Vec::new(),
            correlation_lines: Vec::new(),
            parser: None,
            numeric_df: None,
            selected_model: None,
            data_diagnostics: None,
            active_tab: ResultsTab::Overview,
            prediction_inputs: HashMap::new(),
            prediction_focus: 0,
            prediction_results: Vec::new(),
            prediction_message: String::new(),
            scroll_offset: 0,
            diagnostic_lines: Vec::new(),
            sig_level_input: String::from("0.05"),
            sig_level_cursor: 0,
            status_message: String::new(),
            should_exit: false,
            chat_input: String::new(),
            chat_cursor: 0,
            chat_history: Vec::new(),
            chat_waiting: false,
            chat_status: String::new(),
            chat_focused: false,
        }
    }

    /// Set up analysis screen with time estimates, then draw and run.
    /// Called from handle_key when Enter is pressed on context screen.
    pub fn enter_analysis(&mut self) {
        self.analysis_messages.clear();
        self.status_message.clear();
        let is_small = self.file_path_input.len() < 100;
        self.analysis_messages.push(format!(
            "  ◌ Parsing data file... (< {} sec)",
            if is_small { "1" } else { "3" }
        ));
        self.analysis_messages.push(format!(
            "  ◌ Computing correlations... (< {} sec)",
            if is_small { "1" } else { "2" }
        ));
        if self.cfg.is_configured() {
            self.analysis_messages.push(format!(
                "  ◌ Consulting AI for insights... (< {} sec)",
                if is_small { "5" } else { "15" }
            ));
        }
        self.state = AppState::Analyzing;
    }

    /// Run analysis synchronously and populate results.
    /// This is called from the event loop AFTER the analyzing screen has been drawn.
    fn run_analysis(&mut self) {
        let filepath = self.file_path_input.clone();
        let context = self.context_input.clone();
        let model = self
            .model_override
            .clone()
            .unwrap_or_else(|| self.cfg.model().to_string());

        // Parse data
        let pr = match parser::DataParser::load(&filepath) {
            Ok(p) => p,
            Err(e) => {
                self.analysis_messages.push(format!("  ✗ Error: {}", e));
                self.status_message = format!("Error: {}", e);
                self.state = AppState::FileInput;
                return;
            }
        };

        self.analysis_messages.push("  ✓ Data parsed successfully.".to_string());

        // Build summary
        let num_cols_str = if pr.numeric_columns.is_empty() {
            "None".to_string()
        } else {
            pr.numeric_columns.join(", ")
        };
        let mut summary = vec![
            format!("File: {}", pr.filepath),
            format!("Rows: {}", pr.rows.len()),
            format!("Columns: {}", pr.headers.len()),
            format!(
                "Time Column: {}",
                pr.time_column
                    .as_deref()
                    .unwrap_or("None detected")
            ),
            format!("Numeric Cols: {}", num_cols_str),
            String::new(),
            "Column Analysis:".to_string(),
        ];
        // Separate numeric and non-numeric for cleaner display
        let num_cols: Vec<_> = pr.columns.iter().filter(|c| c.dtype == parser::ColumnType::Numeric).collect();
        let other_cols: Vec<_> = pr.columns.iter().filter(|c| c.dtype != parser::ColumnType::Numeric).collect();

        for meta in &num_cols {
            let mean_str = meta.mean.map_or("—".to_string(), |m| format!("mean={:.2}", m));
            let std_str = meta.std.map_or("".to_string(), |s| format!(" σ={:.2}", s));
            let range = match (meta.min_val, meta.max_val) {
                (Some(lo), Some(hi)) => format!(" [{:.1}–{:.1}]", lo, hi),
                _ => String::new(),
            };
            summary.push(format!(
                "  {:20} {:>8}  Empty cells: {:>3}  {}{}{}",
                crate::display::trunc(&meta.name, 19),
                "Numeric",
                meta.missing_count,
                mean_str,
                std_str,
                range,
            ));
        }
        if !other_cols.is_empty() {
            summary.push(String::from("  ─────────────────────────────────────"));
            for meta in &other_cols {
                let type_str = match meta.dtype {
                    parser::ColumnType::Categorical => "Categorical",
                    parser::ColumnType::DateTime => "DateTime",
                    _ => "String",
                };
                let detail = if meta.dtype == parser::ColumnType::DateTime || meta.dtype == parser::ColumnType::Categorical {
                    format!("{} unique", meta.unique_count)
                } else {
                    format!("{} unique", meta.unique_count)
                };
                summary.push(format!(
                    "  {:20} {:>12}  Empty cells: {:>3}  {}",
                    crate::display::trunc(&meta.name, 19),
                    type_str,
                    meta.missing_count,
                    detail,
                ));
            }
        }
        self.data_summary_lines = summary;

        // Numeric matrix for correlation & prediction
        let numeric_df = pr.get_numeric_matrix();

        // Compute correlations
        if numeric_df.ncols() >= 2 {
            let (cov, cols, corr) = math::covariance::CovarianceEngine::compute(&numeric_df);
            self.analysis_messages.push("  ✓ Correlations computed.".to_string());

            let n = cols.len();
            let mut corr_lines = Vec::new();
            if n > 0 {
                let sd: Vec<f64> = (0..n).map(|j| cov[(j, j)].sqrt()).collect();
                let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
                for i in 0..n {
                    for j in i + 1..n {
                        let d = sd[i] * sd[j];
                        let r = if d > 1e-15 { cov[(i, j)] / d } else { 0.0 };
                        if r.abs() > 0.3 {
                            pairs.push((i, j, r));
                        }
                    }
                }
                pairs.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());

                if pairs.is_empty() {
                    corr_lines.push("  No strong correlations found.".to_string());
                } else {
                    corr_lines.push(format!(
                        "{:<22} {:<22} {:>12}",
                        "Variable A", "Variable B", "Correlation"
                    ));
                    for (i, j, r) in pairs.iter().take(25) {
                        // Map generic col0/col1 to actual column names from the parser
                        let a = pr.numeric_columns.get(*i).cloned()
                            .unwrap_or_else(|| cols.get(*i).cloned().unwrap_or(format!("col{}", i)));
                        let b = pr.numeric_columns.get(*j).cloned()
                            .unwrap_or_else(|| cols.get(*j).cloned().unwrap_or(format!("col{}", j)));
                        corr_lines.push(format!(
                            "  {:<20} {:<20} {:>12.4}",
                            crate::display::trunc(&a, 20),
                            crate::display::trunc(&b, 20),
                            r
                        ));
                    }
                }

                let collinear =
                    math::covariance::CovarianceEngine::find_collinear_pairs(&corr, &cols, 0.9);
                if !collinear.is_empty() {
                    corr_lines.push(String::new());
                    corr_lines.push("⚠ High multicollinearity detected:".to_string());
                    for (a, b, r) in &collinear {
                        corr_lines.push(format!("  {} ↔ {}: r={:.3}", a, b, r));
                    }
                }
            } else {
                corr_lines.push("  No numeric columns for correlation.".to_string());
            }
            self.correlation_lines = corr_lines;
        }

        // ── Model Selection Router ──
        // Build diagnostics and select the appropriate model pipeline
        let (_cov_for_diag, _cols_for_diag, corr_for_diag) =
            if numeric_df.ncols() >= 2 {
                let (cov, cols, corr) = math::covariance::CovarianceEngine::compute(&numeric_df);
                (Some(cov), Some(cols), Some(corr))
            } else {
                (None, None, None)
            };

        let diagnostics = DataDiagnostics::from_data(
            &numeric_df,
            pr.has_time_data(),
            pr.time_column.clone(),
            corr_for_diag.as_ref(),
        );

        let selected_model = model_selection::select_model(&diagnostics);
        let model_explanation = model_selection::explain_selection(&selected_model, &diagnostics);

        // Store for later use by fallback commentary and prediction
        self.selected_model = Some(selected_model.clone());
        self.data_diagnostics = Some(diagnostics.clone());

        // Inject model selection report into data summary
        self.data_summary_lines.push(String::new());
        self.data_summary_lines.push("── Model Selection Diagnostics ──".to_string());
        for line in model_explanation.lines() {
            self.data_summary_lines.push(format!("  {}", line));
        }

        // Build stats payload for AI
        let stats_payload = pr.build_stats_payload(&numeric_df);

        // AI Commentary
        if self.cfg.is_configured() {
            let ai_enhancer = ai::AIEnhancer::new(self.cfg.api_key().unwrap(), &model);
            match ai_enhancer.generate_commentary(&stats_payload, &context) {
                Ok(commentary) => {
                    self.ai_commentary = Some(commentary);
                    self.analysis_messages.push("  ✓ AI analysis complete.".to_string());
                }
                Err(e) => {
                    self.analysis_messages
                        .push(format!("  ✗ AI unavailable: {}", e));
                }
            }
        }

        // ── Anomaly Diagnostics ──
        // Generate per-cell anomaly report (missing values & outliers)
        let mut diag = Vec::new();
        diag.push("── Per-Cell Anomaly Report ──".to_string());
        diag.push(String::new());
        diag.push("Legend: -> = empty cell (imputed)   !! = outlier (Winsorized)".to_string());
        diag.push(String::new());

        let mut any_anomaly = false;

        // 1. Empty cells in numeric columns
        let numeric_df2 = pr.get_numeric_matrix();
        let imputation = math::imputation::median_imputation(&numeric_df2);
        if imputation.missing_counts.iter().any(|&c| c > 0) {
            any_anomaly = true;
            diag.push("Empty Cells (Median Imputation):".to_string());
            diag.push(format!("{:<6} {:<20} {:<18} {:>12} {:>8}", "Row", "Column", "Original", "Imputed", "Imputed?"));
            diag.push("─".repeat(80).to_string());

            for row_idx in 0..imputation.imputed_data.nrows() {
                if imputation.row_has_missing[row_idx] {
                    for col_j in 0..imputation.imputed_data.ncols() {
                        if imputation.missingness_indicators[(row_idx, col_j)] > 0.5 {
                            let col_name = pr.numeric_columns.get(col_j)
                                .cloned()
                                .unwrap_or_else(|| format!("col{}", col_j));
                        diag.push(format!(
                            "  {:>4}  {:<20} {:<18} {:>12.4}   -> YES",
                                row_idx + 1,
                                crate::display::trunc(&col_name, 19),
                                "(empty)",
                                imputation.imputed_data[(row_idx, col_j)],
                            ));
                        }
                    }
                }
            }
            diag.push(String::new());
        }

        // 2. Outliers in numeric columns
        let winsor = math::robust::winsorize(&imputation.imputed_data, 0.05, 0.95);
        if winsor.capped_counts.iter().any(|&c| c > 0) {
            any_anomaly = true;
            diag.push("Outliers (Winsorized at [5%, 95%]):".to_string());
            diag.push(format!("{:<6} {:<20} {:>12} {:>12} {:>14} {:>8}", "Row", "Column", "Original", "Capped To", "Threshold Range", "Action"));
            diag.push("─".repeat(80).to_string());

            for row_idx in 0..winsor.winsorized_data.nrows() {
                if winsor.row_has_outliers[row_idx] {
                    for col_j in 0..winsor.winsorized_data.ncols() {
                        let orig = imputation.imputed_data[(row_idx, col_j)];
                        let capped = winsor.winsorized_data[(row_idx, col_j)];
                        if (orig - capped).abs() > 1e-10 {
                            let col_name = pr.numeric_columns.get(col_j)
                                .cloned()
                                .unwrap_or_else(|| format!("col{}", col_j));
                            let lo = winsor.lower_thresholds.get(col_j).copied().unwrap_or(0.0);
                            let hi = winsor.upper_thresholds.get(col_j).copied().unwrap_or(0.0);
                            let direction = if orig < lo { "v capped" } else { "^ capped" };
                        diag.push(format!(
                            "  {:>4}  {:<20} {:>12.4} {:>12.4}   [{:.1}, {:.1}]  !! {}",
                                row_idx + 1,
                                crate::display::trunc(&col_name, 19),
                                orig,
                                capped,
                                lo,
                                hi,
                                direction,
                            ));
                        }
                    }
                }
            }
            diag.push(String::new());
        }

        // 3. Empty cells in non-numeric columns
        for col_idx in 0..pr.headers.len() {
            if pr.columns[col_idx].dtype != parser::ColumnType::Numeric && pr.columns[col_idx].missing_count > 0 {
                any_anomaly = true;
                let col_name = &pr.headers[col_idx];
                diag.push(format!("Empty cells in '{}' ({}):", col_name,
                    match pr.columns[col_idx].dtype {
                        parser::ColumnType::Categorical => "Categorical",
                        parser::ColumnType::DateTime => "DateTime",
                        _ => "String",
                    }));
                for (row_idx, row) in pr.rows.iter().enumerate() {
                    if let Some(cell) = row.get(col_idx) {
                        if cell.is_empty() {
                        diag.push(format!("  Row {:>4}:  \"{}\"  -> (empty string)", row_idx + 1, col_name));
                        }
                    }
                }
                diag.push(String::new());
            }
        }

        if !any_anomaly {
            diag.push("✓ No anomalies detected — all values present, no outliers.".to_string());
        }
        self.diagnostic_lines = diag;

        self.parser = Some(pr);
        self.numeric_df = Some(numeric_df);
        self.state = AppState::Results;
    }

    /// Run prediction with current inputs
    fn run_prediction(&mut self) {
        let parser = match &self.parser {
            Some(p) => p,
            None => {
                self.prediction_message =
                    "No data loaded. Please load a file first.".to_string();
                return;
            }
        };
        let numeric_df = match &self.numeric_df {
            Some(df) => df,
            None => {
                self.prediction_message = "No numeric data available.".to_string();
                return;
            }
        };

        let numeric_cols = &parser.numeric_columns;
        if numeric_cols.is_empty() {
            self.prediction_message = "No numeric columns available for prediction.".to_string();
            return;
        }

        let (_nrows, ncols) = numeric_df.shape();
        if ncols < 2 {
            self.prediction_message =
                "Need at least 2 numeric columns for prediction.".to_string();
            return;
        }

        // Train models
        let mut models: HashMap<String, prediction::MatrixLinearRegression> = HashMap::new();
        for (target_idx, target_name) in numeric_cols.iter().enumerate() {
            let feature_indices: Vec<usize> = (0..ncols).filter(|&i| i != target_idx).collect();
            let mut model = prediction::MatrixLinearRegression::new(1e-5);
            model.fit(numeric_df, target_idx, &feature_indices, numeric_cols, target_name);
            models.insert(target_name.clone(), model);
        }

        // Parse user inputs
        let mut valid_inputs: HashMap<String, f64> = HashMap::new();
        for (col, val_str) in &self.prediction_inputs {
            let trimmed = val_str.trim();
            if !trimmed.is_empty() {
                if let Ok(n) = trimmed.parse::<f64>() {
                    valid_inputs.insert(col.clone(), n);
                }
            }
        }

        if valid_inputs.is_empty() {
            self.prediction_message =
                "No values entered. Please provide at least one known value.".to_string();
            return;
        }

        // Predict missing columns
        let mut predictions: Vec<display::PredictionDisplay> = Vec::new();
        for target_name in numeric_cols {
            if valid_inputs.contains_key(target_name) {
                continue;
            }
            if let Some(model) = models.get(target_name) {
                let feature_values: Vec<f64> = model
                    .feature_cols
                    .iter()
                    .map(|feat| {
                        valid_inputs.get(feat).copied().unwrap_or_else(|| {
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
                predictions.push(display::PredictionDisplay {
                    target: target_name.clone(),
                    value: pred,
                    lower,
                    upper,
                    cv,
                });
            }
        }

        if predictions.is_empty() {
            self.prediction_message = "All columns were provided. Nothing to predict.".to_string();
        } else {
            predictions.sort_by(|a, b| {
                a.cv
                    .partial_cmp(&b.cv)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.prediction_results = predictions;
            self.prediction_message = format!("Generated {} predictions.", self.prediction_results.len());
        }
    }

    /// Send a chat message in the AI Analysis tab
    fn send_chat_message(&mut self) {
        if self.chat_input.trim().is_empty() || !self.cfg.is_configured() {
            return;
        }

        let user_text = self.chat_input.trim().to_string();
        self.chat_input.clear();
        self.chat_cursor = 0;

        // Add user message to history
        self.chat_history.push(ai::ChatMessage {
            role: "user".to_string(),
            content: user_text.clone(),
        });

        self.chat_waiting = true;
        self.chat_status = "Waiting for AI response...".to_string();

        // Build the AI enhancer and call chat()
        let model = self
            .model_override
            .clone()
            .unwrap_or_else(|| self.cfg.model().to_string());

        let api_key = self.cfg.api_key().unwrap_or("").to_string();
        let mut enhancer = ai::AIEnhancer::new(&api_key, &model);

        // We need to rebuild prompts — use stored parser/numeric_df
        if let (Some(ref parser), Some(ref numeric_df)) = (&self.parser, &self.numeric_df) {
            let stats_payload = parser.build_stats_payload(numeric_df);
            let context = self.context_input.clone();
            enhancer.build_prompts(&stats_payload, &context);
        }

        let result = enhancer.chat(&self.chat_history);

        match result {
            Ok(response) => {
                self.chat_history.push(ai::ChatMessage {
                    role: "assistant".to_string(),
                    content: response,
                });
                self.chat_status = String::new();
            }
            Err(e) => {
                self.chat_status = format!("Error: {}", e);
            }
        }

        self.chat_waiting = false;
    }

    /// Handle keyboard events
    pub fn handle_key(&mut self, code: KeyCode) {
        match self.state {
            AppState::Banner => match code {
                KeyCode::Char('s') => {
                    self.api_key_input.clear();
                    self.api_key_cursor = 0;
                    self.state = AppState::SetupApiKey;
                }
                _ => {
                    self.state = if !self.cfg.is_configured() {
                        AppState::SetupApiKey
                    } else {
                        AppState::FileInput
                    };
                }
            },
            AppState::SetupApiKey => {
                match code {
                    KeyCode::Enter => {
                        if !self.api_key_input.is_empty() {
                            if let Err(e) = self.cfg.set_api_key(self.api_key_input.clone()) {
                                self.status_message = format!("Error saving key: {}", e);
                            } else {
                                self.state = AppState::SetupModel;
                            }
                        }
                    }
                    KeyCode::Esc => {
                        self.cfg
                            .set_api_key("sk-or-placeholder-key-for-bypass".to_string())
                            .ok();
                        self.state = AppState::SetupModel;
                    }
                    KeyCode::Backspace => {
                        if self.api_key_cursor > 0 {
                            self.api_key_cursor -= 1;
                            self.api_key_input.remove(self.api_key_cursor);
                        }
                    }
                    KeyCode::Delete => {
                        if self.api_key_cursor < self.api_key_input.len() {
                            self.api_key_input.remove(self.api_key_cursor);
                        }
                    }
                    KeyCode::Left => {
                        if self.api_key_cursor > 0 { self.api_key_cursor -= 1; }
                    }
                    KeyCode::Right => {
                        if self.api_key_cursor < self.api_key_input.len() { self.api_key_cursor += 1; }
                    }
                    KeyCode::Home => self.api_key_cursor = 0,
                    KeyCode::End => self.api_key_cursor = self.api_key_input.len(),
                    KeyCode::Char(c) => {
                        self.api_key_input.insert(self.api_key_cursor, c);
                        self.api_key_cursor += 1;
                    }
                    _ => {}
                }
            }
            AppState::SetupModel => {
                match code {
                    KeyCode::Enter => {
                        if self.model_input.is_empty() {
                            self.model_input = "anthropic/claude-3.5-sonnet".to_string();
                        }
                        if let Err(e) = self.cfg.set_model(self.model_input.clone()) {
                            self.status_message = format!("Error saving model: {}", e);
                        } else {
                            self.status_message = "Configuration saved!".to_string();
                            self.state = AppState::FileInput;
                        }
                    }
                    KeyCode::Esc => { self.state = AppState::FileInput; }
                    KeyCode::Backspace => {
                        if self.model_cursor > 0 {
                            self.model_cursor -= 1;
                            self.model_input.remove(self.model_cursor);
                        }
                    }
                    KeyCode::Delete => {
                        if self.model_cursor < self.model_input.len() { self.model_input.remove(self.model_cursor); }
                    }
                    KeyCode::Left => { if self.model_cursor > 0 { self.model_cursor -= 1; } }
                    KeyCode::Right => { if self.model_cursor < self.model_input.len() { self.model_cursor += 1; } }
                    KeyCode::Home => self.model_cursor = 0,
                    KeyCode::End => self.model_cursor = self.model_input.len(),
                    KeyCode::Char(c) => {
                        self.model_input.insert(self.model_cursor, c);
                        self.model_cursor += 1;
                    }
                    _ => {}
                }
            }
            AppState::FileInput => {
                match code {
                    KeyCode::Enter => {
                        if !self.file_path_input.is_empty() {
                            self.status_message.clear();
                            self.state = AppState::ContextInput;
                        } else {
                            self.status_message = "Please enter a file path.".to_string();
                        }
                    }
                    KeyCode::Esc => { self.should_exit = true; self.state = AppState::Done; }
                    KeyCode::Backspace => {
                        if self.file_path_cursor > 0 {
                            self.file_path_cursor -= 1;
                            self.file_path_input.remove(self.file_path_cursor);
                        }
                    }
                    KeyCode::Delete => {
                        if self.file_path_cursor < self.file_path_input.len() { self.file_path_input.remove(self.file_path_cursor); }
                    }
                    KeyCode::Left => { if self.file_path_cursor > 0 { self.file_path_cursor -= 1; } }
                    KeyCode::Right => { if self.file_path_cursor < self.file_path_input.len() { self.file_path_cursor += 1; } }
                    KeyCode::Home => self.file_path_cursor = 0,
                    KeyCode::End => self.file_path_cursor = self.file_path_input.len(),
                    KeyCode::Char(c) => {
                        self.file_path_input.insert(self.file_path_cursor, c);
                        self.file_path_cursor += 1;
                    }
                    _ => {}
                }
            }
            AppState::ContextInput => {
                match code {
                    KeyCode::Enter => {
                        if !self.file_path_input.is_empty() {
                            // Set up the analyzing state with time estimates
                            // The event loop will draw this and then call run_analysis
                            self.enter_analysis();
                        } else {
                            self.status_message = "Please enter a file path first.".to_string();
                        }
                    }
                    KeyCode::Esc => { self.state = AppState::FileInput; }
                    KeyCode::Backspace => {
                        if self.context_cursor > 0 {
                            self.context_cursor -= 1;
                            self.context_input.remove(self.context_cursor);
                        }
                    }
                    KeyCode::Delete => {
                        if self.context_cursor < self.context_input.len() { self.context_input.remove(self.context_cursor); }
                    }
                    KeyCode::Left => { if self.context_cursor > 0 { self.context_cursor -= 1; } }
                    KeyCode::Right => { if self.context_cursor < self.context_input.len() { self.context_cursor += 1; } }
                    KeyCode::Home => self.context_cursor = 0,
                    KeyCode::End => self.context_cursor = self.context_input.len(),
                    KeyCode::Char(c) => {
                        self.context_input.insert(self.context_cursor, c);
                        self.context_cursor += 1;
                    }
                    _ => {}
                }
            }
            AppState::Analyzing => {
                // Analysis is handled in the event loop — draw then run synchronously
            }
            AppState::Results => {
                // If chat is focused, handle chat input first
                if self.chat_focused && self.active_tab == ResultsTab::AIAnalysis {
                    match code {
                        KeyCode::Enter => {
                            self.send_chat_message();
                        }
                        KeyCode::Esc => {
                            self.chat_focused = false;
                            self.chat_status.clear();
                        }
                        KeyCode::Backspace => {
                            if self.chat_cursor > 0 {
                                self.chat_cursor -= 1;
                                self.chat_input.remove(self.chat_cursor);
                            }
                        }
                        KeyCode::Delete => {
                            if self.chat_cursor < self.chat_input.len() {
                                self.chat_input.remove(self.chat_cursor);
                            }
                        }
                        KeyCode::Left => {
                            if self.chat_cursor > 0 { self.chat_cursor -= 1; }
                        }
                        KeyCode::Right => {
                            if self.chat_cursor < self.chat_input.len() { self.chat_cursor += 1; }
                        }
                        KeyCode::Home => self.chat_cursor = 0,
                        KeyCode::End => self.chat_cursor = self.chat_input.len(),
                        KeyCode::Up | KeyCode::Down => {
                            self.scroll_offset = if matches!(code, KeyCode::Up) {
                                self.scroll_offset.saturating_sub(1)
                            } else {
                                self.scroll_offset + 1
                            };
                        }
                        KeyCode::PageUp => { self.scroll_offset = self.scroll_offset.saturating_sub(10); }
                        KeyCode::PageDown => { self.scroll_offset += 10; }
                        KeyCode::Char(c) => {
                            self.chat_input.insert(self.chat_cursor, c);
                            self.chat_cursor += 1;
                        }
                        _ => {}
                    }
                    return;
                }

                match code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        if self.chat_focused {
                            self.chat_focused = false;
                            self.chat_status.clear();
                        } else {
                            self.should_exit = true;
                            self.state = AppState::Done;
                        }
                    }
                    KeyCode::Char('1') => { self.chat_focused = false; self.active_tab = ResultsTab::Overview; self.scroll_offset = 0; }
                    KeyCode::Char('2') => { self.chat_focused = false; self.active_tab = ResultsTab::Correlations; self.scroll_offset = 0; }
                    KeyCode::Char('3') => { self.active_tab = ResultsTab::AIAnalysis; self.scroll_offset = 0; }
                    KeyCode::Char('4') => { self.chat_focused = false; self.active_tab = ResultsTab::Diagnostics; self.scroll_offset = 0; }
                    KeyCode::Up => { if self.scroll_offset > 0 { self.scroll_offset -= 1; } }
                    KeyCode::Down => { self.scroll_offset += 1; }
                    KeyCode::PageUp => { self.scroll_offset = self.scroll_offset.saturating_sub(10); }
                    KeyCode::PageDown => { self.scroll_offset += 10; }
                    KeyCode::Char('s') => {
                        self.api_key_input.clear();
                        self.api_key_cursor = 0;
                        self.state = AppState::SetupApiKey;
                    }
                    KeyCode::Char('a') => {
                        self.chat_focused = false;
                        self.sig_level_input = String::from("0.05");
                        self.sig_level_cursor = 0;
                        self.state = AppState::SigLevelInput;
                    }
                    KeyCode::Char('c') => {
                        // Toggle chat focus in AI Analysis tab (only if configured)
                        if self.active_tab == ResultsTab::AIAnalysis && self.cfg.is_configured() {
                            self.chat_focused = !self.chat_focused;
                            if self.chat_focused {
                                self.chat_status = "Type your question and press Enter. Esc to exit chat.".to_string();
                            } else {
                                self.chat_status.clear();
                            }
                        }
                    }
                    KeyCode::Char('p') | KeyCode::Tab => {
                        self.chat_focused = false;
                        self.prediction_inputs.clear();
                        if let Some(ref parser) = self.parser {
                            for col in &parser.numeric_columns {
                                self.prediction_inputs.insert(col.clone(), String::new());
                            }
                        }
                        self.prediction_focus = 0;
                        self.prediction_results.clear();
                        self.prediction_message.clear();
                        self.state = AppState::PredictionMode;
                    }
                    _ => {}
                }
            },
            AppState::PredictionMode => {
                let cols: Vec<String> = if let Some(ref parser) = self.parser { parser.numeric_columns.clone() } else { vec![] };
                if cols.is_empty() { self.state = AppState::Results; return; }

                match code {
                    KeyCode::Enter => { self.run_prediction(); }
                    KeyCode::Esc => { self.state = AppState::Results; }
                    KeyCode::Up => { if self.prediction_focus > 0 { self.prediction_focus -= 1; } }
                    KeyCode::Down => { if self.prediction_focus + 1 < cols.len() { self.prediction_focus += 1; } }
                    KeyCode::Backspace => {
                        if let Some(col) = cols.get(self.prediction_focus) {
                            if let Some(s) = self.prediction_inputs.get_mut(col) {
                                s.pop();
                            }
                        }
                    }
                    KeyCode::Delete => {
                        if let Some(col) = cols.get(self.prediction_focus) {
                            if let Some(s) = self.prediction_inputs.get_mut(col) { s.clear(); }
                        }
                    }
                    KeyCode::Char(c) => {
                        if c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E' {
                            if let Some(col) = cols.get(self.prediction_focus) {
                                if let Some(s) = self.prediction_inputs.get_mut(col) { s.push(c); }
                            }
                        }
                    }
                    _ => {}
                }
            }
            AppState::SigLevelInput => match code {
                KeyCode::Enter => {
                    let alpha: f64 = self.sig_level_input.parse().unwrap_or(0.05);
                    self.compute_sig_level_diagnostics(alpha.clamp(0.001, 0.5));
                    self.active_tab = ResultsTab::Diagnostics;
                    self.scroll_offset = 0;
                    self.state = AppState::Results;
                }
                KeyCode::Esc => { self.state = AppState::Results; }
                KeyCode::Backspace => {
                    if self.sig_level_cursor > 0 {
                        self.sig_level_cursor -= 1;
                        self.sig_level_input.remove(self.sig_level_cursor);
                    }
                }
                KeyCode::Delete => {
                    if self.sig_level_cursor < self.sig_level_input.len() {
                        self.sig_level_input.remove(self.sig_level_cursor);
                    }
                }
                KeyCode::Left => { if self.sig_level_cursor > 0 { self.sig_level_cursor -= 1; } }
                KeyCode::Right => { if self.sig_level_cursor < self.sig_level_input.len() { self.sig_level_cursor += 1; } }
                KeyCode::Home => self.sig_level_cursor = 0,
                KeyCode::End => self.sig_level_cursor = self.sig_level_input.len(),
                KeyCode::Char(c) => {
                    if c.is_ascii_digit() || c == '.' {
                        self.sig_level_input.insert(self.sig_level_cursor, c);
                        self.sig_level_cursor += 1;
                    }
                }
                _ => {}
            },
            AppState::Done => {}
        }
    }

    /// Compute anomaly diagnostics at a user-specified significance level α.
    /// Uses z-score (assuming approximate normality via CLT for large n) from 
    /// statrs' StudentsT for the critical value. Reports values beyond |z| > z_{1-α/2}.
    fn compute_sig_level_diagnostics(&mut self, alpha: f64) {
        let parser = match &self.parser {
            Some(p) => p,
            None => {
                self.diagnostic_lines = vec!["No data loaded.".to_string()];
                return;
            }
        };
        let numeric_df = match &self.numeric_df {
            Some(df) => df.clone(),
            None => {
                self.diagnostic_lines = vec!["No numeric data available.".to_string()];
                return;
            }
        };

        // Critical z-value from Student's t (→ normal for large n)
        let df_for_t = (numeric_df.nrows().max(2) - 1) as f64;
        let t_crit = {
            use statrs::distribution::ContinuousCDF;
            match statrs::distribution::StudentsT::new(0.0, 1.0, df_for_t) {
                Ok(dist) => dist.inverse_cdf(1.0 - alpha / 2.0),
                Err(_) => 1.96,
            }
        };

        let mut diag = Vec::new();
        diag.push(format!("── Anomaly Report at α = {:.4} (|z| > {:.3}) ──", alpha, t_crit));
        diag.push(String::new());
        diag.push(format!("Critical value: t_{{1-α/2}}({:.0}) = {:.3}", df_for_t, t_crit));
        diag.push("Legend: -> = empty cell   !! = severe outlier beyond significance threshold".to_string());
        diag.push(String::new());

        let (nrows, ncols) = numeric_df.shape();
        let imputation = math::imputation::median_imputation(&numeric_df);
        let mut any_anomaly = false;

        // 1. Empty cells
        if imputation.missing_counts.iter().any(|&c| c > 0) {
            any_anomaly = true;
            diag.push("Empty Cells (Median Imputation):".to_string());
            diag.push(format!("{:<6} {:<20} {:>14} {:>10}", "Row", "Column", "Imputed Value", "Status"));
            for row_idx in 0..nrows {
                if imputation.row_has_missing[row_idx] {
                    for col_j in 0..ncols {
                        if imputation.missingness_indicators[(row_idx, col_j)] > 0.5 {
                            let col_name = parser.numeric_columns.get(col_j)
                                .cloned().unwrap_or_else(|| format!("col{}", col_j));
                            diag.push(format!("  {:>4}  {:<20} {:>14.4}   -> imputed",
                                row_idx + 1,
                                crate::display::trunc(&col_name, 19),
                                imputation.imputed_data[(row_idx, col_j)],
                            ));
                        }
                    }
                }
            }
            diag.push(String::new());
        }

        // 2. Z-score outliers
        let mut outlier_count = 0usize;
        diag.push(format!("Outliers (|z| > {:.3}):", t_crit));
        diag.push(format!("{:<6} {:<20} {:>12} {:>12} {:>10} {:>18}", 
            "Row", "Column", "Value", "Mean", "Std Dev", "z-score"));
        diag.push("─".repeat(92).to_string());

        for col_j in 0..ncols {
            let col_name = parser.numeric_columns.get(col_j)
                .cloned().unwrap_or_else(|| format!("col{}", col_j));
            let col_data: Vec<f64> = (0..nrows)
                .map(|i| imputation.imputed_data[(i, col_j)])
                .collect();
            let n = col_data.len() as f64;
            let mean = col_data.iter().sum::<f64>() / n;
            let variance = col_data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            let std_dev = variance.sqrt().max(1e-10);

            for row_idx in 0..nrows {
                let val = imputation.imputed_data[(row_idx, col_j)];
                let z = (val - mean) / std_dev;
                if z.abs() > t_crit {
                    any_anomaly = true;
                    outlier_count += 1;
                    diag.push(format!("  {:>4}  {:<20} {:>12.4} {:>12.4} {:>10.4} {:>15.2}s !! OUTLIER",
                        row_idx + 1,
                        crate::display::trunc(&col_name, 19),
                        val,
                        mean,
                        std_dev,
                        z,
                    ));
                }
            }
        }

        if outlier_count == 0 {
            diag.push(format!("  ✓ No values exceed |z| > {:.3} threshold", t_crit));
        }
        diag.push(String::new());

        // 3. Non-numeric empty cells
        let pr = parser; // reuse parser reference
        for col_idx in 0..pr.headers.len() {
            if pr.columns[col_idx].dtype != parser::ColumnType::Numeric && pr.columns[col_idx].missing_count > 0 {
                any_anomaly = true;
                let col_name = &pr.headers[col_idx];
                diag.push(format!("Empty cells in '{}' ({}):", col_name,
                    match pr.columns[col_idx].dtype {
                        parser::ColumnType::Categorical => "Categorical",
                        parser::ColumnType::DateTime => "DateTime",
                        _ => "String",
                    }));
                for (row_idx, row) in pr.rows.iter().enumerate() {
                    if let Some(cell) = row.get(col_idx) {
                        if cell.is_empty() {
                        diag.push(format!("  Row {:>4}:  \"{}\"  -> (empty string)", row_idx + 1, col_name));
                        }
                    }
                }
                diag.push(String::new());
            }
        }

        if !any_anomaly {
            diag.push(format!("✓ No anomalies detected at α = {:.4} — data is consistent with the model.", alpha));
        }
        self.diagnostic_lines = diag;
    }
}

// ── Rendering ──
fn centered_rect(r: Rect, percent_x: u16, percent_y: u16) -> Rect {
    let popup_width = r.width * percent_x / 100;
    let popup_height = r.height * percent_y / 100;
    let x = r.x + (r.width.saturating_sub(popup_width)) / 2;
    let y = r.y + (r.height.saturating_sub(popup_height)) / 2;
    Rect::new(x, y, popup_width, popup_height)
}

pub fn render(app: &App, frame: &mut Frame) {
    match app.state {
        AppState::Banner => render_banner(app, frame),
        AppState::SetupApiKey => render_setup_api_key(app, frame),
        AppState::SetupModel => render_setup_model(app, frame),
        AppState::FileInput => render_file_input(app, frame),
        AppState::ContextInput => render_context_input(app, frame),
        AppState::Analyzing => render_analyzing(app, frame),
        AppState::Results => render_results(app, frame),
        AppState::PredictionMode => render_prediction_mode(app, frame),
        AppState::SigLevelInput => render_sig_level_input(app, frame),
        AppState::Done => {}
    }
}

fn render_banner(_app: &App, frame: &mut Frame) {
    let area = frame.area();
    let banner = vec![
        Line::from(""),
        Line::from("            ██████╗████████╗ █████╗ ████████╗ ██████╗ ██╗   ██╗██╗██╗     ██╗"),
        Line::from("            ██╔════╝╚══██╔══╝██╔══██╗╚══██╔══╝██╔═══██╗██║   ██║██║██║     ██║"),
        Line::from("            ███████╗   ██║   ███████║   ██║   ██║   ██║██║   ██║██║██║     ██║"),
        Line::from("            ╚════██║   ██║   ██╔══██║   ██║   ██║▄▄ ██║██║   ██║██║██║     ██║"),
        Line::from("            ███████║   ██║   ██║  ██║   ██║   ╚██████╔╝╚██████╔╝██║███████╗███████╗"),
        Line::from("            ╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝    ╚══▀▀═╝  ╚═════╝ ╚═╝╚══════╝╚══════╝"),
        Line::from(""),
        Line::from(Span::styled("            Predictive Analytics CLI v1.0", Style::default().fg(Color::Cyan).bold())),
        Line::from(""),
        Line::from(Span::styled("            [s] Setup Config   |   Any other key to continue", Style::default().fg(Color::DarkGray))),
    ];
    frame.render_widget(Paragraph::new(Text::from(banner)).alignment(Alignment::Center), centered_rect(area, 100, 80));
}

fn input_style() -> Style { Style::default().fg(Color::White) }

fn render_setup_api_key(app: &App, frame: &mut Frame) {
    let area = frame.area(); let popup = centered_rect(area, 80, 40);
    let lines = vec![
        Line::from(Span::styled("Welcome! Let's set up StatQuill.", Style::default().fg(Color::Yellow).bold())),
        Line::from(""),
        Line::from("Get a free API key at: https://openrouter.ai/keys"),
        Line::from(""),
    ];
    frame.render_widget(Paragraph::new(Text::from(lines)).block(
        Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title(" Setup ").title_style(Style::default().fg(Color::Yellow).bold())), popup);
    let input_area = Rect::new(popup.x + 2, popup.y + 7, popup.width.saturating_sub(4), 3);
    let mut ti = input::TextInput::new("OpenRouter API Key", &app.api_key_input, app.api_key_cursor);
    ti.focused = true; ti.input_style = input_style(); ti.cursor_style = Style::default().bg(Color::Cyan).fg(Color::Black);
    frame.render_widget(ti, input_area);
    let hint = Rect::new(popup.x + 2, popup.y + 11, popup.width.saturating_sub(4), 2);
    frame.render_widget(Paragraph::new("Press Enter to confirm | Esc to skip").style(Style::default().fg(Color::DarkGray)), hint);
    if !app.status_message.is_empty() {
        let st = Rect::new(popup.x + 2, popup.y + 14, popup.width.saturating_sub(4), 1);
        frame.render_widget(Paragraph::new(app.status_message.as_str()).style(Style::default().fg(Color::Red)), st);
    }
}

fn render_setup_model(app: &App, frame: &mut Frame) {
    let area = frame.area(); let popup = centered_rect(area, 80, 30);
    frame.render_widget(Paragraph::new("Choose your AI model:").block(
        Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title(" Model ").title_style(Style::default().fg(Color::Yellow).bold())), popup);
    let input_area = Rect::new(popup.x + 2, popup.y + 4, popup.width.saturating_sub(4), 3);
    let mut ti = input::TextInput::new("Model", &app.model_input, app.model_cursor);
    ti.focused = true; ti.input_style = input_style(); ti.cursor_style = Style::default().bg(Color::Cyan).fg(Color::Black);
    frame.render_widget(ti, input_area);
    let hint = Rect::new(popup.x + 2, popup.y + 8, popup.width.saturating_sub(4), 2);
    frame.render_widget(Paragraph::new("Press Enter to confirm | Esc to skip").style(Style::default().fg(Color::DarkGray)), hint);
    if !app.status_message.is_empty() {
        let st = Rect::new(popup.x + 2, popup.y + 11, popup.width.saturating_sub(4), 1);
        frame.render_widget(Paragraph::new(app.status_message.as_str()).style(Style::default().fg(Color::Green)), st);
    }
}

fn render_file_input(app: &App, frame: &mut Frame) {
    let area = frame.area(); let popup = centered_rect(area, 80, 30);
    frame.render_widget(Paragraph::new("Enter the path to your data file (CSV, XLSX, TSV):").block(
        Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title(" Data File ").title_style(Style::default().fg(Color::Magenta).bold())), popup);
    let input_area = Rect::new(popup.x + 2, popup.y + 4, popup.width.saturating_sub(4), 3);
    let mut ti = input::TextInput::new("File Path", &app.file_path_input, app.file_path_cursor);
    ti.focused = true; ti.input_style = input_style(); ti.cursor_style = Style::default().bg(Color::Cyan).fg(Color::Black);
    frame.render_widget(ti, input_area);
    let hint = Rect::new(popup.x + 2, popup.y + 8, popup.width.saturating_sub(4), 2);
    frame.render_widget(Paragraph::new("Press Enter to continue | Esc to quit").style(Style::default().fg(Color::DarkGray)), hint);
    if !app.status_message.is_empty() {
        let st = Rect::new(popup.x + 2, popup.y + 11, popup.width.saturating_sub(4), 1);
        frame.render_widget(Paragraph::new(app.status_message.as_str()).style(Style::default().fg(Color::Red)), st);
    }
}

fn render_sig_level_input(app: &App, frame: &mut Frame) {
    let area = frame.area(); let popup = centered_rect(area, 70, 20);
    frame.render_widget(Paragraph::new("Enter significance level α (e.g. 0.05, 0.01):").block(
        Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title(" Significance Level ").title_style(Style::default().fg(Color::Yellow).bold())), popup);
    let input_area = Rect::new(popup.x + 2, popup.y + 4, popup.width.saturating_sub(4), 3);
    let mut ti = input::TextInput::new("Alpha", &app.sig_level_input, app.sig_level_cursor);
    ti.focused = true; ti.input_style = input_style(); ti.cursor_style = Style::default().bg(Color::Cyan).fg(Color::Black);
    frame.render_widget(ti, input_area);
    let hint = Rect::new(popup.x + 2, popup.y + 8, popup.width.saturating_sub(4), 2);
    frame.render_widget(Paragraph::new("Press Enter to compute anomalies | Esc to cancel").style(Style::default().fg(Color::DarkGray)), hint);
}

fn render_context_input(app: &App, frame: &mut Frame) {
    let area = frame.area(); let popup = centered_rect(area, 80, 30);
    frame.render_widget(Paragraph::new("Provide optional context for AI analysis (or leave blank):").block(
        Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title(" Context ").title_style(Style::default().fg(Color::Magenta).bold())), popup);
    let input_area = Rect::new(popup.x + 2, popup.y + 4, popup.width.saturating_sub(4), 3);
    let mut ti = input::TextInput::new("Context", &app.context_input, app.context_cursor);
    ti.focused = true; ti.input_style = input_style(); ti.cursor_style = Style::default().bg(Color::Cyan).fg(Color::Black);
    frame.render_widget(ti, input_area);
    let hint = Rect::new(popup.x + 2, popup.y + 8, popup.width.saturating_sub(4), 2);
    frame.render_widget(Paragraph::new("Press Enter to start analysis | Esc to go back").style(Style::default().fg(Color::DarkGray)), hint);
}

fn render_analyzing(app: &App, frame: &mut Frame) {
    let area = frame.area();
    let popup = centered_rect(area, 70, 50);
    let mut lines: Vec<Line> = vec![
        Line::from(Span::styled("Analyzing...", Style::default().fg(Color::Green).bold())),
        Line::from(""),
        Line::from(Span::styled("  Processing steps with estimated times:", Style::default().fg(Color::DarkGray))),
        Line::from(""),
    ];
    for msg in &app.analysis_messages {
        lines.push(Line::from(Span::styled(format!("{}", msg), Style::default().fg(Color::Cyan))));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled("  Please wait, this may take a moment...", Style::default().fg(Color::DarkGray))));
    frame.render_widget(Paragraph::new(Text::from(lines)).block(
        Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title(" Processing ").title_style(Style::default().fg(Color::Green).bold())), popup);
}

fn render_results(app: &App, frame: &mut Frame) {
    let area = frame.area();
    let chunks = Layout::default().direction(Direction::Vertical).constraints([
        Constraint::Length(3), Constraint::Min(0), Constraint::Length(2)]).split(area);
    let tab_titles = vec!["1. Overview", "2. Correlations", "3. AI Analysis", "4. Anomalies"];
    let selected = match app.active_tab { ResultsTab::Overview => 0, ResultsTab::Correlations => 1, ResultsTab::AIAnalysis => 2, ResultsTab::Diagnostics => 3 };
    let tabs = Tabs::new(tab_titles).select(selected).block(Block::default().borders(Borders::BOTTOM))
        .style(Style::default().fg(Color::White)).highlight_style(Style::default().fg(Color::Cyan).bold());
    frame.render_widget(tabs, chunks[0]);
    match app.active_tab {
        ResultsTab::Overview => {
            let text: Text = app.data_summary_lines.iter().map(|l| Line::from(l.as_str())).collect();
            let scroll = app.scroll_offset as u16;
            frame.render_widget(Paragraph::new(text).block(Block::default().title(" Data Overview ").borders(Borders::ALL)).wrap(Wrap::default()).scroll((scroll, 0)), chunks[1]);
        }
        ResultsTab::Correlations => {
            let text: Text = app.correlation_lines.iter().map(|l| Line::from(l.as_str())).collect();
            let scroll = app.scroll_offset as u16;
            frame.render_widget(Paragraph::new(text).block(Block::default().title(" Correlations ").borders(Borders::ALL)).wrap(Wrap::default()).scroll((scroll, 0)), chunks[1]);
        }
        ResultsTab::AIAnalysis => {
            // Build combined content: AI commentary + chat history
            let mut all_lines: Vec<Line> = Vec::new();

            // 1. AI commentary (initial analysis)
            let commentary_text = match &app.ai_commentary {
                Some(text) => text.clone(),
                None => {
                    if let (Some(ref parser), Some(ref numeric_df), Some(ref diagnostics), Some(ref selected_model)) =
                        (&app.parser, &app.numeric_df, &app.data_diagnostics, &app.selected_model)
                    {
                        commentary::LocalCommentary::generate(parser, numeric_df, diagnostics, selected_model)
                    } else {
                        "No data available for analysis.".to_string()
                    }
                }
            };
            let md_text = display::render_markdown(&commentary_text);
            let owned_lines: Vec<Line<'static>> = md_text.lines.iter().map(|l| Line::from(l.to_string())).collect();
            all_lines.extend(owned_lines);

            // 2. Chat history
            if !app.chat_history.is_empty() {
                all_lines.push(Line::from(Span::styled("─── Chat Conversation ───", Style::default().fg(Color::Magenta).bold())));
                all_lines.push(Line::from(""));
                for msg in &app.chat_history {
                    if msg.role == "user" {
                        all_lines.push(Line::from(vec![
                            Span::styled("You: ", Style::default().fg(Color::Cyan).bold()),
                            Span::styled(&msg.content, Style::default().fg(Color::White)),
                        ]));
                    } else {
                        all_lines.push(Line::from(vec![
                            Span::styled("AI: ", Style::default().fg(Color::Green).bold()),
                            Span::styled(&msg.content, Style::default().fg(Color::White)),
                        ]));
                    }
                    all_lines.push(Line::from(""));
                }
            }

            // 3. Chat status (e.g., "Waiting for response...")
            if !app.chat_status.is_empty() {
                if !app.chat_history.is_empty() {
                    all_lines.push(Line::from(""));
                }
                all_lines.push(Line::from(Span::styled(&app.chat_status, Style::default().fg(Color::Yellow))));
            }

            let content_text = Text::from(all_lines);
            let scroll = app.scroll_offset as u16;
            frame.render_widget(
                Paragraph::new(content_text)
                    .block(Block::default().title(" 🤖 AI Analysis ").borders(Borders::ALL).border_type(BorderType::Rounded))
                    .wrap(Wrap::default())
                    .scroll((scroll, 0)),
                chunks[1],
            );
        }
        ResultsTab::Diagnostics => {
            let text: Text = app.diagnostic_lines.iter().map(|l| Line::from(l.as_str())).collect();
            let scroll = app.scroll_offset as u16;
            frame.render_widget(Paragraph::new(text).block(Block::default().title(" Anomaly Diagnostics ").borders(Borders::ALL)).wrap(Wrap::default()).scroll((scroll, 0)), chunks[1]);
        }
    }
    frame.render_widget(Paragraph::new(" [1|2|3|4] Tabs | [a] Anomalies at α | [s] Setup | [p] Predict | ↑↓ scroll | [q] Quit ").style(Style::default().fg(Color::DarkGray)).alignment(Alignment::Center), chunks[2]);
}

fn render_prediction_mode(app: &App, frame: &mut Frame) {
    let area = frame.area();
    let chunks = Layout::default().direction(Direction::Vertical).constraints([
        Constraint::Length(1), Constraint::Min(0), Constraint::Length(2)]).split(area);
    frame.render_widget(Paragraph::new("Prediction Mode - Enter known values, leave unknowns blank")
        .style(Style::default().fg(Color::Cyan).bold()).alignment(Alignment::Center), chunks[0]);
    let main = Layout::default().direction(Direction::Horizontal).constraints([Constraint::Percentage(50), Constraint::Percentage(50)]).split(chunks[1]);
    let cols: Vec<String> = if let Some(ref parser) = app.parser { parser.numeric_columns.clone() } else { vec![] };
    let input_area = Rect::new(main[0].x, main[0].y, main[0].width, (cols.len() as u16 * 3).max(10));
    let mut input_lines: Vec<Line> = Vec::new();
    for (i, col) in cols.iter().enumerate() {
        let value = app.prediction_inputs.get(col).map(|s| s.as_str()).unwrap_or("");
        let prefix = if i == app.prediction_focus { "▶ " } else { "  " };
        let is_focus = i == app.prediction_focus;
        input_lines.push(Line::from(vec![
            Span::styled(prefix, Style::default().fg(Color::Cyan)),
            Span::styled(format!("{:<20}", col), if is_focus { Style::default().fg(Color::Yellow).bold() } else { Style::default().fg(Color::DarkGray) }),
            Span::styled(if value.is_empty() { "(empty)".to_string() } else { value.to_string() }, Style::default().fg(Color::Cyan)),
        ]));
    }
    frame.render_widget(Paragraph::new(Text::from(input_lines)).block(Block::default().borders(Borders::ALL).title(" Input Values ").title_style(Style::default().fg(Color::Magenta).bold())), input_area);
    let mut result_lines: Vec<Line> = if !app.prediction_message.is_empty() {
        vec![Line::from(Span::styled(&app.prediction_message, Style::default().fg(Color::Green).bold())), Line::from("")]
    } else { vec![] };
    if !app.prediction_results.is_empty() {
        result_lines.push(Line::from(Span::styled(format!("{:<20} {:>12} {:>12} {:>12} {:>8} {:>10}", "Target", "Prediction", "95% Lower", "95% Upper", "CV", "Confidence"), Style::default().fg(Color::Magenta).bold())));
        for p in &app.prediction_results {
            let (conf, color) = if p.cv < 0.1 { ("High", Color::Green) } else if p.cv < 0.3 { ("Medium", Color::Yellow) } else { ("Low", Color::Red) };
            result_lines.push(Line::from(vec![
                Span::styled(format!("{:<20}", display::trunc(&p.target, 19)), Style::default().fg(Color::White)),
                Span::styled(format!(" {:>11.4}", p.value), Style::default().fg(Color::Cyan)),
                Span::styled(format!(" {:>11.4}", p.lower), Style::default().fg(Color::DarkGray)),
                Span::styled(format!(" {:>11.4}", p.upper), Style::default().fg(Color::DarkGray)),
                Span::styled(format!(" {:>7.4}", p.cv), Style::default().fg(Color::White)),
                Span::styled(format!(" {:>10}", conf), Style::default().fg(color).bold()),
            ]));
        }
    }
    frame.render_widget(Paragraph::new(Text::from(result_lines)).block(Block::default().borders(Borders::ALL).title(" Predictions ").title_style(Style::default().fg(Color::Magenta).bold())), main[1]);
    frame.render_widget(Paragraph::new(" ↑↓ Navigate | Type value | Enter to predict | Esc back to results ").style(Style::default().fg(Color::DarkGray)).alignment(Alignment::Center), chunks[2]);
}

// ── Clipboard helpers ──
fn insert_at_cursor(buf: &mut String, cursor: &mut usize, text: &str) {
    for ch in text.chars() { buf.insert(*cursor, ch); *cursor += 1; }
}

fn handle_paste(app: &mut App, text: &str) {
    match app.state {
        AppState::SetupApiKey => { insert_at_cursor(&mut app.api_key_input, &mut app.api_key_cursor, text); }
        AppState::SetupModel => { insert_at_cursor(&mut app.model_input, &mut app.model_cursor, text); }
        AppState::FileInput => { insert_at_cursor(&mut app.file_path_input, &mut app.file_path_cursor, text); }
        AppState::ContextInput => { insert_at_cursor(&mut app.context_input, &mut app.context_cursor, text); }
        AppState::PredictionMode => {
            if let Some(ref parser) = app.parser {
                if let Some(col) = parser.numeric_columns.get(app.prediction_focus) {
                    if let Some(s) = app.prediction_inputs.get_mut(col) {
                        for ch in text.chars() {
                            if ch.is_ascii_digit() || ch == '.' || ch == '-' || ch == '+' || ch == 'e' || ch == 'E' { s.push(ch); }
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

fn get_clipboard_text() -> Option<String> {
    use std::process::Command;
    let output = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", "Get-Clipboard"])
        .output().ok()?;
    if output.status.success() {
        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !text.is_empty() { return Some(text); }
    }
    None
}

// ── TUI Runner ──
pub fn run_app(mut app: App) -> Result<()> {
    execute!(std::io::stdout(), crossterm::event::EnableBracketedPaste)?;
    let mut terminal = ratatui::init();
    terminal.clear()?;

    while !app.should_exit {
        terminal.draw(|frame| render(&app, frame))?;

        // If we just entered Analyzing state, draw the frame first so the user sees the
        // "Processing" screen, THEN run the synchronous analysis.
        if matches!(app.state, AppState::Analyzing) {
            terminal.draw(|frame| render(&app, frame))?;
            app.run_analysis();
            terminal.draw(|frame| render(&app, frame))?;
        }
        if matches!(app.state, AppState::Done) { break; }

        match event::read()? {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                if key.code == KeyCode::Char('v') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    if let Some(text) = get_clipboard_text() {
                        handle_paste(&mut app, &text);
                    }
                } else {
                    app.handle_key(key.code);
                }
            }
            Event::Paste(text) => { handle_paste(&mut app, &text); }
            _ => {}
        }
    }

    ratatui::restore();
    execute!(std::io::stdout(), crossterm::event::DisableBracketedPaste)?;
    Ok(())
}
