use anyhow::{Context, Result};
use nalgebra::DMatrix;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub enum ColumnType {
    Numeric,
    DateTime,
    Categorical,
    String,
}

#[derive(Debug, Clone)]
pub struct ColumnMeta {
    pub name: String,
    pub dtype: ColumnType,
    pub missing_count: usize,
    pub unique_count: usize,
    pub mean: Option<f64>,
    pub std: Option<f64>,
    pub min_val: Option<f64>,
    pub max_val: Option<f64>,
}

pub struct DataParser {
    pub filepath: String,
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub columns: Vec<ColumnMeta>,
    pub time_column: Option<String>,
    pub numeric_columns: Vec<String>,
    /// All numeric data: rows × columns (only numeric cols), stored as f64
    numeric_data: Vec<Vec<f64>>,
}

const TIME_HINTS: &[&str] = &[
    "date", "time", "timestamp", "datetime", "year", "month", "day",
];

impl DataParser {
    pub fn load(filepath: &str) -> Result<Self> {
        let path = Path::new(filepath);
        if !path.exists() {
            return Err(anyhow::anyhow!("File not found: {}", filepath));
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();

        match ext.as_str() {
            "csv" => Self::load_csv(filepath),
            "tsv" | "txt" => Self::load_tsv(filepath),
            "xlsx" | "xls" => Self::load_excel(filepath),
            other => Err(anyhow::anyhow!(
                "Unsupported format: .{} (supported: .csv, .tsv, .txt, .xlsx, .xls)",
                other
            )),
        }
    }

    fn load_csv(filepath: &str) -> Result<Self> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_path(filepath)
            .context("Failed to open CSV file")?;

        let headers: Vec<String> = reader
            .headers()
            .context("Failed to read CSV headers")?
            .iter()
            .map(|h| h.trim().to_string())
            .collect();

        let rows: Vec<Vec<String>> = reader
            .records()
            .filter_map(|r| r.ok())
            .map(|r| r.iter().map(|f| f.trim().to_string()).collect())
            .collect();

        Self::from_raw(headers, rows).context("Failed to parse CSV data")
    }

    fn load_tsv(filepath: &str) -> Result<Self> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .delimiter(b'\t')
            .from_path(filepath)
            .context("Failed to open TSV file")?;

        let headers: Vec<String> = reader
            .headers()
            .context("Failed to read TSV headers")?
            .iter()
            .map(|h| h.trim().to_string())
            .collect();

        let rows: Vec<Vec<String>> = reader
            .records()
            .filter_map(|r| r.ok())
            .map(|r| r.iter().map(|f| f.trim().to_string()).collect())
            .collect();

        Self::from_raw(headers, rows).context("Failed to parse TSV data")
    }

    fn load_excel(filepath: &str) -> Result<Self> {
        use calamine::{open_workbook, Reader, Xlsx};

        let mut workbook: Xlsx<_> = open_workbook(filepath)
            .context("Failed to open Excel file")?;

        let sheet_name = workbook
            .sheet_names()
            .first()
            .cloned()
            .context("Excel file has no sheets")?;

        let range = workbook
            .worksheet_range(&sheet_name)
            .context("Failed to read Excel sheet")?;

        let mut rows_iter = range.rows();
        let first_row = rows_iter
            .next()
            .context("Excel sheet is empty")?;

        let headers: Vec<String> = first_row
            .iter()
            .map(|c| c.to_string().trim().to_string())
            .collect();

        let n_cols = headers.len();
        let rows: Vec<Vec<String>> = rows_iter
            .map(|row| {
                let mut r: Vec<String> = row.iter().map(|c| c.to_string().trim().to_string()).collect();
                while r.len() < n_cols {
                    r.push(String::new());
                }
                r.truncate(n_cols);
                r
            })
            .collect();

        Self::from_raw(headers, rows).context("Failed to parse Excel data")
    }

    fn from_raw(raw_headers: Vec<String>, rows: Vec<Vec<String>>) -> Result<Self> {
        let filepath = "unknown".to_string();
        // Normalize headers: strip whitespace, handle duplicates
        let headers = Self::normalize_headers(&raw_headers);
        let n_cols = headers.len();

        // Pre-process: detect which columns are datetime, numeric, etc.
        let mut col_types: Vec<ColumnType> = vec![ColumnType::String; n_cols];
        let mut numeric_values: Vec<Vec<Option<f64>>> = vec![vec![None; rows.len()]; n_cols];

        // First pass: try parsing
        for col_idx in 0..n_cols {
            let col_name_lower = headers[col_idx].to_lowercase();

            // Check if this column hints at being a time column
            let is_time_hint = TIME_HINTS.iter().any(|h| col_name_lower.contains(h));

            let mut try_datetime = is_time_hint;
            let mut try_numeric = true;

            let mut datetime_count = 0usize;
            let mut numeric_count = 0usize;
            let mut valid_count = 0usize;

            for row_idx in 0..rows.len() {
                let val = &rows[row_idx].get(col_idx).map(|s| s.as_str()).unwrap_or("");
                if val.is_empty() {
                    continue;
                }
                valid_count += 1;

                if try_datetime {
                    if Self::parse_datetime(val).is_some() {
                        datetime_count += 1;
                    } else {
                        try_datetime = false;
                    }
                }

                if try_numeric {
                    if let Ok(n) = val.parse::<f64>() {
                        numeric_values[col_idx][row_idx] = Some(n);
                        numeric_count += 1;
                    } else if val.parse::<i64>().is_ok() {
                        // Re-parse as f64
                        if let Ok(n) = val.parse::<f64>() {
                            numeric_values[col_idx][row_idx] = Some(n);
                            numeric_count += 1;
                        } else {
                            try_numeric = false;
                        }
                    } else {
                        try_numeric = false;
                    }
                }
            }

            if valid_count == 0 {
                col_types[col_idx] = ColumnType::String;
            } else if try_datetime && datetime_count > 0 {
                if datetime_count as f64 / valid_count as f64 > 0.8 {
                    col_types[col_idx] = ColumnType::DateTime;
                    continue;
                }
            }

            if try_numeric && numeric_count > 0 {
                if numeric_count as f64 / valid_count as f64 > 0.8 {
                    col_types[col_idx] = ColumnType::Numeric;
                    continue;
                }
            }

            // Check categorical
            let unique: HashSet<&str> = rows
                .iter()
                .filter_map(|r| r.get(col_idx))
                .map(|s| s.as_str())
                .filter(|s| !s.is_empty())
                .collect();
            if !unique.is_empty() && unique.len() < 50 && (unique.len() as f64) / (valid_count as f64) < 0.1 {
                col_types[col_idx] = ColumnType::Categorical;
            } else {
                col_types[col_idx] = ColumnType::String;
            }
        }

        // Identify time column
        let mut time_column: Option<String> = None;
        for col_idx in 0..n_cols {
            if col_types[col_idx] == ColumnType::DateTime {
                time_column = Some(headers[col_idx].clone());
                break;
            }
        }
        if time_column.is_none() {
            // Fallback: check first DateTime hint by name
            for col_idx in 0..n_cols {
                let col_name_lower = headers[col_idx].to_lowercase();
                if TIME_HINTS.iter().any(|h| col_name_lower.contains(h)) {
                    time_column = Some(headers[col_idx].clone());
                    col_types[col_idx] = ColumnType::DateTime;
                    break;
                }
            }
        }

        // Build numeric data matrix (drop rows with any NaN in numeric columns)
        let numeric_column_names: Vec<String> = headers
            .iter()
            .enumerate()
            .filter(|(i, _)| col_types[*i] == ColumnType::Numeric)
            .map(|(_, h)| h.clone())
            .collect();

        // Build a list of numeric column indices
        let num_indices: Vec<usize> = (0..n_cols)
            .filter(|i| col_types[*i] == ColumnType::Numeric)
            .collect();

        let mut numeric_data: Vec<Vec<f64>> = Vec::new();
        for row_idx in 0..rows.len() {
            let all_numeric = num_indices.iter().all(|&ci| numeric_values[ci][row_idx].is_some());
            if all_numeric && !num_indices.is_empty() {
                let row: Vec<f64> = num_indices
                    .iter()
                    .map(|&ci| numeric_values[ci][row_idx].unwrap())
                    .collect();
                numeric_data.push(row);
            }
        }

        // Build ColumnMeta
        let columns: Vec<ColumnMeta> = (0..n_cols)
            .map(|col_idx| {
                let dtype = col_types[col_idx].clone();
                let missing_count = rows
                    .iter()
                    .filter(|r| r.get(col_idx).map(|s| s.is_empty()).unwrap_or(true))
                    .count();
                let unique_values: HashSet<&str> = rows
                    .iter()
                    .filter_map(|r| r.get(col_idx))
                    .map(|s| s.as_str())
                    .filter(|s| !s.is_empty())
                    .collect();

                let (mean, std, min_val, max_val) = if dtype == ColumnType::Numeric
                    && !numeric_column_names.contains(&"".to_string())
                {
                    let vals: Vec<f64> = rows
                        .iter()
                        .enumerate()
                        .filter_map(|(ri, _r)| numeric_values[col_idx][ri])
                        .collect();
                    if !vals.is_empty() {
                        let n = vals.len() as f64;
                        let mean = vals.iter().sum::<f64>() / n;
                        let variance =
                            vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
                        let std = variance.sqrt();
                        let min_val = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                        let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        (Some(mean), Some(std), Some(min_val), Some(max_val))
                    } else {
                        (None, None, None, None)
                    }
                } else {
                    (None, None, None, None)
                };

                ColumnMeta {
                    name: headers[col_idx].clone(),
                    dtype,
                    missing_count,
                    unique_count: unique_values.len(),
                    mean,
                    std,
                    min_val,
                    max_val,
                }
            })
            .collect();

        Ok(DataParser {
            filepath,
            headers,
            rows,
            columns,
            time_column,
            numeric_columns: numeric_column_names,
            numeric_data,
        })
    }

    fn normalize_headers(raw: &[String]) -> Vec<String> {
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut result = Vec::new();

        for name in raw {
            let trimmed = name.trim().to_string();
            if let Some(count) = seen.get_mut(&trimmed) {
                *count += 1;
                result.push(format!("{}_{}", trimmed, count));
            } else {
                seen.insert(trimmed.clone(), 0);
                result.push(trimmed);
            }
        }
        result
    }

    fn parse_datetime(s: &str) -> Option<chrono::NaiveDateTime> {
        // Try various common date/time formats
        let formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d",
            "%d-%m-%Y %H:%M:%S",
            "%d-%m-%Y",
            "%Y%m%d",
            "%B %d, %Y",
            "%d %B %Y",
        ];

        for fmt in &formats {
            if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
                return Some(dt);
            }
            if let Ok(d) = chrono::NaiveDate::parse_from_str(s, fmt) {
                return Some(d.and_hms_opt(0, 0, 0).unwrap());
            }
        }

        // Try ISO 8601
        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
            return Some(dt.naive_utc());
        }
        if let Ok(dt) = chrono::DateTime::parse_from_rfc2822(s) {
            return Some(dt.naive_utc());
        }

        None
    }

    pub fn has_time_data(&self) -> bool {
        self.time_column.is_some()
    }

    pub fn get_numeric_matrix(&self) -> DMatrix<f64> {
        if self.numeric_data.is_empty() {
            return DMatrix::from_element(0, 0, 0.0);
        }
        let nrows = self.numeric_data.len();
        let ncols = self.numeric_data.first().map(|r| r.len()).unwrap_or(0);
        let data: Vec<f64> = self.numeric_data.iter().flatten().copied().collect();
        DMatrix::from_vec(nrows, ncols, data)
    }

    pub fn numeric_data_values(&self) -> &Vec<Vec<f64>> {
        &self.numeric_data
    }

    /// Build a JSON payload for the AI commentary
    pub fn build_stats_payload(&self, _numeric_df: &DMatrix<f64>) -> Value {
        let numeric_summary: Value = self
            .columns
            .iter()
            .filter(|c| c.dtype == ColumnType::Numeric)
            .map(|c| {
                (
                    c.name.clone(),
                    json!({
                        "mean": c.mean,
                        "std": c.std,
                        "min": c.min_val,
                        "max": c.max_val,
                    }),
                )
            })
            .collect::<serde_json::Map<_, _>>()
            .into();

        json!({
            "file": self.filepath,
            "summary": {
                "rows": self.rows.len(),
                "columns": self.headers.len(),
                "time_column": self.time_column,
                "numeric_columns": self.numeric_columns,
                "column_meta": self.columns.iter().map(|c| json!({
                    "name": c.name,
                    "type": format!("{:?}", c.dtype),
                    "missing_count": c.missing_count,
                    "unique_count": c.unique_count,
                })).collect::<Vec<_>>(),
            },
            "numeric_summary": numeric_summary,
        })
    }
}
