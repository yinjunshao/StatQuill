use crate::math::autoregressive::ARResult;
use crate::math::linear_regression::RegressionResult;
use crate::parser::{ColumnType, DataParser};
use anyhow::Result;
use nalgebra::DMatrix;
use ratatui::{
    layout::Constraint,
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, Cell, Paragraph, Row, Table, Widget, Wrap},
};
use std::io::{self, Write};

#[derive(Debug, Clone)]
pub struct PredictionDisplay {
    pub target: String,
    pub value: f64,
    pub lower: f64,
    pub upper: f64,
    pub cv: f64,
}

pub struct DisplayEngine;

impl DisplayEngine {
    pub fn new() -> Self { Self }

    fn c(&self, color: Color) -> Style { Style::default().fg(color) }
    fn bld(&self) -> Style { Style::default().add_modifier(Modifier::BOLD) }
    fn bc(&self, color: Color) -> Style { Style::default().fg(color).add_modifier(Modifier::BOLD) }

    // ── banner ──
    pub fn show_banner(&self) {
        let lines = vec![
            Line::from("            ██████╗████████╗ █████╗ ████████╗ ██████╗ ██╗   ██╗██╗██╗     ██╗"),
            Line::from("            ██╔════╝╚══██╔══╝██╔══██╗╚══██╔══╝██╔═══██╗██║   ██║██║██║     ██║"),
            Line::from("            ███████╗   ██║   ███████║   ██║   ██║   ██║██║   ██║██║██║     ██║"),
            Line::from("            ╚════██║   ██║   ██╔══██║   ██║   ██║▄▄ ██║██║   ██║██║██║     ██║"),
            Line::from("            ███████║   ██║   ██║  ██║   ██║   ╚██████╔╝╚██████╔╝██║███████╗███████╗"),
            Line::from("            ╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝    ╚══▀▀═╝  ╚═════╝ ╚═╝╚══════╝╚══════╝"),
            Line::from(""),
            Line::from(Span::styled("            Predictive Analytics CLI v1.0", self.bc(Color::Cyan))),
        ];
        println!("{}", render_widget(Paragraph::new(Text::from(lines))));
    }

    fn section(&self, title: &str) {
        println!();
        println!("{}", Span::styled(title, self.bc(Color::Magenta)));
        println!("{}", Span::styled("─".repeat(title.len().min(60)), self.c(Color::Magenta)));
    }

    fn line(&self, key: &str, value: String) -> Line<'static> {
        Line::from(vec![
            Span::styled(format!("  {:<18}", key), self.c(Color::Cyan)),
            Span::from(value),
        ])
    }

    fn tbl<'a>(&self, title: &'a str, hdr: Row<'a>, rows: Vec<Row<'a>>, wz: &'a [Constraint]) -> String {
        render_widget(
            Table::new(rows, wz)
                .header(hdr.bold().fg(Color::Magenta))
                .block(
                    Block::default()
                        .title(format!(" {} ", title))
                        .title_style(self.bc(Color::Magenta))
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .border_style(self.c(Color::DarkGray)),
                )
                .column_spacing(2),
        )
    }

    // ── dataset overview ──
    pub fn show_data_summary(&self, parser: &DataParser) {
        let num_cols_str = if parser.numeric_columns.is_empty() {
            "None".to_string()
        } else {
            parser.numeric_columns.join(", ")
        };
        self.section("Dataset Overview");
        let lines = vec![
            self.line("File", parser.filepath.clone()),
            self.line("Rows", parser.rows.len().to_string()),
            self.line("Columns", parser.headers.len().to_string()),
            self.line("Time Column", parser.time_column.as_deref().unwrap_or("None detected").to_string()),
            self.line("Numeric Cols", num_cols_str),
        ];
        println!("{}", render_widget(Paragraph::new(Text::from(lines))));

        let hdr = Row::new(vec!["Column", "Type", "Missing", "Mean / Unique"]);
        let rows: Vec<Row> = parser.columns.iter().map(|meta| {
            let dt = format!("{:?}", meta.dtype);
            let mv = if meta.dtype == ColumnType::Numeric {
                meta.mean.map_or("N/A".to_string(), |m| format!("{:.4}", m))
            } else { format!("{} unique", meta.unique_count) };
            Row::new(vec![
                Cell::from(trunc(&meta.name, 19)),
                Cell::from(dt).fg(tc(&meta.dtype)),
                Cell::from(meta.missing_count.to_string()),
                Cell::from(mv),
            ])
        }).collect();
        println!("{}", self.tbl("Column Analysis", hdr, rows, &[Constraint::Length(20), Constraint::Length(14), Constraint::Length(10), Constraint::Min(15)]));
    }

    // ── correlations ──
    pub fn show_covariance(&self, cov: &DMatrix<f64>, cols: &[String], _corr: &DMatrix<f64>) {
        let n = cols.len();
        if n == 0 { return; }
        let sd: Vec<f64> = (0..n).map(|j| cov[(j, j)].sqrt()).collect();
        let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n { for j in i+1..n {
            let d = sd[i] * sd[j];
            let r = if d > 1e-15 { cov[(i, j)] / d } else { 0.0 };
            if r.abs() > 0.3 { pairs.push((i, j, r)); }
        }}
        pairs.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());

        if pairs.is_empty() {
            self.section("Correlations");
            println!("  No strong correlations found.");
            return;
        }

        let hdr = Row::new(vec!["Variable A", "Variable B", "Correlation"]);
        let rows: Vec<Row> = pairs.iter().take(15).map(|(i, j, r)| {
            let cl = if r.abs() > 0.9 { Color::Red } else if r.abs() > 0.7 { Color::Yellow } else { Color::Green };
            let a = cols.get(*i).cloned().unwrap_or(format!("col{}", i));
            let b = cols.get(*j).cloned().unwrap_or(format!("col{}", j));
            Row::new(vec![Cell::from(trunc(&a, 19)), Cell::from(trunc(&b, 19)), Cell::from(format!("{:.4}", r)).fg(cl)])
        }).collect();
        println!("{}", self.tbl("Top Correlations", hdr, rows, &[Constraint::Length(20), Constraint::Length(20), Constraint::Length(12)]));
    }

    // ── AI commentary using ratatui Markdown rendering (table-friendly) ──
    pub fn show_ai_commentary(&self, text: &str) {
        println!();
        // Render the entire markdown text as one Paragraph with Wrap
        let block = Block::default()
            .title(" 🤖 AI Analysis ")
            .title_style(self.bc(Color::Green))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(self.c(Color::DarkGray));

        // Parse markdown-like syntax into styled Text
        let md_text = parse_markdown_to_text(text, self);

        let paragraph = Paragraph::new(md_text)
            .block(block)
            .wrap(Wrap::default());

        println!("{}", render_widget(paragraph));
    }

    // ── model results ──
    pub fn show_model_results(&self, r: &RegressionResult) {
        let ms = se_avg(&r.standard_error);
        let mc = cv_avg(&r.cv);
        self.section("Model Results");
        let lines = vec![
            self.line("Target", r.target_column.clone()),
            self.line("RSS", format!("{:.4}", r.rss)),
            self.line("σ² (Error Var)", format!("{:.6}", r.sigma2)),
            self.line("Mean SE", format!("{:.4}", ms)),
            self.line("Mean CV", format!("{:.4}", mc)),
        ];
        println!("{}", render_widget(Paragraph::new(Text::from(lines))));
        if !r.coefficients.is_empty() {
            let hdr = Row::new(vec!["Feature", "Weight (β)"]);
            let mut rows: Vec<Row> = r.feature_columns.iter().zip(r.coefficients.iter())
                .map(|(f, c)| Row::new(vec![Cell::from(trunc(f, 19)), Cell::from(format!("{:.6}", c))]))
                .collect();
            rows.push(Row::new(vec![Cell::from("(intercept)"), Cell::from(format!("{:.6}", r.intercept))]));
            println!("{}", self.tbl("Coefficients", hdr, rows, &[Constraint::Length(20), Constraint::Length(16)]));
        }
    }

    // ── AR results ──
    pub fn show_ar_results(&self, r: &ARResult, p: usize, phi: &[f64], c: f64) {
        let ms = se_avg(&r.standard_error);
        let mc = cv_avg(&r.cv);
        self.section(&format!("AR Model (order p={})", p));
        let lines = vec![
            self.line("RSS", format!("{:.4}", r.rss)),
            self.line("σ²", format!("{:.6}", r.sigma2)),
            self.line("Mean SE", format!("{:.4}", ms)),
            self.line("Mean CV", format!("{:.4}", mc)),
        ];
        println!("{}", render_widget(Paragraph::new(Text::from(lines))));
        let hdr = Row::new(vec!["Coefficient", "Value"]);
        let mut rows = vec![Row::new(vec![Cell::from("c (intercept)"), Cell::from(format!("{:.6}", c))])];
        for (i, coef) in phi.iter().enumerate() {
            rows.push(Row::new(vec![Cell::from(format!("φ_{}", i + 1)), Cell::from(format!("{:.6}", coef))]));
        }
        println!("{}", self.tbl("AR Coefficients", hdr, rows, &[Constraint::Length(20), Constraint::Length(16)]));
    }

    // ── predictions ──
    pub fn show_predictions(&self, preds: &[PredictionDisplay]) {
        let hdr = Row::new(vec!["Target", "Prediction", "95% Lower", "95% Upper", "CV", "Confidence"]);
        let mut s = preds.to_vec();
        s.sort_by(|a, b| a.cv.partial_cmp(&b.cv).unwrap_or(std::cmp::Ordering::Equal));
        let rows: Vec<Row> = s.iter().map(|p| {
            let (conf, cl) = if p.cv < 0.1 { ("High", Color::Green) } else if p.cv < 0.3 { ("Medium", Color::Yellow) } else { ("Low", Color::Red) };
            Row::new(vec![
                Cell::from(trunc(&p.target, 19)), Cell::from(format!("{:.4}", p.value)),
                Cell::from(format!("{:.4}", p.lower)), Cell::from(format!("{:.4}", p.upper)),
                Cell::from(format!("{:.4}", p.cv)), Cell::from(conf).fg(cl),
            ])
        }).collect();
        println!("{}", self.tbl("Predictions (sorted by lowest CV)", hdr, rows, &[
            Constraint::Length(20), Constraint::Length(14), Constraint::Length(14), Constraint::Length(14), Constraint::Length(10), Constraint::Length(12),
        ]));
    }

    pub fn show_future_predictions(&self, target: &str, pred: &[f64], lo: &[f64], hi: &[f64]) {
        let hdr = Row::new(vec!["Step", "Prediction", "95% Lower", "95% Upper"]);
        let rows: Vec<Row> = pred.iter().zip(lo.iter()).zip(hi.iter()).enumerate()
            .map(|(i, ((p, l), u))| Row::new(vec![
                Cell::from(format!("+{}", i + 1)), Cell::from(format!("{:.4}", p)),
                Cell::from(format!("{:.4}", l)), Cell::from(format!("{:.4}", u)),
            ])).collect();
        let title = format!("Future Predictions for '{}'", target);
        println!("{}", self.tbl(&title, hdr, rows, &[Constraint::Length(8), Constraint::Length(14), Constraint::Length(14), Constraint::Length(14)]));
    }

    // ── prompts ──
    pub fn prompt_file_path(&self) -> Result<String> { self.rd("Enter path to data file") }
    pub fn prompt_context(&self) -> Result<String> { self.rd("Optional context/description") }
    pub fn prompt_input(&self, p: &str) -> Result<String> { self.rd(p) }
    pub fn confirm(&self, prompt: &str, default: bool) -> bool {
        let yn = if default { "[Y/n]" } else { "[y/N]" };
        print!("{} {} ", prompt, yn); io::stdout().flush().ok();
        let mut s = String::new(); io::stdin().read_line(&mut s).ok();
        let s = s.trim().to_lowercase();
        if s.is_empty() { default } else { s == "y" || s == "yes" }
    }
    fn rd(&self, p: &str) -> Result<String> {
        print!("{}: ", p); io::stdout().flush()?;
        let mut s = String::new(); io::stdin().read_line(&mut s)?;
        Ok(s.trim().to_string())
    }

    // ── status ──
    pub fn print_status(&self, msg: &str) { println!("{}", Span::styled(format!("● {}", msg), self.bc(Color::Green))); }
    pub fn print_warning(&self, msg: &str) { println!("{}", Span::styled(format!("⚠  {}", msg), self.bc(Color::Yellow))); }
    pub fn print_info(&self, msg: &str) { println!("{}", Span::styled(format!("ℹ  {}", msg), self.bc(Color::Cyan))); }
    pub fn print_error_header(&self, msg: &str) {
        let bl = Block::default().borders(Borders::ALL).border_style(self.c(Color::Red));
        println!("{}", render_widget(Paragraph::new(msg.to_string()).block(bl).fg(Color::Red)));
    }
    pub fn print_text(&self, msg: &str) { println!("{}", msg); }
    pub fn print_separator(&self) {
        println!();
        println!("{}", Span::styled("═".repeat(60), self.bc(Color::Green)));
    }
    pub fn wait_for_exit(&self) {
        println!();
        println!("Press Enter to exit...");
        io::stdout().flush().ok();
        let mut buf = String::new();
        io::stdin().read_line(&mut buf).ok();
    }
}

// ── parse markdown → ratatui Text ──
// Handles headings, bold/italic, code blocks, lists, numbered lists, tables, and inline code
fn parse_markdown_to_text(text: &str, d: &DisplayEngine) -> Text<'static> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut in_code_block = false;

    for raw in text.lines() {
        let owned = raw.to_string();

        if owned.trim().starts_with("```") {
            in_code_block = !in_code_block;
            if in_code_block {
                lines.push(Line::from(Span::styled("  ```", d.c(Color::DarkGray))));
            } else {
                lines.push(Line::from(Span::styled("  ```", d.c(Color::DarkGray))));
            }
            continue;
        }

        if in_code_block {
            lines.push(Line::from(Span::styled(format!("  {}", owned), d.c(Color::DarkGray))));
            continue;
        }

        if owned.trim().is_empty() {
            lines.push(Line::from(""));
            continue;
        }

        // Tables: detect if the line contains pipe separators and looks like a table
        if owned.trim().starts_with('|') && owned.trim().ends_with('|') {
            lines.push(parse_table_row(&owned, d));
            continue;
        }
        // Separator line like |---|---|
        if owned.trim().starts_with('|') && owned.contains("---") {
            lines.push(Line::from(Span::styled(owned, d.c(Color::DarkGray))));
            continue;
        }

        // Headings
        if let Some(c) = owned.trim().strip_prefix("### ") {
            lines.push(Line::from(Span::styled(format!("  ▸ {}", c.trim()), d.bc(Color::Cyan))));
            continue;
        }
        if let Some(c) = owned.trim().strip_prefix("## ") {
            lines.push(Line::from(Span::styled(format!("  ▸ {}", c.trim()), d.bc(Color::Cyan))));
            continue;
        }
        if let Some(c) = owned.trim().strip_prefix("# ") {
            lines.push(Line::from(Span::styled(format!("  ▸ {}", c.trim()), d.bc(Color::Yellow))));
            continue;
        }

        // Unordered lists
        if owned.trim().starts_with("- ") || owned.trim().starts_with("* ") {
            let content = owned.trim()[2..].to_string();
            let styled = parse_inline_styles(&content, d);
            let mut spans = vec![Span::styled("  • ", d.c(Color::DarkGray))];
            spans.extend(styled);
            lines.push(Line::from(spans));
            continue;
        }

        // Numbered lists
        if let Some((num, rest)) = parse_numbered(&owned.trim()) {
            let styled = parse_inline_styles(rest, d);
            let mut spans = vec![Span::styled(format!("  {}. ", num), d.c(Color::DarkGray))];
            spans.extend(styled);
            lines.push(Line::from(spans));
            continue;
        }

        // Regular paragraph with inline styling
        let styled = parse_inline_styles(&owned, d);
        lines.push(Line::from(styled));
    }

    Text::from(lines)
}

/// Parse a markdown table row like `| A | B | C |` into a styled Line
fn parse_table_row(row: &str, d: &DisplayEngine) -> Line<'static> {
    let trimmed = row.trim();
    let inner = trimmed.trim_start_matches('|').trim_end_matches('|');
    let cells: Vec<&str> = inner.split('|').map(|s| s.trim()).collect();

    let mut spans: Vec<Span<'static>> = Vec::new();
    spans.push(Span::styled("│ ", d.c(Color::DarkGray)));
    for (i, cell) in cells.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled(" │ ", d.c(Color::DarkGray)));
        }
        // Make first row (header) bold
        spans.push(Span::styled(cell.to_string(), Style::default()));
    }
    spans.push(Span::styled(" │", d.c(Color::DarkGray)));
    Line::from(spans)
}

/// Parse inline bold (**text**) and italic (*text*) markers
fn parse_inline_styles(s: &str, d: &DisplayEngine) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut rem = s;

    while !rem.is_empty() {
        // Bold: **text**
        if let Some(open) = rem.find("**") {
            if open > 0 {
                // Check for italic * before bold
                let before = &rem[..open];
                spans.extend(parse_italic_only(before, d));
            }
            let after = &rem[open + 2..];
            if let Some(close) = after.find("**") {
                spans.push(Span::styled(after[..close].to_string(), d.bld()));
                rem = &after[close + 2..];
            } else {
                // No closing **, emit rest as italic-style (stray bold marker)
                spans.push(Span::from(rem.to_string()));
                rem = "";
            }
        }
        // Italic: *text* (but not **)
        else if let Some(open) = rem.find('*') {
            // Check if it's actually ** by looking ahead
            if rem.len() > open + 1 && rem.as_bytes().get(open + 1) == Some(&b'*') {
                // It's a bold start — emit as plain text up to here and continue
                spans.push(Span::from(rem[..open + 1].to_string()));
                rem = &rem[open + 1..];
            } else {
                let before = &rem[..open];
                if !before.is_empty() {
                    spans.push(Span::from(before.to_string()));
                }
                let after = &rem[open + 1..];
                if let Some(close) = after.find('*') {
                    spans.push(Span::styled(after[..close].to_string(), d.c(Color::Cyan).italic()));
                    rem = &after[close + 1..];
                } else {
                    spans.push(Span::from(rem.to_string()));
                    rem = "";
                }
            }
        }
        // Inline code: `text`
        else if let Some(open) = rem.find('`') {
            let before = &rem[..open];
            if !before.is_empty() {
                spans.push(Span::from(before.to_string()));
            }
            let after = &rem[open + 1..];
            if let Some(close) = after.find('`') {
                spans.push(Span::styled(after[..close].to_string(), d.c(Color::Green).bg(Color::DarkGray)));
                rem = &after[close + 1..];
            } else {
                spans.push(Span::from(rem.to_string()));
                rem = "";
            }
        } else {
            spans.push(Span::from(rem.to_string()));
            rem = "";
        }
    }

    spans
}

/// Parse only italic (*) in text (helper to avoid double-parsing when bold already handled)
fn parse_italic_only(s: &str, d: &DisplayEngine) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut rem = s;

    while !rem.is_empty() {
        if let Some(open) = rem.find('*') {
            let before = &rem[..open];
            if !before.is_empty() {
                spans.push(Span::from(before.to_string()));
            }
            let after = &rem[open + 1..];
            if let Some(close) = after.find('*') {
                spans.push(Span::styled(after[..close].to_string(), d.c(Color::Cyan).italic()));
                rem = &after[close + 1..];
            } else {
                spans.push(Span::from(rem.to_string()));
                rem = "";
            }
        } else {
            spans.push(Span::from(rem.to_string()));
            rem = "";
        }
    }

    spans
}

fn parse_numbered(s: &str) -> Option<(usize, &str)> {
    let dot = s.find('.')?;
    let num: usize = s[..dot].parse().ok()?;
    Some((num, s[dot + 1..].trim_start()))
}

// ── helpers ──
#[allow(dead_code)]
fn se_avg(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len().max(1) as f64 }
#[allow(dead_code)]
fn cv_avg(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len().max(1) as f64 }
pub fn trunc(s: &str, max: usize) -> String { if s.len() <= max { s.to_string() } else { format!("{}…", &s[..max - 1]) } }
#[allow(dead_code)]
fn tc(dtype: &ColumnType) -> Color {
    match dtype { ColumnType::Numeric => Color::Green, ColumnType::DateTime => Color::Cyan, ColumnType::Categorical => Color::Yellow, ColumnType::String => Color::Gray }
}

/// Public wrapper: parse markdown text into styled ratatui Text for use in TUI rendering.
/// Convenience function that creates a temporary DisplayEngine internally.
pub fn render_markdown(text: &str) -> Text<'static> {
    let d = DisplayEngine::new();
    parse_markdown_to_text(text, &d)
}

/// Render any ratatui Widget to a String using the actual terminal size.
/// Falls back to 120x60 if terminal size cannot be determined.
#[allow(dead_code)]
fn render_widget(w: impl Widget) -> String {
    let (term_width, term_height) = crossterm::terminal::size()
        .map(|(w, h)| (w, h))
        .unwrap_or((120, 60));

    // Use a reasonable height: fill the terminal but cap at 200 to avoid massive allocations
    let width = term_width.max(40).min(400);
    let height = term_height.max(10).min(200);

    let area = ratatui::layout::Rect::new(0, 0, width, height);
    let mut buf = ratatui::buffer::Buffer::empty(area);
    w.render(area, &mut buf);

    let mut lines = Vec::new();
    for y in 0..height {
        let line: String = (0..width)
            .map(|x| buf.cell((x, y)).unwrap().symbol().chars().next().unwrap_or(' '))
            .collect();
        let t = line.trim_end().to_string();
        lines.push(t);
    }
    // Trim trailing empty lines
    while lines.last().map(|l| l.is_empty()).unwrap_or(false) {
        lines.pop();
    }
    lines.join("\n")
}
