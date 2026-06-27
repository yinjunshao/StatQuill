// Silence dead-code warnings for legacy modules during TUI transition.
// These APIs are kept for backward compatibility but aren't directly called by the new TUI.
#![allow(dead_code)]

mod ai;
mod commentary;
mod config;
mod display;
mod math;
mod parser;
mod prediction;
mod tui;

use anyhow::Result;
use clap::Parser;

/// StatQuill - Predictive Analytics CLI
///
/// A terminal-based prediction system for CSV/Excel data with AI-enhanced commentary.
#[derive(Parser, Debug)]
#[command(
    name = "statquill",
    version,
    about = "StatQuill - Predictive Analytics CLI",
    long_about = "A terminal-based prediction system for CSV/Excel data with AI-enhanced commentary.\n\nExamples:\n  statquill data.csv\n  statquill sales.xlsx --context \"Q3 retail sales forecast\"\n  statquill --setup"
)]
struct Cli {
    /// Path to data file (CSV, XLSX, TSV, TXT)
    file: Option<String>,

    /// Domain context for AI analysis
    #[arg(short = 'c', long, default_value = "")]
    context: String,

    /// Run configuration wizard
    #[arg(short = 's', long)]
    setup: bool,

    /// Override OpenRouter model
    #[arg(short = 'm', long)]
    model: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load (or create) config
    let cfg = config::ConfigManager::load_or_create()?;

    let app = tui::App::new(cfg, cli.model.clone());

    // If file was passed on command line, pre-fill it
    // (We handle this by setting state; the TUI requires manual file input for flexibility)
    tui::run_app(app)?;

    Ok(())
}
