use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConfigManager {
    api_key: Option<String>,
    model: String,
}

impl ConfigManager {
    pub fn load_or_create() -> Result<Self> {
        let config_path = config_path();
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)
                .context("Failed to read config file")?;
            toml::from_str(&content).context("Failed to parse config file")
        } else {
            let cfg = ConfigManager {
                api_key: None,
                model: "anthropic/claude-3.5-sonnet".to_string(),
            };
            cfg.save()?;
            Ok(cfg)
        }
    }

    fn save(&self) -> Result<()> {
        let config_path = config_path();
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create config directory")?;
        }
        let content = toml::to_string_pretty(self).context("Failed to serialize config")?;
        std::fs::write(&config_path, content).context("Failed to write config file")?;
        Ok(())
    }

    pub fn is_configured(&self) -> bool {
        self.api_key.as_ref().map_or(false, |k| k.len() > 20)
    }

    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn set_api_key(&mut self, key: String) -> Result<()> {
        self.api_key = Some(key);
        self.save()
    }

    pub fn set_model(&mut self, model: String) -> Result<()> {
        self.model = model;
        self.save()
    }
}

fn config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("statquill")
        .join("config.toml")
}

pub fn setup_wizard() -> Result<()> {
    println!();
    println!("\x1b[1;33mWelcome! Let's set up StatQuill.\x1b[0m");
    println!();

    // Read API key
    print!("Enter your OpenRouter API Key: ");
    io::stdout().flush()?;
    let mut api_key = String::new();
    io::stdin().read_line(&mut api_key)?;
    let api_key = api_key.trim().to_string();

    if api_key.len() < 20 {
        println!("\x1b[31mThat doesn't look like a valid API key.\x1b[0m");
        print!("Continue anyway? (y/N): ");
        io::stdout().flush()?;
        let mut answer = String::new();
        io::stdin().read_line(&mut answer)?;
        if answer.trim().to_lowercase() != "y" {
            return Ok(());
        }
    }

    // Read model
    print!("Enter OpenRouter Model [anthropic/claude-3.5-sonnet]: ");
    io::stdout().flush()?;
    let mut model = String::new();
    io::stdin().read_line(&mut model)?;
    let model = model.trim().to_string();
    let model = if model.is_empty() {
        "anthropic/claude-3.5-sonnet".to_string()
    } else {
        model
    };

    let mut cfg = ConfigManager {
        api_key: None,
        model: "anthropic/claude-3.5-sonnet".to_string(),
    };
    cfg.set_api_key(api_key)?;
    cfg.set_model(model)?;

    println!("\x1b[32mConfiguration saved!\x1b[0m");
    Ok(())
}
