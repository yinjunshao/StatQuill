use anyhow::{Context, Result};
use serde_json::Value;

/// A single message in a chat conversation
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String, // "user" or "assistant"
    pub content: String,
}

pub struct AIEnhancer {
    api_key: String,
    model: String,
    client: reqwest::blocking::Client,
    /// System prompt used for the initial commentary generation
    system_prompt: String,
    /// The initial user message (statistical payload + context)
    initial_user_content: String,
}

impl AIEnhancer {
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            client: reqwest::blocking::Client::new(),
            system_prompt: String::new(),
            initial_user_content: String::new(),
        }
    }

    /// Build the system and initial user prompts from stats + context.
    /// Must be called before `generate_commentary` or `chat`.
    pub fn build_prompts(&mut self, stats_payload: &Value, context: &str) {
        self.system_prompt = r#"You are StatQuill, an expert data analyst and strategic consultant. Analyze the provided statistical summary and prediction results. Provide:
1. A brief executive summary (2-3 sentences)
2. Key trends and patterns observed
3. Notable risks or uncertainties
4. Actionable insights based on the predictions
5. Overall interpretation — summarize what the dataset reveals, the core trend direction, and your consulting-style suggestions tailored to the provided context (if any). Your final paragraph should feel like practical advice from a data-savvy consultant.

Be concise, professional, and data-driven. Use markdown formatting."#.to_string();

        self.initial_user_content = if context.is_empty() {
            format!(
                "## Statistical Analysis Results\n\n```json\n{}\n```\n\nPlease provide your analysis.",
                serde_json::to_string_pretty(stats_payload).unwrap_or_default()
            )
        } else {
            format!(
                "## Statistical Analysis Results\n\n```json\n{}\n```\n\n## Domain Context\n{}\n\nPlease provide your analysis.",
                serde_json::to_string_pretty(stats_payload).unwrap_or_default(),
                context
            )
        };
    }

    /// Send raw statistics to OpenRouter and return humanized commentary
    pub fn generate_commentary(&self, stats_payload: &Value, context: &str) -> Result<String> {
        // Fallback for when build_prompts wasn't called — build inline
        let system_prompt = r#"You are StatQuill, an expert data analyst and strategic consultant. Analyze the provided statistical summary and prediction results. Provide:
1. A brief executive summary (2-3 sentences)
2. Key trends and patterns observed
3. Notable risks or uncertainties
4. Actionable insights based on the predictions
5. Overall interpretation — summarize what the dataset reveals, the core trend direction, and your consulting-style suggestions tailored to the provided context (if any). Your final paragraph should feel like practical advice from a data-savvy consultant.

Be concise, professional, and data-driven. Use markdown formatting."#;

        let user_content = if context.is_empty() {
            format!(
                "## Statistical Analysis Results\n\n```json\n{}\n```\n\nPlease provide your analysis.",
                serde_json::to_string_pretty(stats_payload).unwrap_or_default()
            )
        } else {
            format!(
                "## Statistical Analysis Results\n\n```json\n{}\n```\n\n## Domain Context\n{}\n\nPlease provide your analysis.",
                serde_json::to_string_pretty(stats_payload).unwrap_or_default(),
                context
            )
        };

        let messages = vec![
            serde_json::json!({"role": "system", "content": system_prompt}),
            serde_json::json!({"role": "user", "content": user_content}),
        ];

        self.send_request(&messages)
    }

    /// Send a chat message continuing from the existing conversation history.
    /// `history` should contain the full sequence of messages (system, user, assistant, …).
    /// The new user message is appended to history before sending.
    pub fn chat(&self, history: &[ChatMessage]) -> Result<String> {
        // Build messages from history: system prompt first, then the initial user message,
        // then all subsequent exchanges.
        let mut messages: Vec<serde_json::Value> = Vec::new();

        // Start with system prompt
        messages.push(serde_json::json!({
            "role": "system",
            "content": &self.system_prompt
        }));

        // Add initial user content
        messages.push(serde_json::json!({
            "role": "user",
            "content": &self.initial_user_content
        }));

        // Add the initial assistant response (commentary) if not already in history
        // Then add conversation history exchanges
        for msg in history {
            messages.push(serde_json::json!({
                "role": msg.role,
                "content": msg.content
            }));
        }

        self.send_request(&messages)
    }

    /// Low-level request to OpenRouter API
    fn send_request(&self, messages: &[serde_json::Value]) -> Result<String> {
        let payload = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1500
        });

        let response = self
            .client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("HTTP-Referer", "https://statquill.local")
            .header("X-Title", "StatQuill CLI")
            .json(&payload)
            .timeout(std::time::Duration::from_secs(120))
            .send()
            .context("Failed to contact OpenRouter API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(anyhow::anyhow!(
                "OpenRouter API error ({}): {}",
                status,
                body
            ));
        }

        let data: Value = response.json().context("Failed to parse OpenRouter response")?;

        data["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .context("Unexpected response format from OpenRouter")
    }
}
