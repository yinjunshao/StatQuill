// Custom text input widget for ratatui
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style, Stylize},
    widgets::{Block, Borders, Widget},
};

/// A text input field that renders a label, content, and cursor.
pub struct TextInput<'a> {
    pub label: &'a str,
    pub buffer: &'a str,
    pub cursor: usize,
    pub focused: bool,
    pub block: Option<Block<'a>>,
    pub label_style: Style,
    pub input_style: Style,
    pub cursor_style: Style,
    pub mask: Option<char>,
}

impl<'a> TextInput<'a> {
    pub fn new(label: &'a str, buffer: &'a str, cursor: usize) -> Self {
        Self {
            label,
            buffer,
            cursor,
            focused: false,
            block: None,
            label_style: Style::default(),
            input_style: Style::default(),
            cursor_style: Style::default().add_modifier(ratatui::style::Modifier::REVERSED),
            mask: None,
        }
    }

    pub fn focused(mut self, focused: bool) -> Self {
        self.focused = focused;
        self
    }

    pub fn block(mut self, block: Block<'a>) -> Self {
        self.block = Some(block);
        self
    }
}

impl<'a> Widget for TextInput<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = self.block.unwrap_or_else(|| {
            let border_style = if self.focused {
                Style::default().fg(Color::Cyan)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            Block::default()
                .borders(Borders::ALL)
                .border_style(border_style)
                .title(self.label)
                .title_style(if self.focused {
                    Style::default().fg(Color::Cyan).bold()
                } else {
                    Style::default().fg(Color::Gray)
                })
        });
        let inner = block.inner(area);
        block.render(area, buf);

        // Render input text with cursor
        let max_width = inner.width as usize;
        if max_width == 0 {
            return;
        }

        // Calculate visible portion
        let start = if self.cursor >= max_width {
            self.cursor - max_width + 1
        } else {
            0
        };

        let display_text: String = if let Some(mask) = self.mask {
            self.buffer.chars().map(|_| mask).collect()
        } else {
            self.buffer.to_string()
        };

        let visible = if display_text.len() > start {
            &display_text[start..]
        } else {
            ""
        };

        for (i, ch) in visible.chars().enumerate() {
            if i < max_width {
                buf[(inner.x + i as u16, inner.y)]
                    .set_char(ch)
                    .set_style(self.input_style);
            }
        }

        // Render cursor
        if self.focused {
            let cursor_x = (self.cursor - start).min(max_width.saturating_sub(1));
            let cx = inner.x + cursor_x as u16;
            if cx < inner.x + inner.width {
                let cell = &mut buf[(cx, inner.y)];
                if self.cursor < self.buffer.len() {
                    // Set cursor style on existing character
                    cell.set_style(self.cursor_style);
                } else {
                    // Cursor at end - show block cursor
                    cell.set_char(' ')
                        .set_style(self.cursor_style);
                }
            }
        }
    }
}
