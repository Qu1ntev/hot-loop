#[derive(Debug, Clone, Copy)]
pub struct Settings {
    pub sample_len: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            sample_len: 256,
            temperature: 0.7,
            top_p: None,
            top_k: None,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        }
    }
}