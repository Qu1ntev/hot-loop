#[derive(Debug, Clone, Copy)]
pub struct Settings {
    pub sample_len: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: Option<u64>,
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
            seed: None,
        }
    }
}

impl Settings {
    pub fn with_sample_len(mut self, len: usize) -> Self {
        self.sample_len = len;
        self
    }
    
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }
    
    pub fn with_top_p(mut self, top_p: Option<f64>) -> Self {
        self.top_p = top_p;
        self
    }
    
    pub fn with_top_k(mut self, top_k: Option<usize>) -> Self {
        self.top_k = top_k;
        self
    }
    
    pub fn with_repeat_penalty(mut self, repeat_penalty: f32) -> Self {
        self.repeat_penalty = repeat_penalty;
        self
    }
    
    pub fn with_repeat_last_n(mut self, repeat_last_n: usize) -> Self {
        self.repeat_last_n = repeat_last_n;
        self
    }
    
    pub fn with_seed(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        self
    }
}