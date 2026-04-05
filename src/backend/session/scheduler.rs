use super::super::settings::{Settings, Seed};
use crate::utils::kv_cache::KvCache;
use crate::utils::token_output_stream::TokenOutputStream;
use crate::models::models_core::model::Model;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use crate::Error;
use super::history::Role;
use candle_core::Tensor;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

struct Generate<'session, M: Model> {
    model: &'session M,
    index: usize,
    next_token: u32,
    tokens_prefill: Option<Vec<u32>>,
    all_tokens: Vec<u32>,
    settings: Settings,
    eos_token: u32,
    logits_processor: LogitsProcessor,
    tos: &'session mut TokenOutputStream,
    kv_cache: &'session mut KvCache
}

impl<'session, M: Model> Generate<'session, M> {
    pub(crate) fn new(
        model: &'session M,
        tokens_prefill: Vec<u32>,
        logits_processor: LogitsProcessor,
        settings: Settings,
        tos: &'session mut TokenOutputStream,
        kv_cache: &'session mut KvCache,
        eos_token: u32,
    ) -> Self {
        Self {
            model,
            index: 0,
            next_token: 0,
            all_tokens: Vec::new(),
            tokens_prefill: Some(tokens_prefill),
            logits_processor,
            settings,
            tos,
            kv_cache,
            eos_token,
        }
    }

    pub fn next_chunk(&mut self) -> Result<Option<String>, Error> {
        loop {
            if self.settings.sample_len <= self.index || self.next_token == self.eos_token {
                return Ok(None);
            }

            let current_pos = self.kv_cache.current_pos();

            let input = if self.index == 0 &&
                let Some(tokens_prefill) = self.tokens_prefill.take() {
                Tensor::new(tokens_prefill.as_slice(), self.model.device())?.unsqueeze(0)?

            } else {
                Tensor::new(&[self.next_token], self.model.device())?.unsqueeze(0)?
            };

            let logits = self.model.forward(&input, current_pos, &mut self.kv_cache)?;
            let logits = logits.squeeze(0)?;

            let logits = if self.settings.repeat_penalty == 1. {
                logits
            } else {
                let start_at = self.all_tokens.len().saturating_sub(self.settings.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.settings.repeat_penalty,
                    &self.all_tokens[start_at..],
                )?
            };

            self.next_token = self.logits_processor.sample(&logits)?;
            self.all_tokens.push(self.next_token);

            self.index += 1;

            if let Some(chunk) = self.tos.next_token(self.model.tokenizer(), self.next_token)? {
                return Ok(Some(chunk))
            }
        }
    }
}

pub(crate) struct RunConfig {
    pub sender: Sender<Option<String>>,
    pub prompt: String,
    pub settings: Settings,
}

pub(crate) enum Action {
    Run(RunConfig)
}

pub(crate) struct Scheduler<M: Model> {
    model: M,
    action_receiver: Receiver<Action>,
    kv_cache: KvCache,
    tos: TokenOutputStream,
    system_prompt_pos: Option<usize>,
}

impl<M: Model> Scheduler<M> {
    pub fn spawn(
        model: M,
        kv_cache: KvCache,
        tos: TokenOutputStream,
    ) -> Sender<Action> {
        let (action_sender, action_receiver) = mpsc::channel();
        
        let mut scheduler = Self {
            model,
            action_receiver,
            kv_cache,
            tos,
            system_prompt_pos: None,
        };

        thread::spawn(move || {
            scheduler.main_loop();
        });
        
        action_sender
    }

    fn main_loop(&mut self) {
        // println!("schedule wait...");
        
        loop {
            match self.action_receiver.recv() {
                Ok(Action::Run(cfg)) => {
                    let mut stream = match self.generate(&cfg) {
                        Ok(stream) => stream,
                        Err(_) => continue,
                    };
                    while let Ok(chunk) = stream.next_chunk() {
                        let _ = cfg.sender.send(chunk);
                    }
                },
                Err(_) => {
                    println!("stopping scheduler!");
                    break
                },
            }
        }
    }

    fn generate(&mut self, run_config: &RunConfig) -> Result<Generate<'_, M>, Error> {
        let user_tokens = self.model.fmt_prompt(&run_config.prompt, Role::User)?;
        let assistant_start_tokens = self.model.assistant_start_template();

        let mut tokens = Vec::with_capacity(
            user_tokens.len() + assistant_start_tokens.len()
        );

        tokens.extend_from_slice(&user_tokens);
        tokens.extend_from_slice(&assistant_start_tokens);

        let logits_processor = {
            let temperature = run_config.settings.temperature;
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (run_config.settings.top_k, run_config.settings.top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };

            let seed = match run_config.settings.seed {
                Seed::Custom(seed) => seed,
                Seed::Random => rand::random()
            };

            LogitsProcessor::from_sampling(seed, sampling)
        };
        Ok(Generate::new(
            &self.model,
            tokens,
            logits_processor,
            run_config.settings,
            &mut self.tos,
            &mut self.kv_cache,
            self.model.eos_token(),
        ))
    }
}