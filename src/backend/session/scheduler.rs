use super::super::settings::{Settings, Seed};
use crate::utils::kv_cache::KvCache;
use crate::utils::token_output_stream::TokenOutputStream;
use crate::models::models_core::model::Model;
// use candle_transformers::generation::{LogitsProcessor, Sampling};
// use crate::Error;
// use super::history::Role;
// use candle_core::Tensor;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

pub(crate) struct RunConfig {
    pub sender: Sender<String>,
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
                    cfg.sender.send(cfg.prompt).unwrap();
                },
                Err(_) => {
                    println!("stopping scheduler!");
                    break
                },
            }
        }
    }

    // fn generate(&mut self, run_config: RunConfig) -> Result<(), Error> {
    //     let user_tokens = self.model.fmt_prompt(&run_config.prompt, Role::User)?;
    //     let assistant_start_tokens = self.model.assistant_start_template();
    // 
    //     let mut tokens = Vec::with_capacity(
    //         user_tokens.len() + assistant_start_tokens.len()
    //     );
    // 
    //     tokens.extend_from_slice(&user_tokens);
    //     tokens.extend_from_slice(&assistant_start_tokens);
    // 
    //     let logits_processor = {
    //         let temperature = run_config.settings.temperature;
    //         let sampling = if temperature <= 0. {
    //             Sampling::ArgMax
    //         } else {
    //             match (run_config.settings.top_k, run_config.settings.top_p) {
    //                 (None, None) => Sampling::All { temperature },
    //                 (Some(k), None) => Sampling::TopK { k, temperature },
    //                 (None, Some(p)) => Sampling::TopP { p, temperature },
    //                 (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
    //             }
    //         };
    // 
    //         let seed = match run_config.settings.seed {
    //             Seed::Custom(seed) => seed,
    //             Seed::Random => rand::random()
    //         };
    // 
    //         LogitsProcessor::from_sampling(seed, sampling)
    //     };
    //     Ok(())
    // }
}