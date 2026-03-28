use tokenizers::Tokenizer;
use crate::Error;
use crate::session::history::{Message, Role};
use crate::ChatFormat;

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const SYSTEM: &str = "system";
const USER: &str = "user";
const ASSISTANT: &str = "assistant";
const NEW_LINE: &str = "\n";

pub struct Qwen3Format {
    tokenizer: Tokenizer,

    im_start: u32,
    im_end: u32,
    eos_tokens: [u32; 2],

    system: u32,
    user: u32,
    assistant: u32,

    new_line: u32,
}

impl Qwen3Format {
    pub fn new(
        tokenizer: Tokenizer
    ) -> Result<Self, Error> {
        let get = |text: &str| tokenizer.token_to_id(text)
            .ok_or_else(|| Error::MissingValue(format!("No token named '{text}'")));

        let im_start =  get(IM_START)?;
        let im_end =    get(IM_END)?;

        let system =    get(SYSTEM)?;
        let user =      get(USER)?;
        let assistant = get(ASSISTANT)?;

        let new_line = *tokenizer
            .encode(NEW_LINE, false)?
            .get_ids()
            .get(0)
            .ok_or_else(|| Error::MissingValue(format!("No token named '{NEW_LINE}'")))?;
        
        let eos_tokens = [im_end, new_line];
        
        Ok(Self {
            tokenizer,

            im_start,
            im_end,
            eos_tokens,

            system,
            user,
            assistant,

            new_line
        })
    }
}

impl ChatFormat for Qwen3Format {
    fn format_for_gen(&self, messages: &[Message], model_start: bool) -> Result<Vec<u32>, Error> {
        let mut tokens = Vec::new();

        for message in messages {
            let text = message.text.as_str();

            let role = match message.role {
                Role::System => self.system,
                Role::User => self.user,
                Role::Assistant => self.assistant,
            };

            let left = [self.im_start, role, self.new_line];
            let right = [self.im_end, self.new_line];

            let prompt = self.tokenizer.encode(text, false)?;

            let mut tk = Vec::with_capacity(
                left.len() + prompt.get_ids().len() + right.len()
            );

            tk.extend_from_slice(&left);
            tk.extend_from_slice(prompt.get_ids());
            tk.extend_from_slice(&right);

            tokens.extend_from_slice(&tk);
        }

        if model_start {
            let assistant_start = [self.im_start, self.assistant, self.new_line];
            tokens.extend_from_slice(&assistant_start);
        }

        Ok(tokens)
    }

    fn eos_tokens(&self) -> &[u32] {
        &self.eos_tokens
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}