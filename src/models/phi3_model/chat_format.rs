use tokenizers::Tokenizer;
use crate::Error;
use crate::session::history::{Message, Role};
use super::super::models_core::model::ChatTemplate;

const END: &str =             "<|end|>";
const SYSTEM_START: &str =    "<|system|>";
const USER_START: &str =      "<|user|>";
const ASSISTANT_START: &str = "<|assistant|>";
const NEW_LINE: &str =  "\n";

pub(crate) struct ChatFormat {
    end: u32,

    system_start: u32,
    user_start: u32,
    assistant_start: u32,

    new_line: u32,
}

impl ChatFormat {
    pub fn new(
        tokenizer: &Tokenizer
    ) -> Result<Self, Error> {
        let get = |text: &str| tokenizer.token_to_id(text)
            .ok_or_else(|| Error::MissingValue(format!("No token named: {text}")));

        let end =             get(END)?;

        let system_start =    get(SYSTEM_START)?;
        let user_start =      get(USER_START)?;
        let assistant_start = get(ASSISTANT_START)?;

        let new_line = *tokenizer
            .encode(NEW_LINE, false)?
            .get_ids()
            .get(0)
            .ok_or_else(|| Error::MissingValue("No token named \\n".into()))?;

        Ok(Self {
            end,

            system_start,
            user_start,
            assistant_start,

            new_line
        })
    }
}

impl ChatTemplate for ChatFormat {
    /// ## output ids:
    /// ```rust
    /// "<|{role}|>\n{prompt}<|end|>\n"
    /// ```
    fn fmt_history(
        &self,
        tokenizer: &Tokenizer,
        history: &[Message],
        add_start: bool,
    ) -> Result<Vec<u32>, Error> {
        let mut tokens = Vec::new();

        for message in history {
            let role = message.role;
            let text = message.text.as_str();

            let role_start = match role {
                Role::System => self.system_start,
                Role::User => self.user_start,
                Role::Assistant => self.assistant_start,
            };

            let left = [role_start, self.new_line];
            let right = [self.end, self.new_line];

            let text_tk = tokenizer.encode(text, false)?;

            let mut tk = Vec::with_capacity(
                left.len() + text_tk.get_ids().len() + right.len()
            );

            tk.extend_from_slice(&left);
            tk.extend_from_slice(text_tk.get_ids());
            tk.extend_from_slice(&right);

            tokens.extend_from_slice(&tk);
        }

        if add_start {
            let start_template = [self.assistant_start, self.new_line];
            tokens.extend_from_slice(&start_template);
        }

        Ok(tokens)
    }

    fn eos_token(&self) -> u32 {
        self.end
    }
}