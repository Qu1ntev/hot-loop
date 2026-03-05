use tokenizers::Tokenizer;
use crate::{Error, Role};

const IM_START: &str =  "<|im_start|>";
const IM_END: &str =    "<|im_end|>";
const SYSTEM: &str =    "system";
const USER: &str =      "user";
const ASSISTANT: &str = "assistant";
const NEW_LINE: &str =  "\n";

#[derive(Clone)]
pub(crate) struct ChatTemplate {
    tokenizer: Tokenizer,

    im_start: u32,
    im_end: u32,

    system: u32,
    user: u32,
    assistant: u32,

    new_line: u32,
}

impl ChatTemplate {
    pub fn new(
        tokenizer: Tokenizer
    ) -> Result<Self, Error> {
        let get = |text: &str| tokenizer.token_to_id(text).ok_or(Error::None);

        let im_start =  get(IM_START)?;
        let im_end =    get(IM_END)?;

        let system =    get(SYSTEM)?;
        let user =      get(USER)?;
        let assistant = get(ASSISTANT)?;

        let new_line = *tokenizer.encode(NEW_LINE, false)?
            .get_ids().get(0).ok_or(Error::None)?;

        Ok(Self {
            tokenizer,

            im_start,
            im_end,

            system,
            user,
            assistant,

            new_line
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    
    /// ## output ids:
    /// ```rust
    /// "<|im_start|>{role}\n{prompt}<|im_end|>\n"
    /// ```
    pub fn fmt_prompt(&self, prompt: &str, role: Role) -> Result<Vec<u32>, Error> {
        let role = match role {
            Role::System => self.system,
            Role::User => self.user,
            Role::Assistant => self.assistant,
        };

        let left = [self.im_start, role, self.new_line];
        let right = [self.im_end, self.new_line];

        let prompt = self.tokenizer.encode(prompt, false)?;

        let mut tokens = Vec::with_capacity(
            left.len() + prompt.get_ids().len() + right.len()
        );

        tokens.extend_from_slice(&left);
        tokens.extend_from_slice(prompt.get_ids());
        tokens.extend_from_slice(&right);

        Ok(tokens)
    }
    
    /// ## output ids:
    /// ```rust
    /// "<|im_start|>assistant\n"
    /// ```
    pub fn assistant_start_template(&self) -> Vec<u32> { // FIX!!! think mode
        vec![self.im_start, self.assistant, self.new_line]
    }
    
    pub fn eos_token(&self) -> u32 {
        self.im_end
    }
}