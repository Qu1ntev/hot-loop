#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Message {
    pub role: Role,
    pub text: String,
}

impl Message {
    pub fn new<S: Into<String>>(role: Role, text: S) -> Self {
        Self { role, text: text.into() }
    }
}