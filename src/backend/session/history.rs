#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Message {
    pub role: Role,
    pub message: String,
}

impl Message {
    pub fn new<S: Into<String>>(role: Role, message: S) -> Self {
        Self { role, message: message.into() }
    }
}