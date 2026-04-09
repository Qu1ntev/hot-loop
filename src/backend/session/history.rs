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

pub trait Msg {
    fn role(&self) -> Role;

    fn text(&self) -> &str;
}

impl Msg for Message {
    fn role(&self) -> Role {
        self.role
    }

    fn text(&self) -> &str {
        &self.text
    }
}

pub trait History {
    fn read(&self) -> &[impl Msg];
}

impl<T: History> History for &T {
    fn read(&self) -> &[impl Msg] {
        (**self).read()
    }
}

impl History for Vec<Message> {
    fn read(&self) -> &[impl Msg] {
        self
    }
}

impl History for &[Message] {
    fn read(&self) -> &[impl Msg] {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_history<H: History>(history: H) {
        for msg in history.read() {
            let _ = msg.role();
            let _ = msg.text();
        }
    }

    #[test]
    fn history_test() {
        let history = vec![
            Message::new(Role::System, "hello world"),
        ];

        let history_ref = vec![
            Message::new(Role::System, "hello world"),
        ];

        put_history(history);
        put_history(&history_ref);
    }
}