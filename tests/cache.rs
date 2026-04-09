use std::fs::{File, read};
use std::io::{stdout, Write};

use hot_loop::{
    Model,
    models::Qwen3,
    session::history::{Message, Role},
    Device,
    Error,
};

#[test]
fn cache_test() -> Result<(), Error> {
    let mut model_file = File::open("models/Qwen3-4B-it-Q4_K_M.gguf").unwrap();
    let tokenizer_bytes = read("models/tokenizer.json").unwrap();

    let model = Qwen3::load(&mut model_file, &tokenizer_bytes, Device::Cpu)?;

    let mut session = model.new_session();

    let history_tests = vec![
        vec![
            Message::new(Role::System, "Your Assistant"),
            Message::new(Role::User, "Hello!"),
        ],
        vec![
            Message::new(Role::System, "Your Assistant"),
            Message::new(Role::User, "Hello!"),
        ],
    ];

    for history in history_tests {
        let mut generate = session.generate(&history)?;

        while let Some(chunk) = generate.next_chunk()? {
            print!("{chunk}");
            stdout().flush().unwrap();
        }

        print!("\n\n");
    }

    Ok(())
}