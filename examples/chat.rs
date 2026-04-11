use std::fs::File;
use std::io::{stdin, stdout, Write};
use std::str::FromStr;
use hot_loop::{
    Model,
    models::Qwen3,
    models::ModelBuilder,
    session::history::{Message, Role},
    Device,
    Error,
};

const MODEL_PATH: &str = "models/Qwen3-4B-it-Q4_K_M.gguf";
// const TOK_PATH: &str = "models/tokenizer.json";

fn input<T: FromStr>() -> Result<T, Error> {
    let mut s = String::new();
    stdin().read_line(&mut s).map_err(
        |_| Error::MissingValue("input error".into())
    )?;
    s.trim().parse().map_err(
        |_| Error::MissingValue("parse error".into())
    )
}

fn main() -> Result<(), Error> {
    println!("-------- | Hot-Loop Chat | --------");

    let model_file = File::open(MODEL_PATH)
        .expect(&format!("MODEL FILE NOT FOUND: {}", MODEL_PATH));

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Run with: {:?}", device);

    println!("Set System-Prompt? [y/n]:");
    let is_set: String = input()?;
    let sys_prompt: Option<String> = match is_set.as_str() {
        "y" => {
            println!("Please paste System-Prompt:");
            let sys_prompt: String = input()?;
            Some(sys_prompt)
        },
        _ => None
    };

    let mut history = Vec::new();

    if let Some(sys_prompt) = sys_prompt {
        history.push(Message::new(Role::System, sys_prompt));
    }

    println!("Running Model...");
    // model read only
    let model: Qwen3 = ModelBuilder::new(model_file)
        .load(Device::Cpu)?;

    let mut session = model.new_session();
    // and more sessions!
    // let mut session2 = Session::new(Arc::new(model));
    // let mut session3 = (&model).new_session();

    loop {
        print!("\n\n");
        let mut answer = String::new();

        print!("prompt: ");
        stdout().flush().unwrap();
        let prompt: String = input()?;

        history.push(Message::new(Role::User, prompt));

        let mut generate = session.generate(&history)?;

        while let Some(chunk) = generate.next_chunk()? {
            print!("{chunk}");
            stdout().flush().unwrap();
            answer.push_str(&chunk);
        }

        history.push(Message::new(Role::Assistant, answer));
    }
}