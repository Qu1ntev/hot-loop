<div align="center">
    <h1>⚡ Hot-Loop</h1>
    <p><strong>
    High-Level 🦀 Pure-Rust Crate for Running Gguf Chat-Models,
    Uses the Candle 🕯️ Backend
    </strong></p>
</div>

---

<div align="center">
    <p><strong>
    This project is currently in Beta. API is subject to change
    </strong></p>
</div>

---

[![Crates.io](https://img.shields.io/crates/v/hot-loop.svg)](https://crates.io/crates/hot-loop)

## Quick Start

### Install: ```cargo add hot-loop```

```rust
use std::fs::{File, read};
use std::io::{stdout, Write};

use hot_loop::{
    Model,
    models::Qwen3,
    Device,
    Error,
};

fn main() -> Result<(), Error> {
    let mut model_file = File::open("models/Qwen3-4B-it-Q4_K_M.gguf").unwrap();
    let tokenizer_bytes = read("models/tokenizer.json").unwrap();

    // model read only
    let model = Qwen3::load(&mut model_file, &tokenizer_bytes, &Device::Cpu)?;

    let mut session = model.new_session();
    // and more sessions!
    // let mut session2 = model.new_session();
    // let mut session3 = model.new_session();

    let mut generate = session.generate("Hello!")?;

    while let Some(chunk) = generate.next_chunk()? {
        print!("{chunk}");
        stdout().flush().unwrap();
    }

    Ok(())
}
```

---

# Boost Your Code 🚀🦀

## Use ```target-cpu=native``` to boost generation speed!

```
your-project/
├── .cargo/
│   └── config.toml
├── src/
│   └── main.rs
└── Cargo.toml
```

## .cargo/config.toml:

```toml
rustflags = ["-C", "target-cpu=native"]
```

## Cargo.toml:

```toml
[profile.release]
lto = "fat"
opt-level = 3
strip = true
codegen-units = 1
panic = "abort"
```

---

## Typing

```rust
use std::fs::{File, read};

use hot_loop::{
    models::Qwen3,
    session::{Session, Generation},
    Model, // trait
    Device,
    Error
};

fn func1(_model: &impl Model) {}

fn func2(_session: &mut Session<impl Model>) {}

fn func3(_generation: &mut Generation<impl Model>) {}

fn main() -> Result<(), Error> {
    let mut model_file = File::open("Qwen3.gguf").unwrap();
    let tokenizer_bytes = read("tokenizer.json").unwrap();

    let model = Qwen3::load(&mut model_file, &tokenizer_bytes, &Device::Cpu)?;
    func1(&model);

    let mut session: Session<Qwen3> = model.new_session();
    func2(&mut session);

    let mut generation: Generation<Qwen3> = session.generate("Hello")?;
    func3(&mut generation);

    Ok(())
}
```

---

## Session Settings

```rust
use std::fs::{File, read};

use hot_loop::{
    Model,
    models::Qwen3,
    Device,
    Error,
    settings::{Settings, Seed},
};

fn main() -> Result<(), Error> {
    let mut model_file = File::open("Qwen3.gguf").unwrap();
    let tokenizer_bytes = read("tokenizer.json").unwrap();

    let model = Qwen3::load(&mut model_file, &tokenizer_bytes, &Device::Cpu)?;

    let settings = Settings::default()
        .with_temperature(0.7)
        .with_sample_len(512)
        .with_seed(Seed::Custom(12345)) // or Seed::Default
        .with_top_p(Some(0.5))
        .with_top_k(Some(40))
        .with_repeat_penalty(1.1)
        .with_repeat_last_n(64);

    let mut session = model.new_session()
        .with_settings(settings); // set settings
    
    // OR
    
    session.set_settings(Settings::default()); // set settings

    Ok(())
}
```

---

## Session System-prompt

```rust
use std::fs::{File, read};

use hot_loop::{
    Model,
    models::Qwen3,
    Device,
    Error,
};

fn main() -> Result<(), Error> {
    let mut model_file = File::open("Qwen3.gguf").unwrap();
    let tokenizer_bytes = read("tokenizer.json").unwrap();

    let model = Qwen3::load(&mut model_file, &tokenizer_bytes, &Device::Cpu)?;

    let sys_prompt = "always answer in json!";

    let mut session = model.new_session()
        .with_system_prompt(sys_prompt)?; // set system prompt

    // OR
    session.set_system_prompt_and_clear_history(sys_prompt)?;
    
    
    session.clear_system_prompt_and_history(); // clear system prompt

    Ok(())
}
```

---

## Session History

```rust
use std::fs::{File, read};

use hot_loop::{
    Model,
    models::Qwen3,
    Device,
    Error,
};

fn main() -> Result<(), Error> {
    let mut model_file = File::open("Qwen3.gguf").unwrap();
    let tokenizer_bytes = read("tokenizer.json").unwrap();

    let model = Qwen3::load(&mut model_file, &tokenizer_bytes, &Device::Cpu)?;
    let mut session = model.new_session();

    let questions = ["Hello!", "what can you do?", "ok"];

    for question in questions {
        let mut generate = session.generate(question)?;
        while let Some(_) = generate.next_chunk()? {
            // model answers
        }
    }

    session.clear_history(); // clear history

    Ok(())
}
```

---