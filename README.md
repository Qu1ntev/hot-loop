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

## Quick Start

```rust
use std::fs::{File, read};
use std::io::{stdout, Write};

use hot_loop::{
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