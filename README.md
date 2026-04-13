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

# Quick Start

### Install: ```cargo add hot-loop```

```rust
use std::fs::File;
use std::io::{stdout, Write};

use hot_loop::{
    Model,
    models::Qwen3,
    models::ModelBuilder,
    session::history::{Message, Role},
    Device,
    Error,
};

fn main() -> Result<(), Error> {
    let model_file = File::open("models/Qwen3-4B-it-Q4_K_M.gguf").unwrap(); // Read + Seek

    // model read only
    let model: Qwen3 = ModelBuilder::new(model_file)
        .load(Device::Cpu)?;

    let mut session = model.new_session();
    // and more sessions!
    // let mut session2 = Session::new(Arc::new(model));
    // let mut session3 = (&model).new_session();
    
    let history = vec![
        Message::new(Role::User, "Hello!")
    ];
    
    let mut generate = session.generate(&history)?;

    while let Some(chunk) = generate.next_chunk()? {
        print!("{chunk}");
        stdout().flush().unwrap();
    }

    Ok(())
}
```

---

# ⚡ [Benchmarks](benchmarks/README.md)

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
[build]
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

# Enable CUDA 🔋

## Cargo.toml:
```toml
[dependencies]
hot-loop = { version = "...", features = ["cuda"] }
```

## Rust Code
```diff
- let device = Device::Cpu;
+ let device = Device::new_cuda(0)?;
```

---

# Typing

```rust
use std::fs::{File, read};
use std::sync::Arc;

use hot_loop::{
    models::Qwen3,
    models::ModelBuilder,
    session::{Session, Generation},
    session::history::{Message, Role},
    Model, // trait
    Device,
    Error,
    Dtype
};

fn func1<M: Model>(_model: M) {}
fn func2(_session: &mut Session<impl Model>) {}
fn func3(_generation: &mut Generation<impl Model>) {}

fn main() -> Result<(), Error> {
    let model_file = File::open("Qwen3.gguf").unwrap();
    let tokenizer_bytes = read("tokenizer.json").unwrap();

    let model: Qwen3 = ModelBuilder::new(model_file)
        .with_tokenizer(tokenizer_bytes)
        .with_dtype(Dtype::F16)
        .load::<Qwen3>(Device::Cpu)?;
    
    func1(&model);

    let mut session: Session<Qwen3> = model.new_session();
    func2(&mut session);

    let history = vec![
        Message::new(Role::User, "Hello!")
    ];

    let mut generation: Generation<Qwen3> = session.generate(&history)?;
    func3(&mut generation);

    let model: Qwen3 = model.clone();
    let model_ref: &Qwen3 = &model.clone();
    let model_arc: Arc<Qwen3> = Arc::new(model.clone());
    let model_arc_dyn: Arc<dyn Model> = Arc::new(model.clone());
    let model_box_dyn: Box<dyn Model> = Box::new(model.clone());

    let _session: Session<Qwen3> = model.new_session();
    let _session: Session<&Qwen3> = model_ref.new_session();
    let _session: Session<Arc<Qwen3>> = model_arc.new_session();
    let _session: Session<Arc<dyn Model>> = model_arc_dyn.new_session();
    let _session: Session<Box<dyn Model>> = model_box_dyn.new_session();

    Ok(())
}
```

---

# Session Settings

```rust
use std::fs::File;

use hot_loop::{
    Model,
    models::Qwen3,
    models::ModelBuilder,
    Device,
    Error,
    settings::{Settings, Seed},
};

fn main() -> Result<(), Error> {
    let model_file = File::open("Qwen3.gguf").unwrap();

    let model: Qwen3 = ModelBuilder::new(model_file)
        .load(Device::Cpu)?;

    let settings = Settings::default()
        .with_temperature(0.7)
        .with_sample_len(512)
        .with_seed(Seed::Custom(12345)) // or Seed::Random
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

# Session System-prompt

```rust
use std::fs::File;
use std::io::{stdout, Write};

use hot_loop::{
    Model,
    models::Qwen3,
    models::ModelBuilder,
    session::history::{Message, Role},
    Device,
    Error,
};

fn main() -> Result<(), Error> {
    let model_file = File::open("Qwen3.gguf").unwrap();

    let model: Qwen3 = ModelBuilder::new(model_file)
        .load(Device::Cpu)?;

    let mut history = vec![ // set system prompt
        Message::new(Role::System, "always answer in json!")
    ];

    let mut session = model.new_session();
    history.push(Message::new(Role::User, "Hello!"));
    
    let mut generate = session.generate(&history)?;

    while let Some(chunk) = generate.next_chunk()? {
        print!("{chunk}");
        stdout().flush().unwrap();
    }

    Ok(())
}
```

---

# Session History

```rust
use std::fs::File;
use std::io::{stdout, Write};

use hot_loop::{
    Model,
    models::Qwen3,
    models::ModelBuilder,
    session::history::{Message, Role},
    Device,
    Error,
};

fn main() -> Result<(), Error> {
    let model_file = File::open("Qwen3.gguf").unwrap();

    let model: Qwen3 = ModelBuilder::new(model_file)
        .load(Device::Cpu)?;
    
    let mut session = model.new_session();

    let questions = ["Hello!", "what can you do?", "ok"];
    let mut history = vec![];

    for question in questions {
        history.push(Message::new(Role::User, question));
        let mut answer = String::new();
        
        let mut generate = session.generate(&history)?;
        while let Some(chunk) = generate.next_chunk()? {
            print!("{chunk}");
            stdout().flush().unwrap();
            answer.push_str(&chunk);
        }
        
        history.push(Message::new(Role::Assistant, answer)); // update history
    }

    history.clear(); // clear history
    session.clear_cache(); // clear session kv cache

    Ok(())
}
```

---

# Thread Safety
### Parallelism Generation in different independent sessions

```rust
use std::fs::File;
use std::thread;
use std::sync::Arc;
use hot_loop::{
    Model,
    models::Qwen3,
    models::ModelBuilder,
    session::Session,
    session::history::{Message, Role},
    Device,
    Error,
};

fn generate<M: Model>(mut session: Session<M>, prompt: &str) -> Result<String, Error> {
    let history = vec![Message::new(Role::User, prompt)];
    let mut answer = String::new();
    
    let mut generate = session.generate(&history)?;

    while let Some(chunk) = generate.next_chunk()? {
        answer.push_str(&chunk);
    }
    Ok(answer)
}

// one question for one thread
const QUESTIONS: [&str; 4] = [
    "hello!",
    "what can you do?",
    "what is the book war and peace about?",
    "make a plan for my trip to Shanghai"
];

fn main() -> Result<(), Error> {
    let model_file = File::open("models/Qwen3-4B-it-Q4_K_M.gguf").unwrap();

    let model = Arc::new(ModelBuilder::new(model_file).load::<Qwen3>(Device::Cpu)?);

    let mut handles = Vec::new();

    for prompt in QUESTIONS {
        let session = model.clone().new_session();
        // OR
        // let session: Session<Arc<Qwen3>> = Session::new(model.clone());

        let handle = thread::spawn(move || {
            generate(session, prompt)
        });
        handles.push(handle);
    }

    for handle in handles {
        let answer = handle.join().unwrap()?;
        println!("{answer}");
    }

    Ok(())
}
```

---