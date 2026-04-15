// use std::fs::File;
// use std::io::{stdout, Write};
//
// use hot_loop::{
//     Model,
//     models::Qwen3,
//     models::ModelBuilder,
//     session::history::{Message, Role},
//     Device,
//     Error,
// };
//
// fn get_history_tests() -> Vec<Vec<Message>> {
//     vec![
//         vec![
//             Message::new(Role::System, "Your Assistant"),
//             Message::new(Role::User, "Hello!"),
//         ],
//         vec![
//             Message::new(Role::System, "Your Assistant"),
//             Message::new(Role::User, "Hello!"),
//             Message::new(Role::Assistant, "Hi! How can I help?"),
//         ],
//         vec![
//             Message::new(Role::System, "Your Assistant"),
//             Message::new(Role::User, "Hello!"),
//         ],
//         vec![
//             Message::new(Role::User, "Hello!"),
//         ],
//         vec![
//             Message::new(Role::User, "Hello!"),
//             Message::new(Role::User, "Hello!"),
//             Message::new(Role::User, "Hello!"),
//         ],
//         vec![
//             Message::new(Role::System, "Your Assistant"),
//             Message::new(Role::User, "Hello!"),
//         ],
//         vec![
//             Message::new(Role::System, "Your Assistant"),
//             Message::new(Role::User, "sup!"),
//         ],
//     ]
// }
//
// #[test]
// fn cache_prefill() -> Result<(), Error> {
//     let model_file = File::open("models/Qwen3-4B-it-Q4_K_M.gguf").unwrap();
//     // let tokenizer_bytes = read("models/tokenizer.json").unwrap();
//
//     let model: Qwen3 = ModelBuilder::new(model_file)
//         .load(Device::Cpu)?;
//
//     let mut session = model.new_session();
//
//     let history_tests = get_history_tests();
//
//     for history in &history_tests {
//         {
//             let mut generate = session.generate(&history)?;
//
//             while let Some(chunk) = generate.next_chunk()? {
//                 print!("{chunk}");
//                 stdout().flush().unwrap();
//             }
//         }
//
//         let cached_tokens_len = session.cached_tokens().len();
//         let kv_cache_len = session.context();
//
//         assert_eq!(cached_tokens_len, kv_cache_len);
//
//         print!("\n\n");
//     }
//
//     session.clear_cache();
//
//     let cached_tokens_len = session.cached_tokens().len();
//     let kv_cache_len = session.context();
//
//     assert_eq!(cached_tokens_len, 0);
//     assert_eq!(kv_cache_len, 0);
//
//     Ok(())
// }
//
// #[test]
// fn cache_gen_drop() -> Result<(), Error> {
//     let model_file = File::open("models/Qwen3-4B-it-Q4_K_M.gguf").unwrap();
//
//     let model: Qwen3 = ModelBuilder::new(model_file)
//         .load(Device::Cpu)?;
//
//     let mut session = model.new_session();
//
//     let history_tests = get_history_tests();
//
//     {
//         let h = vec![Message::new(Role::User, "a Hello Qwen!")];
//         let _generate = session.generate(&h)?;
//     }
//
//     let cached_tokens_len = session.cached_tokens().len();
//     let kv_cache_len = session.context();
//
//     assert_eq!(cached_tokens_len, kv_cache_len);
//
//     for (i, history) in history_tests.iter().enumerate() {
//         {
//             let mut counter = 0;
//             let mut generate = session.generate(&history)?;
//
//             while let Some(chunk) = generate.next_chunk()? {
//                 print!("{chunk}");
//                 stdout().flush().unwrap();
//
//                 if counter == i {
//                     break;
//                 }
//
//                 counter += 1;
//             }
//         }
//
//         let cached_tokens_len = session.cached_tokens().len();
//         let kv_cache_len = session.context();
//
//         assert_eq!(cached_tokens_len, kv_cache_len);
//
//         print!("\n\n");
//     }
//
//     Ok(())
// }