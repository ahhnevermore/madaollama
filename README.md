Serves 4B models. Should work with any model really on llama cpp, but ive done it with Qwen3-4B-abliterated_dark.i1-Q4_K_M.gguf
AI helped quite a lot with understanding llama cpps concepts, and various Rust things, although the code was often buggy




```
cd madaollama-server
cargo run -p madaollama-server --release -- <Your model name>
```

You dont need to test with curl. Run madaollama-test
```
cargo run -p madaollama-test --release
```
