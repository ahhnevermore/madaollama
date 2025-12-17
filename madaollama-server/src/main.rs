use std::{path::Path, sync::Arc};

use axum::{Router, routing::post};

use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 2);
    let model_name = &args[1];

    let dev_path = Path::new("models").join(model_name);

    let load_path = if dev_path.exists() {
        dev_path.to_path_buf()
    } else {
        // Otherwise, load relative to the binary
        let exe_dir = std::env::current_exe()
            .expect("cant find current exe")
            .parent()
            .expect("no valid parent directory for exe")
            .to_owned();
        exe_dir.join("models").join(model_name)
    };

    let model = Arc::new(
        madaollama_server::load_model(load_path.to_str().expect("bad model path"))
            .expect("could not load model"),
    );

    let (tx, rx) = mpsc::channel::<madaollama_server::Request>(1024);

    // Spawn inference worker
    tokio::task::spawn_blocking(move || {
        madaollama_server::inference_worker(rx, model);
    });

    let app = Router::new()
        .route("/generate", post(madaollama_server::generate))
        .with_state(tx);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();

    axum::serve(listener, app).await.unwrap();
}
