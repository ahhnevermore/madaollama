use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Serialize)]
struct HTTPRequest {
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
}

#[derive(Deserialize, Debug)]
struct Response {
    text: String,
    gen_tokens: usize,
}

#[tokio::main]
async fn main() {
    let client = Client::new();
    let url = "http://127.0.0.1:3000/generate";

    let num_requests = 50;
    let mut handles = Vec::new();

    let start = Instant::now();

    for i in 0..num_requests {
        let client = client.clone();
        let prompt = format!("Request {}: Hello world", i);

        let req = HTTPRequest {
            prompt,
            max_tokens: Some(128),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
        };

        let handle = tokio::spawn(async move {
            let t0 = Instant::now();

            let resp = client
                .post(url)
                .json(&req)
                .send()
                .await
                .expect("request failed")
                .json::<Response>()
                .await
                .expect("invalid response");

            let elapsed = t0.elapsed().as_millis();

            (elapsed, resp)
        });

        handles.push(handle);
    }

    for h in handles {
        let (latency_ms, resp) = h.await.unwrap();
        println!(
            "[{} ms] tokens={} text={:?}",
            latency_ms,
            resp.gen_tokens,
            resp.text.chars().take(80).collect::<String>()
        );
    }

    println!(
        "Total wall time: {} ms",
        start.elapsed().as_millis()
    );
}
