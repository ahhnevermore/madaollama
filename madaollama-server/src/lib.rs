use axum::http::StatusCode;
use llama_cpp_2::{
    context::{LlamaContext, params::LlamaContextParams},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, Special, params::LlamaModelParams},
    sampling::LlamaSampler,
};
use serde::{Deserialize, Serialize};
use std::{num::NonZeroU32, sync::Arc};

use axum::{Json, extract::State};
use tokio::sync::mpsc;
use tokio::sync::oneshot;

#[derive(Debug)]
pub struct Request {
    pub prompt: String,
    pub response: oneshot::Sender<Response>,
    pub max_tokens: usize,
    pub params: Sampling,
}

#[derive(Debug)]
pub struct Sampling {
    pub temp: f32,
    pub top_p: f32,
    pub top_k: i32,
}

#[derive(Deserialize)]
pub struct HTTPRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
}

#[derive(Debug, Serialize)]
pub struct Response {
    pub text: String,
    pub gen_tokens: usize,
}

struct RequestState {
    req_id: usize,
    seq_id: i32,
    pos: i32,
    sampler: LlamaSampler,
    finished: bool,
    output: String,
}

pub struct Worker {
    model: Arc<Model>,
}

pub async fn generate(
    State(tx): State<mpsc::Sender<Request>>,
    Json(req): Json<HTTPRequest>,
) -> Result<Json<Response>, StatusCode> {
    let (reply_tx, reply_rx) = oneshot::channel();

    let min_token_len = req.prompt.len() + 1;
    let msg = Request {
        prompt: req.prompt,
        response: reply_tx,
        max_tokens: req.max_tokens.unwrap_or(256).clamp(min_token_len, 512),
        params: Sampling {
            temp: req.temperature.unwrap_or(0.7).clamp(0.0, 1.5),
            top_p: req.top_p.unwrap_or(0.9).clamp(0.7, 1.0),
            top_k: req.top_k.unwrap_or(40).clamp(1, 100),
        },
    };

    tx.send(msg)
        .await
        .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?;

    let resp = reply_rx
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(resp))
}

pub fn inference_worker(mut rx: mpsc::Receiver<Request>, model: Arc<Model>) {
    while let Some(req) = rx.blocking_recv() {
        handle_request(&model, req);
    }
}

fn handle_request(model: &Model, req: Request) {
    let result = run_inference(model, &req);

    let response = match result {
        Ok((text, gen_tokens)) => Response { text, gen_tokens },
        Err(e) => {
            eprintln!("inference error: {e:?}");
            Response {
                text: "inference failed".into(),
                gen_tokens: 0,
            }
        }
    };

    // Ignore error if client disconnected
    let _ = req.response.send(response);
}

fn run_inference(model: &Model, req: &Request) -> anyhow::Result<(String, usize)> {
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(2048).unwrap()))
        .with_n_batch(256)
        .with_n_ubatch(128);

    let mut ctx = model.model.new_context(&model.backend, ctx_params)?;

    // Tokenize prompt
    let tokens = model.model.str_to_token(&req.prompt, AddBos::Always)?;

    let mut pos = 0;

    for (i, token) in tokens.iter().enumerate() {
        let mut batch = LlamaBatch::new(1, 1);
        let want_logits = i == tokens.len() - 1;
        batch.add(*token, pos, &[0], want_logits)?;
        ctx.decode(&mut batch)?;
        pos += 1;
    }

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::top_k(req.params.top_k),
        LlamaSampler::top_p(req.params.top_p, 1),
        LlamaSampler::temp(req.params.temp),
        LlamaSampler::dist(42),
    ]);

    let mut output = String::new();
    let mut gen_tokens = 0;

    for _ in 0..req.max_tokens {
        let token = sampler.sample(&ctx, 0);

        if model.model.is_eog_token(token) {
            break;
        }

        let mut batch = LlamaBatch::new(1, 1);
        batch.add(token, pos, &[0], true)?;
        ctx.decode(&mut batch)?;

        let piece = model.model.token_to_str(token, Special::Plaintext)?;
        output.push_str(&piece);

        pos += 1;
        gen_tokens += 1;
    }

    Ok((output, gen_tokens))
}

pub struct Model {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
}
impl Model {
    pub fn new_context(&self) -> anyhow::Result<LlamaContext<'_>> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(2048))
            .with_n_batch(256)
            .with_n_ubatch(128);
        Ok(self.model.new_context(&self.backend, ctx_params)?)
    }

    pub fn infer_once(&self, prompt: &str) -> anyhow::Result<String> {
        let mut ctx = self.new_context()?;
        // Tokenize prompt with BOS
        let tokens = self
            .model
            .str_to_token(prompt, llama_cpp_2::model::AddBos::Always)?;
        let mut pos = 0;
        let length: i32 = tokens.len().try_into().expect("length of tokens too big");

        // Feed prompt tokens
        for token in tokens {
            let mut batch = LlamaBatch::new(1, 1); // batch of 1 token
            let _ = batch.add(token, pos, &[0], pos == length - 1);
            ctx.decode(&mut batch)?;
            pos += 1;
        }

        let mut sampler = LlamaSampler::greedy();
        let mut output = String::new();

        // Generate up to max tokens
        for _ in 0..256 {
            // Sample next token
            let token = sampler.sample(&ctx, 0);

            // Stop on EOS
            if self.model.is_eog_token(token) {
                break;
            }

            // Decode the sampled token
            let mut batch = LlamaBatch::new(1, 1);
            let _ = batch.add(token, pos, &[0], true);
            ctx.decode(&mut batch)?;

            // Convert token → text
            let piece = self.model.token_to_str(token, Special::Plaintext)?;
            output.push_str(&piece);

            pos += 1;
        }

        Ok(output)
    }
}
pub fn load_model(path: &str) -> anyhow::Result<Model> {
    let backend = Arc::new(LlamaBackend::init()?);
    let model_params = LlamaModelParams::default().with_n_gpu_layers(999);
    let model = Arc::new(LlamaModel::load_from_file(&backend, path, &model_params)?);
    Ok(Model { backend, model })
}
