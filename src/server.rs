use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Json,
    },
    routing::{get, post},
    Router,
};
use serde_json::json;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::api_types::*;
use crate::engine;
use crate::memory::ExpertMemoryManager;
use crate::model::Model;
use crate::tokenizer::QwenTokenizer;

// ---- Inference actor types ----

pub enum PromptInput {
    Messages(Vec<(String, String)>), // (role, content)
    Text(String),
}

pub struct InferenceRequest {
    pub input: PromptInput,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub event_tx: mpsc::Sender<InferenceEvent>,
}

pub enum InferenceEvent {
    Token(String),
    Done { output: String, prompt_tokens: u32, completion_tokens: u32 },
    Error(String),
}

// InferenceEvent must be Send so the tokio channel works across threads
unsafe impl Send for InferenceEvent {}

// ---- Server state ----

#[derive(Clone)]
struct AppState {
    req_tx: std::sync::Arc<std::sync::Mutex<std::sync::mpsc::Sender<InferenceRequest>>>,
    model_id: Arc<String>,
    created_at: u64,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn new_id(prefix: &str) -> String {
    format!("{}-{}", prefix, Uuid::new_v4().simple())
}

// ---- Inference thread ----

pub fn start_inference_thread(
    mut model: Model,
    tokenizer: QwenTokenizer,
    mem_mgr: ExpertMemoryManager,
    kv_quant_bits: Option<u8>,
    rep_penalty: f32,
) -> std::sync::mpsc::Sender<InferenceRequest> {
    let (req_tx, req_rx) = std::sync::mpsc::channel::<InferenceRequest>();

    std::thread::spawn(move || {
        while let Ok(req) = req_rx.recv() {
            let prompt = match req.input {
                PromptInput::Messages(pairs) => {
                    let msgs: Vec<crate::tokenizer::ChatMessage> = pairs
                        .into_iter()
                        .map(|(role, content)| crate::tokenizer::ChatMessage { role, content })
                        .collect();
                    match tokenizer.apply_chat_template(&msgs) {
                        Ok(p) => p,
                        Err(e) => {
                            req.event_tx.blocking_send(InferenceEvent::Error(e.to_string())).ok();
                            continue;
                        }
                    }
                }
                PromptInput::Text(p) => p,
            };

            let prompt_tokens = tokenizer.encode(&prompt).map(|ids| ids.len() as u32).unwrap_or(0);

            let event_tx = req.event_tx.clone();
            let mut on_token = |text: &str| {
                event_tx.blocking_send(InferenceEvent::Token(text.to_string())).ok();
            };

            match engine::generate(
                &mut model,
                &tokenizer,
                &prompt,
                req.max_tokens,
                req.temperature,
                req.top_p,
                rep_penalty,
                &mem_mgr,
                kv_quant_bits,
                false,
                None,
                None,
                false,
                false,
                Some(&mut on_token),
            ) {
                Ok((output, _)) => {
                    let completion_tokens = tokenizer.encode(&output).map(|ids| ids.len() as u32).unwrap_or(0);
                    req.event_tx.blocking_send(InferenceEvent::Done {
                        output,
                        prompt_tokens,
                        completion_tokens,
                    }).ok();
                }
                Err(e) => {
                    req.event_tx.blocking_send(InferenceEvent::Error(e.to_string())).ok();
                }
            }
        }
    });

    req_tx
}

// ---- Axum server ----

pub async fn run(
    host: &str,
    port: u16,
    model: Model,
    tokenizer: QwenTokenizer,
    mem_mgr: ExpertMemoryManager,
    model_id: String,
    kv_quant_bits: Option<u8>,
    rep_penalty: f32,
) -> anyhow::Result<()> {
    let req_tx = start_inference_thread(model, tokenizer, mem_mgr, kv_quant_bits, rep_penalty);

    let state = AppState {
        req_tx: Arc::new(std::sync::Mutex::new(req_tx)),
        model_id: Arc::new(model_id),
        created_at: now_secs(),
    };

    let cors = tower_http::cors::CorsLayer::permissive();

    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .layer(cors)
        .with_state(state);

    let addr = format!("{}:{}", host, port);
    eprintln!("Listening on http://{}", addr);
    eprintln!("OpenAI-compatible base URL: http://{}/v1", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// ---- Handlers ----

async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelObject {
            id: (*state.model_id).clone(),
            object: "model",
            created: state.created_at,
            owned_by: "local",
        }],
    })
}

fn send_request(state: &AppState, req: InferenceRequest) -> Result<(), StatusCode> {
    state
        .req_tx
        .lock()
        .unwrap()
        .send(req)
        .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let id = new_id("chatcmpl");
    let created = now_secs();
    let model_id = (*state.model_id).clone();

    let (event_tx, mut event_rx) = mpsc::channel::<InferenceEvent>(512);

    let infer_req = InferenceRequest {
        input: PromptInput::Messages(
            req.messages
                .iter()
                .map(|m| (m.role.clone(), m.content.clone()))
                .collect(),
        ),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        event_tx,
    };

    if let Err(code) = send_request(&state, infer_req) {
        return (code, Json(json!({"error": "inference thread unavailable"}))).into_response();
    }

    if req.stream {
        let id2 = id.clone();
        let model_id2 = model_id.clone();

        let sse_stream = async_stream::stream! {
            let role_chunk = ChatCompletionChunk {
                id: id2.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_id2.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: DeltaContent { role: Some("assistant"), content: None },
                    finish_reason: None,
                }],
            };
            yield Ok::<Event, Infallible>(Event::default().data(
                serde_json::to_string(&role_chunk).unwrap_or_default()
            ));

            while let Some(event) = event_rx.recv().await {
                match event {
                    InferenceEvent::Token(text) => {
                        let chunk = ChatCompletionChunk {
                            id: id2.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model_id2.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: DeltaContent { role: None, content: Some(text) },
                                finish_reason: None,
                            }],
                        };
                        yield Ok(Event::default().data(
                            serde_json::to_string(&chunk).unwrap_or_default()
                        ));
                    }
                    InferenceEvent::Done { .. } => {
                        let finish = ChatCompletionChunk {
                            id: id2.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model_id2.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: DeltaContent { role: None, content: None },
                                finish_reason: Some("stop"),
                            }],
                        };
                        yield Ok(Event::default().data(
                            serde_json::to_string(&finish).unwrap_or_default()
                        ));
                        yield Ok(Event::default().data("[DONE]"));
                        break;
                    }
                    InferenceEvent::Error(e) => {
                        eprintln!("inference error: {}", e);
                        yield Ok(Event::default().data("[DONE]"));
                        break;
                    }
                }
            }
        };

        Sse::new(sse_stream).into_response()
    } else {
        let mut full_text = String::new();
        let mut prompt_tokens = 0u32;
        let mut completion_tokens = 0u32;
        let mut err: Option<String> = None;
        loop {
            match event_rx.recv().await {
                Some(InferenceEvent::Token(t)) => full_text.push_str(&t),
                Some(InferenceEvent::Done { output, prompt_tokens: pt, completion_tokens: ct }) => {
                    full_text = output;
                    prompt_tokens = pt;
                    completion_tokens = ct;
                    break;
                }
                None => break,
                Some(InferenceEvent::Error(e)) => {
                    err = Some(e);
                    break;
                }
            }
        }

        if let Some(e) = err {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": {"message": e, "type": "server_error"}})),
            )
                .into_response();
        }

        Json(ChatCompletionResponse {
            id,
            object: "chat.completion",
            created,
            model: model_id,
            choices: vec![ChatChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: full_text,
                },
                finish_reason: "stop",
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        })
        .into_response()
    }
}

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    let id = new_id("cmpl");
    let created = now_secs();
    let model_id = (*state.model_id).clone();

    let (event_tx, mut event_rx) = mpsc::channel::<InferenceEvent>(512);

    let infer_req = InferenceRequest {
        input: PromptInput::Text(req.prompt.clone()),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        event_tx,
    };

    if let Err(code) = send_request(&state, infer_req) {
        return (code, Json(json!({"error": "inference thread unavailable"}))).into_response();
    }

    if req.stream {
        let id2 = id.clone();
        let model_id2 = model_id.clone();

        let sse_stream = async_stream::stream! {
            while let Some(event) = event_rx.recv().await {
                match event {
                    InferenceEvent::Token(text) => {
                        let chunk = CompletionChunk {
                            id: id2.clone(),
                            object: "text_completion",
                            created,
                            model: model_id2.clone(),
                            choices: vec![CompletionChunkChoice {
                                text,
                                index: 0,
                                finish_reason: None,
                            }],
                        };
                        yield Ok::<Event, Infallible>(Event::default().data(
                            serde_json::to_string(&chunk).unwrap_or_default()
                        ));
                    }
                    InferenceEvent::Done { .. } => {
                        let finish = CompletionChunk {
                            id: id2.clone(),
                            object: "text_completion",
                            created,
                            model: model_id2.clone(),
                            choices: vec![CompletionChunkChoice {
                                text: String::new(),
                                index: 0,
                                finish_reason: Some("stop"),
                            }],
                        };
                        yield Ok(Event::default().data(
                            serde_json::to_string(&finish).unwrap_or_default()
                        ));
                        yield Ok(Event::default().data("[DONE]"));
                        break;
                    }
                    InferenceEvent::Error(e) => {
                        eprintln!("inference error: {}", e);
                        yield Ok(Event::default().data("[DONE]"));
                        break;
                    }
                }
            }
        };

        Sse::new(sse_stream).into_response()
    } else {
        let mut full_text = String::new();
        let mut prompt_tokens = 0u32;
        let mut completion_tokens = 0u32;
        let mut err: Option<String> = None;
        loop {
            match event_rx.recv().await {
                Some(InferenceEvent::Token(t)) => full_text.push_str(&t),
                Some(InferenceEvent::Done { output, prompt_tokens: pt, completion_tokens: ct }) => {
                    full_text = output;
                    prompt_tokens = pt;
                    completion_tokens = ct;
                    break;
                }
                None => break,
                Some(InferenceEvent::Error(e)) => {
                    err = Some(e);
                    break;
                }
            }
        }

        if let Some(e) = err {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": {"message": e, "type": "server_error"}})),
            )
                .into_response();
        }

        Json(CompletionResponse {
            id,
            object: "text_completion",
            created,
            model: model_id,
            choices: vec![CompletionChoice {
                text: full_text,
                index: 0,
                finish_reason: "stop",
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        })
        .into_response()
    }
}
