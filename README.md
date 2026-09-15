# Banking AI Copilot API

FastAPI service wrapping a multi-task DistilBERT ticket classifier + KB retrieval (RAG) + LLM
guidance generation, for banking support ticket triage. Served via ONNX Runtime (not raw PyTorch)
for lightweight, memory-efficient inference on free-tier cloud hosting. Requests and responses
are both formally schema-enforced, not just loosely shaped JSON.

## Architecture

- **Code** (this repo) -> containerized with **Docker** -> deployed on **Render.com** (free web service)
- **Model artifacts** (too large for a normal git repo) -> stored in a **Hugging Face model repo**
  (separate from Spaces, unaffected by Spaces pricing changes) and downloaded automatically
  the first time the app starts up, then cached locally inside the container
- **Inference engine**: ONNX Runtime, not PyTorch. The trained PyTorch model is exported once,
  offline, to ONNX format, so the running service never needs to import `torch` or `transformers`'
  model backends at all, only a lightweight tokenizer
- **Schemas**: request and response are both formally defined with Pydantic (`app/schemas.py`).
  FastAPI validates every incoming request against `TicketRequest` and every outgoing response
  against `PredictResponse`, so a malformed request is rejected clearly, and a bug that would
  accidentally return the wrong shape is caught as a clear server error instead of silently
  sending broken JSON to the caller
- **Secrets**: `OPENAI_API_KEY`, `HF_TOKEN`, and `API_KEY` are all read from environment variables
  at runtime; nothing sensitive is hardcoded in source

### Request flow

```
Ticket text (validated: known channel/segment, valid timestamp, length limits)
   |
   v
Tokenizer (Hugging Face, fast tokenizer, no torch backend)
   |
   v
ONNX Runtime session (single forward pass)
   |
   +--> 6 classification heads: intent, issue_type, product, urgency, sentiment, routing_queue
   |
   +--> pooled embedding (shared CLS output, reused for retrieval below)
   |
   v
KB retrieval: cosine similarity search over precomputed KB embeddings (numpy),
              reranked using the model's own predicted labels
   |
   v
Prompt assembly: ticket (wrapped as untrusted data, with a prompt-injection guard)
                 + predictions + top KB snippets
   |
   v
LLM call (OpenAI) -> structured JSON guidance, schema-validated
   |
   v
Combined, schema-enforced JSON response, with full request metadata attached
```

## Engineering notes: why ONNX instead of PyTorch

The original serving stack (PyTorch + Transformers, loading the `.pt` checkpoint directly)
consistently exceeded Render's free-tier 512MB memory limit during startup, even after:
- Building the backbone from config instead of downloading pretrained weights a second time
- Freeing the raw checkpoint dict from memory immediately after `load_state_dict()`
- Dynamic INT8 quantization of the model's linear layers

None of that was enough, since PyTorch + Transformers alone typically use 250-350MB just from
being imported, before any model weights are loaded. The fix was to remove PyTorch from the
serving path entirely: the model is exported once (offline, in the training environment) to
ONNX, and the API only ever loads `onnxruntime`, a much lighter runtime with no PyTorch
dependency, no CUDA libraries, and a fraction of the baseline memory footprint.

## Request validation (production-grade input hardening)

- `channel` is restricted to the exact 7 values the model was trained on (an enum, not free text)
- `customer_segment` is restricted to the 7 confirmed valid segment codes
- `timestamp`, if provided, must be a genuinely parseable date; a malformed value is rejected
  outright rather than silently substituted with the current time
- `ticket_text` and `subject` have enforced max lengths (4000 / 200 characters)
- Whitespace-only `ticket_text` is rejected, not just empty strings
- The ticket text is wrapped in explicit delimiters inside the LLM prompt, with an instruction
  telling the model to treat it strictly as data, never as commands to follow, as a first line
  of defense against prompt injection via ticket content

## Response structure (enforced, not just assembled)

Every response is validated against a formal schema before being sent, in a fixed field order:

- `meta` - full request traceability: a fresh `request_id` (GUID, always server-generated), an
  echoed or auto-generated `correlation_id`, an optional caller-supplied `client_id`, the
  `environment`, `api_version`, `model_version`, `kb_version`, which `llm_model` answered,
  the request's received timestamp, and processing time in milliseconds
- `interaction_id`, `timestamp`, `mode` - generated once by this service, appear exactly once
- `ticket_context` - the ticket and its derived context as clean structured fields (channel,
  segment, day of week, time bucket, business hours, weekend flag), not a single string with
  tags jammed into it
- `model_predictions` - all 6 classifier outputs, each required
- `kb_hits` - top matching knowledge base articles, including which team (`routing_queue`) owns
  that policy area
- `llm_response` - the LLM's structured guidance (summary, actions, clarifications, risk notes,
  cited KB policies), or `null` if the LLM's reply failed validation
- `llm_raw_output` - the LLM's raw text, populated only when `llm_response` is `null`, so nothing
  is silently lost
- `validation` - whether the LLM's reply passed schema validation, and why not if it didn't

## Artifacts required

Upload these 5 files to a Hugging Face **model repo** (not a Space):

- `model.onnx` - exported model graph (all 6 classification heads + pooled embedding, in one graph)
- `model.onnx.data` - external weights file (PyTorch's ONNX exporter splits large models into two files; both are required together)
- `label_encoders.pkl` - label encoders, used to turn predicted class indices back into readable labels
- `kb_policies_rag.csv` - knowledge base used for RAG retrieval
- `kb_embeddings.npy` - precomputed KB embeddings, saved as plain numpy (not a `.pt` file, so no torch is needed to read it)

## Environment variables / secrets required (set these in Render)

- `OPENAI_API_KEY` - your OpenAI key
- `HF_REPO_ID` - e.g. `yourusername/banking-ai-artifacts` (the model repo from above)
- `HF_TOKEN` - required if the Hugging Face repo above is Private
- `API_KEY` - secret value required in the `x-api-key` header on every `/v1/predict` call
- `MODEL_VERSION`, `KB_VERSION`, `ENVIRONMENT` - optional, shown in every response's `meta` block

## Endpoints

- `GET /health` - readiness check, returns whether the model session has loaded (kept unversioned;
  monitoring tools expect this at a stable path regardless of API version)
- `POST /v1/predict` - runs the full pipeline: classification, KB retrieval, LLM guidance. Requires
  an `x-api-key` header matching `API_KEY`. Optional headers: `x-correlation-id` (echoed back if
  sent, generated if not), `x-client-id` (identifies the calling system)

### Example request

```
POST /v1/predict
x-api-key: your-secret-key
x-correlation-id: {{$guid}}
Content-Type: application/json
```

```json
{
  "ticket_text": "My international transfer has been stuck for 3 days and I need it urgently.",
  "channel": "mobile_banking",
  "customer_segment": "vip",
  "subject": "Delayed transfer",
  "timestamp": "2026-09-14 10:30:00"
}
```

### Sample response

```json
{
  "meta": {
    "request_id": "9f3a7c2e-4b1d-4a6e-8f2a-1c3d5e7f9a0b",
    "correlation_id": "test-001",
    "client_id": null,
    "environment": "production",
    "api_version": "v1",
    "model_version": "1.0.0",
    "kb_version": "1.0.0",
    "llm_model": "gpt-5-mini-2025-08-07",
    "request_received_at": "2026-09-14T14:30:00.123456+00:00",
    "processing_time_ms": 842.31
  },
  "interaction_id": "14092614300012345",
  "timestamp": "14/09/26 14:30:00",
  "mode": "HIGH_CONFIDENCE",
  "ticket_context": {
    "channel": "mobile_banking",
    "customer_segment": "vip",
    "subject": "Delayed transfer",
    "day_of_week": "Monday",
    "time_bucket": "morning",
    "business_hours": true,
    "weekend": false,
    "ticket_text": "My international transfer has been stuck for 3 days and I need it urgently."
  },
  "model_predictions": {
    "intent": { "label": "transaction_status_inquiry", "confidence": 0.9421, "tag": "OK" },
    "issue_type": { "label": "delayed_transfer", "confidence": 0.9187, "tag": "OK" },
    "product": { "label": "international_transfer", "confidence": 0.8952, "tag": "OK" },
    "urgency": { "label": "high", "confidence": 0.8734, "tag": "OK" },
    "sentiment": { "label": "frustrated", "confidence": 0.8103, "tag": "OK" },
    "routing_queue": { "label": "payments_operations", "confidence": 0.9016, "tag": "OK" }
  },
  "kb_hits": [
    { "kb_id": "KB-0142", "title": "International Wire Transfer SLA and Delay Escalation", "routing_queue": "payments_operations", "similarity": 0.812, "final_score": 0.962 },
    { "kb_id": "KB-0087", "title": "Tracing a Pending SWIFT Payment", "routing_queue": "payments_operations", "similarity": 0.771, "final_score": 0.871 }
  ],
  "llm_response": {
    "summary": "Customer reports a delayed international transfer, 3 days pending, requesting urgent resolution.",
    "actions": [
      "Trace the SWIFT payment using the reference number and confirm current status with the correspondent bank",
      "Escalate to Payments Operations given the 3-day delay exceeds standard SLA"
    ],
    "clarifications": [
      "Can you confirm the transaction reference or approximate transfer date?"
    ],
    "risk_notes": [
      "Delay exceeds standard SLA; escalation required per policy"
    ],
    "kb_policies_used": [
      { "kb_id": "KB-0142", "title": "International Wire Transfer SLA and Delay Escalation" }
    ]
  },
  "llm_raw_output": null,
  "validation": { "is_valid": true, "error": null }
}
```

If any of the 6 classifier predictions falls below the confidence threshold, `mode` switches to
`"REVIEW_REQUIRED"` and the LLM is instructed to lean on clarifying questions rather than
confident actions, so low-confidence cases are never silently auto-resolved.

## Local testing

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
export HF_REPO_ID=yourusername/banking-ai-artifacts
export HF_TOKEN=hf_...
export API_KEY=your-secret-key
uvicorn app.main:app --reload
```

Then POST to `http://127.0.0.1:8000/v1/predict` with the `x-api-key` header set.

## Tech stack

Python | FastAPI | Pydantic | Docker | ONNX Runtime | Hugging Face Hub | Render | RAG | LLM Orchestration | NLP | REST API Design | API Authentication | API Versioning | Request/Response Schema Validation | Postman | MLOps
