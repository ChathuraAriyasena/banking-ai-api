# Banking AI Copilot API

FastAPI service wrapping a multi-task DistilBERT ticket classifier + KB retrieval (RAG) + LLM
guidance generation, for banking support ticket triage. Served via ONNX Runtime (not raw PyTorch)
for lightweight, memory-efficient inference on free-tier cloud hosting.

## Architecture

- **Code** (this repo) -> containerized with **Docker** -> deployed on **Render.com** (free web service)
- **Model artifacts** (too large for a normal git repo) -> stored in a **Hugging Face model repo**
  (separate from Spaces, unaffected by Spaces pricing changes) and downloaded automatically
  the first time the app starts up, then cached locally inside the container
- **Inference engine**: ONNX Runtime, not PyTorch. The trained PyTorch model is exported once,
  offline, to ONNX format, so the running service never needs to import `torch` or `transformers`'
  model backends at all, only a lightweight tokenizer
- **Secrets**: `OPENAI_API_KEY`, `HF_TOKEN`, and `API_KEY` are all read from environment variables
  at runtime; nothing sensitive is hardcoded in source

### Request flow

```
Ticket text
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
Prompt assembly: ticket + predictions + top KB snippets
   |
   v
LLM call (OpenAI) -> structured JSON guidance, schema-validated
   |
   v
Combined JSON response
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
dependency, no CUDA libraries, and a fraction of the baseline memory footprint. This is the
difference between a "works on my machine" prototype and something that survives a genuinely
constrained production environment.

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
- `API_KEY` - secret value required in the `x-api-key` header on every `/predict` call

## Endpoints

- `GET /health` - readiness check, returns whether the model session has loaded
- `POST /predict` - runs the full pipeline: classification, KB retrieval, LLM guidance. Requires an `x-api-key` header matching `API_KEY`

### Example request

```
POST /predict
x-api-key: your-secret-key
Content-Type: application/json
```

```json
{
  "ticket_text": "My international transfer has been stuck for 3 days and I need it urgently.",
  "channel": "mobile_banking",
  "customer_segment": "retail_plus",
  "subject": "Delayed transfer",
  "timestamp": "2026-09-14 10:30:00"
}
```

### Sample response

```json
{
  "input_text": "[CHANNEL=mobile_banking] [SEGMENT=retail_plus] [DOW=Monday] [TIME_BUCKET=morning] [BUSINESS_HOURS=yes] [WEEKEND=no] [SUBJECT=Delayed transfer] My international transfer has been stuck for 3 days and I need it urgently.",
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
    { "kb_id": "KB-0087", "title": "Tracing a Pending SWIFT Payment", "routing_queue": "payments_operations", "similarity": 0.771, "final_score": 0.871 },
    { "kb_id": "KB-0203", "title": "Customer Communication Templates for Payment Delays", "routing_queue": "customer_service", "similarity": 0.664, "final_score": 0.664 }
  ],
  "llm_response": {
    "interaction_id": "14092614300012345",
    "timestamp": "14/09/26 14:30:00",
    "mode": "HIGH_CONFIDENCE",
    "summary": "Customer reports a delayed international transfer, 3 days pending, requesting urgent resolution.",
    "actions": [
      "Trace the SWIFT payment using the reference number and confirm current status with the correspondent bank",
      "Escalate to Payments Operations given the 3-day delay exceeds standard SLA",
      "Provide the customer an estimated resolution window per KB-0142"
    ],
    "clarifications": [
      "Can you confirm the transaction reference or approximate transfer date?"
    ],
    "risk_notes": [
      "Delay exceeds standard SLA; escalation required per policy"
    ],
    "kb_policies_used": [
      { "kb_id": "KB-0142", "title": "International Wire Transfer SLA and Delay Escalation" },
      { "kb_id": "KB-0087", "title": "Tracing a Pending SWIFT Payment" }
    ]
  },
  "llm_raw_output": null,
  "validation": { "is_valid": true, "error": null },
  "interaction_id": "14092614300012345",
  "timestamp": "14/09/26 14:30:00",
  "mode": "HIGH_CONFIDENCE"
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

Then POST to `http://127.0.0.1:8000/predict` with the `x-api-key` header set.

## Tech stack

Python | FastAPI | Docker | ONNX Runtime | Hugging Face Hub | Render | RAG | LLM Orchestration | NLP | REST API Design | API Authentication | Postman | MLOps
