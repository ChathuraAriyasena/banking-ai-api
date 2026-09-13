# Banking AI Copilot API

FastAPI service wrapping a multi-task DistilBERT ticket classifier + KB retrieval (RAG) + LLM
guidance generation, for banking support ticket triage.

## Architecture

- **Code** (this repo) -> deployed on **Render.com** (free web service, Docker-based)
- **Model artifacts** (too large for a normal git repo) -> stored in a **Hugging Face model repo**
  (separate from Spaces, unaffected by Spaces pricing changes) and downloaded automatically
  the first time the app starts up.

## Artifacts required

Upload these 4 files to a Hugging Face **model repo** (not a Space):

- `multitask_distilbert_clean.pt` - trained model checkpoint
- `label_encoders.pkl` - label encoders
- `kb_policies_rag.csv` - knowledge base used for RAG retrieval
- `kb_embeddings.pt` - precomputed KB embeddings

## Environment variables / secrets required (set these in Render)

- `OPENAI_API_KEY` - your OpenAI key
- `HF_REPO_ID` - e.g. `yourusername/banking-ai-artifacts` (the model repo from above)

## Endpoints

- `GET /health` - readiness check
- `POST /predict` - runs the full pipeline

### Example request body

```json
{
  "ticket_text": "My international transfer has been stuck for 3 days and I need it urgently.",
  "channel": "mobile_banking",
  "customer_segment": "retail_plus",
  "subject": "Delayed transfer",
  "timestamp": "2026-09-14 10:30:00"
}
```

## Local testing

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
export HF_REPO_ID=yourusername/banking-ai-artifacts
uvicorn app.main:app --reload
```

Then POST to `http://127.0.0.1:8000/predict`.
