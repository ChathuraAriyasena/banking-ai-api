"""
FastAPI entrypoint.

POST /v1/predict  -> runs the full pipeline: model predictions + KB retrieval + LLM guidance
GET  /health       -> simple readiness check (kept unversioned -- load balancers / uptime
                      monitors expect this at a stable path regardless of API version)
"""

import os
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException

from app import pipeline
from app.schemas import PredictResponse, TicketRequest

app = FastAPI(title="Banking AI Copilot API", version=pipeline.API_VERSION)

# Secret value that must be sent in the x-api-key header on every /predict
# call. Set this in Render's Environment Variables. If it's not set at all,
# the check is skipped (useful for local testing) -- but always set it in
# production so the endpoint isn't wide open.
API_KEY = os.environ.get("API_KEY")


@app.on_event("startup")
def startup_event():
    pipeline.load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline.session is not None}


@app.post("/v1/predict", response_model=PredictResponse)
def predict(
    req: TicketRequest,
    x_api_key: str | None = Header(default=None),
    x_correlation_id: str | None = Header(default=None),
    x_client_id: str | None = Header(default=None),
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")

    request_received_at = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()

    # If the caller didn't supply their own correlation ID, generate one --
    # every response always has one, either way, so nothing is ever untraceable.
    correlation_id = x_correlation_id or str(uuid.uuid4())

    try:
        result = pipeline.run_pipeline(
            channel=req.channel,
            customer_segment=req.customer_segment,
            subject=req.subject,
            timestamp=req.timestamp,
            ticket_text=req.ticket_text,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    result["meta"] = {
        "correlation_id": correlation_id,
        "client_id": x_client_id,
        "environment": pipeline.ENVIRONMENT,
        "api_version": pipeline.API_VERSION,
        "model_version": pipeline.MODEL_VERSION,
        "kb_version": pipeline.KB_VERSION,
        "llm_model": pipeline.OPENAI_MODEL,
        "request_received_at": request_received_at,
        "processing_time_ms": processing_time_ms,
    }

    return result
