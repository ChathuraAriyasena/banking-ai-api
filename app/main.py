"""
FastAPI entrypoint.

POST /predict  -> runs the full pipeline: model predictions + KB retrieval + LLM guidance
GET  /health   -> simple readiness check
"""

import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app import pipeline

app = FastAPI(title="Banking AI Copilot API", version="1.0")

# Secret value that must be sent in the x-api-key header on every /predict
# call. Set this in Render's Environment Variables. If it's not set at all,
# the check is skipped (useful for local testing) -- but always set it in
# production so the endpoint isn't wide open.
API_KEY = os.environ.get("API_KEY")


class TicketRequest(BaseModel):
    ticket_text: str = Field(..., description="The raw customer support ticket text")
    channel: str = Field("mobile_banking", description="e.g. mobile_banking, internet_banking, chatbot, call_center, branch, email, whatsapp")
    customer_segment: str = Field("retail_plus")
    subject: str | None = Field(None)
    timestamp: str | None = Field(None, description="YYYY-MM-DD HH:MM:SS, defaults to now")


@app.on_event("startup")
def startup_event():
    pipeline.load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline.session is not None}


@app.post("/predict")
def predict(req: TicketRequest, x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")

    if not req.ticket_text.strip():
        raise HTTPException(status_code=400, detail="ticket_text must not be empty")

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

    return result
