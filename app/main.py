"""
FastAPI entrypoint.

POST /predict  -> runs the full pipeline: model predictions + KB retrieval + LLM guidance
GET  /health   -> simple readiness check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import pipeline

app = FastAPI(title="Banking AI Copilot API", version="1.0")


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
    return {"status": "ok", "model_loaded": pipeline.model is not None}


@app.post("/predict")
def predict(req: TicketRequest):
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
