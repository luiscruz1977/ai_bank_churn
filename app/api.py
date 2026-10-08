from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text

from app.churn import predict_churn
from app.retention import trigger_retention_action

from pydantic import BaseModel
from app.foundry_agent import analyze_customer

DATABASE_URL = "sqlite:///./data/bank.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

app = FastAPI(
    title="Bank Churn AI API",
    version="1.0.0"
)


@app.get("/customers/{customer_id}")
def get_customer(customer_id: str):

    with engine.connect() as connection:

        customer = connection.execute(
            text("""
                SELECT *
                FROM customers
                WHERE customer_id = :customer_id
            """),
            {"customer_id": customer_id}
        ).mappings().first()

        if not customer:
            raise HTTPException(
                status_code=404,
                detail="Customer not found"
            )

        complaints = connection.execute(
            text("""
                SELECT *
                FROM complaints
                WHERE customer_id = :customer_id
            """),
            {"customer_id": customer_id}
        ).mappings().all()

        customer_data = dict(customer)

        churn_result = predict_churn(customer_data)

        return {
            "customer": customer_data,
            "churn": churn_result,
            "complaints": [dict(c) for c in complaints]
        }

@app.post("/retention/action")
def retention_action(payload: dict):

    result = trigger_retention_action(
        customer_id=payload["customer_id"],
        action_type=payload["action_type"],
        details=payload.get("details", {})
    )

    return result

class AnalyzeRequest(BaseModel):
    customer_id: str
    complaint: str


@app.post("/analyze")
def analyze_customer_request(payload: AnalyzeRequest):

    with engine.connect() as connection:
        customer = connection.execute(
            text("""
                SELECT *
                FROM customers
                WHERE customer_id = :customer_id
            """),
            {"customer_id": payload.customer_id}
        ).mappings().first()

        if not customer:
            raise HTTPException(
                status_code=404,
                detail="Customer not found"
            )

    customer_data = dict(customer)

    churn_result = predict_churn(customer_data)

    result = analyze_customer(
        customer=customer_data,
        churn=churn_result,
        complaint=payload.complaint
    )

    return {
        "customer_id": payload.customer_id,
        "churn": churn_result,
        "analysis": result
    }