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

    # 1. Get customer
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

    # 2. Calculate churn
    churn_result = predict_churn(customer_data)

    # 3. Create retention request immediately
    with engine.begin() as connection:
        request_id = connection.execute(
            text("""
                INSERT INTO retention_requests
                (
                    customer_id,
                    action_type,
                    description,
                    currency,
                    reason,
                    status,
                    segment,
                    risk,
                    created_at
                )
                VALUES
                (
                    :customer_id,
                    :action_type,
                    :description,
                    :currency,
                    :reason,
                    :status,
                    :segment,
                    :risk,
                    CURRENT_TIMESTAMP
                )
                RETURNING id
            """),
            {
                "customer_id": payload.customer_id,
                "action_type": "AI_ANALYSIS",
                "description": payload.complaint,
                "currency": "EUR",
                "reason": "Customer complaint",
                "status": "PENDING",
                "segment": "CUSTOMER",
                "risk": churn_result["risk"]
            }
        ).scalar_one()

    # 4. Try Foundry
    try:
        result = analyze_customer(
            customer=customer_data,
            churn=churn_result,
            complaint=payload.complaint
        )

        # 5. Foundry succeeded
        with engine.begin() as connection:
            connection.execute(
                text("""
                    UPDATE retention_requests
                    SET status = 'COMPLETED',
                        description = :description
                    WHERE id = :id
                """),
                {
                    "id": request_id,
                    "description": f"{payload.complaint}\n\nAI Analysis: {result}"
                }
            )

        return {
            "customer_id": payload.customer_id,
            "request_id": request_id,
            "churn": churn_result,
            "analysis": result,
            "status": "COMPLETED"
        }

    except Exception as e:

        # 6. Foundry failed, but request remains stored
        with engine.begin() as connection:
            connection.execute(
                text("""
                    UPDATE retention_requests
                    SET status = 'FAILED'
                    WHERE id = :id
                """),
                {"id": request_id}
            )

        return {
            "customer_id": payload.customer_id,
            "request_id": request_id,
            "churn": churn_result,
            "analysis": None,
            "status": "FAILED",
            "error": str(e)
        }