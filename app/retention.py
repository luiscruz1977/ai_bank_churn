from datetime import datetime

from sqlalchemy import create_engine, text

from app.churn import predict_churn


DATABASE_URL = "sqlite:///./data/bank.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)


VOUCHER_LIMITS = {
    "STANDARD": {
        "LOW": 0,
        "MEDIUM": 25,
        "HIGH": 50,
        "CRITICAL": 100
    },
    "AFFLUENT": {
        "LOW": 0,
        "MEDIUM": 50,
        "HIGH": 100,
        "CRITICAL": 200
    },
    "PRIVATE": {
        "LOW": 0,
        "MEDIUM": 100,
        "HIGH": 200,
        "CRITICAL": 400
    }
}


def calculate_segment(balance: float, num_products: int) -> str:
    if balance >= 50000 or num_products >= 4:
        return "PRIVATE"

    if balance >= 15000:
        return "AFFLUENT"

    return "STANDARD"


def trigger_retention_action(
    customer_id: str,
    action_type: str,
    details: dict
) -> dict:

    allowed_actions = {
        "voucher",
        "retention_call",
        "fee_waiver"
    }

    if action_type not in allowed_actions:
        return {
            "status": "rejected",
            "customer_id": customer_id,
            "reason": f"Action '{action_type}' is not allowed."
        }

    # Get customer directly from the database
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
        return {
            "status": "rejected",
            "customer_id": customer_id,
            "reason": "Customer not found."
        }

    customer_data = dict(customer)

    # Segment is calculated by application logic
    segment = calculate_segment(
        customer_data["balance"],
        customer_data["num_products"]
    )

    # Risk is calculated by the application ML model
    churn_result = predict_churn(customer_data)
    risk = churn_result["risk"]

    # Voucher-specific policy validation
    if action_type == "voucher":

        amount = float(details.get("amount", 0))

        max_voucher = VOUCHER_LIMITS[segment][risk]

        # Minimum tenure requirement
        if customer_data["tenure_months"] < 3:
            return {
                "status": "escalation_required",
                "customer_id": customer_id,
                "reason": "Customer tenure is below the minimum required threshold.",
                "segment": segment,
                "risk": risk
            }

        # Policy maximum
        if amount > max_voucher:
            return {
                "status": "escalation_required",
                "customer_id": customer_id,
                "action_type": action_type,
                "requested_amount": amount,
                "maximum_allowed": max_voucher,
                "segment": segment,
                "risk": risk,
                "reason": "Requested voucher exceeds the policy limit."
            }

        # Human approval threshold
        if amount >= 100:
            return {
                "status": "human_approval_required",
                "customer_id": customer_id,
                "action_type": action_type,
                "requested_amount": amount,
                "maximum_allowed": max_voucher,
                "segment": segment,
                "risk": risk,
                "reason": "Voucher requires explicit human approval."
            }

    return {
        "status": "pending",
        "customer_id": customer_id,
        "action_type": action_type,
        "details": details,
        "segment": segment,
        "risk": risk,
        "created_at": datetime.utcnow().isoformat()
    }