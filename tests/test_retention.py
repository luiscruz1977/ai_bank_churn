from app.retention import trigger_retention_action


result = trigger_retention_action(
    customer_id="C00327",
    action_type="voucher",
    details={
        "amount": 100,
        "currency": "EUR",
        "reason": "Customer retention"
    }
)

print(result)