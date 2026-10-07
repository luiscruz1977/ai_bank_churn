from sqlalchemy import create_engine, text

from app.churn import predict_churn


DATABASE_URL = "sqlite:///./data/bank.db"

engine = create_engine(DATABASE_URL)

customer_id = input("Customer ID: ")

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
        print("Customer not found.")
        exit()

    result = predict_churn(dict(customer))

    print("\nChurn Prediction")
    print("----------------")
    print(f"Customer: {customer_id}")
    print(f"Probability: {result['churn_probability']:.2%}")
    print(f"Risk: {result['risk']}")