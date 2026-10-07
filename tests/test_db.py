from sqlalchemy import create_engine, text

DATABASE_URL = "sqlite:///./data/bank.db"

engine = create_engine(DATABASE_URL)

customer_id = input("Customer ID: ")

with engine.connect() as connection:

    result = connection.execute(
        text("""
            SELECT *
            FROM customers
            WHERE customer_id = :customer_id
        """),
        {"customer_id": customer_id}
    )

    customer = result.fetchone()

    if customer:
        print("\nCustomer found:")
        print(customer)

        complaints = connection.execute(
            text("""
                SELECT *
                FROM complaints
                WHERE customer_id = :customer_id
            """),
            {"customer_id": customer_id}
        )

        print("\nComplaints:")

        for complaint in complaints:
            print(complaint)

    else:
        print("Customer not found.")