import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "sqlite:///./data/bank.db"

engine = create_engine(DATABASE_URL)

customers = pd.read_csv("data/customer_churn.csv")
complaints = pd.read_csv("data/customer_complaints.csv")

customers.to_sql(
    "customers",
    engine,
    if_exists="replace",
    index=False
)

complaints.to_sql(
    "complaints",
    engine,
    if_exists="replace",
    index=False
)

print("Database created successfully.")
print(f"Customers: {len(customers)}")
print(f"Complaints: {len(complaints)}")