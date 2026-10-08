from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from .database import Base


class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(String, primary_key=True)
    age = Column(Integer)
    gender = Column(String)
    tenure_months = Column(Integer)
    num_products = Column(Integer)
    balance = Column(Float)
    credit_score = Column(Integer)
    is_active_member = Column(Boolean)
    estimated_salary = Column(Float)
    has_credit_card = Column(Boolean)
    num_transactions_last_month = Column(Integer)
    support_tickets_last_6m = Column(Integer)
    churn = Column(Integer)


class Complaint(Base):
    __tablename__ = "complaints"

    complaint_id = Column(String, primary_key=True)
    customer_id = Column(String, ForeignKey("customers.customer_id"))
    date = Column(DateTime)
    channel = Column(String)
    text = Column(String)
    sentiment = Column(String)

class RetentionRequest(Base):
    __tablename__ = "retention_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String, ForeignKey("customers.customer_id"))
    action_type = Column(String)
    description = Column(String)
    currency = Column(String)
    reason = Column(String)
    status = Column(String)
    segment = Column(String)
    risk = Column(String)
    created_at = Column(DateTime)
    aproved_at = Column(DateTime, nullable=True)
    executed_at = Column(DateTime, nullable=True)