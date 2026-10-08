# Bank Churn AI

## 1. Project overview

Bank Churn AI is a prototype customer retention application for a
banking context.

Its purpose is to help a bank employee analyse a customer who may be at
risk of leaving the bank, understand the customer's complaints, obtain
an AI supported analysis and create a retention request.

The application combines customer data, a churn prediction model,
complaint history and a Microsoft Foundry Agent.

The current implementation intentionally keeps the architecture simple
and modular.

------------------------------------------------------------------------

## 2. What the application does

The main user journey is:

1.  The user enters a Customer ID.
2.  The user enters the customer's current complaint.
3.  The application retrieves the customer from the database.
4.  The application calculates the customer's churn probability.
5.  The application classifies the risk as LOW, MEDIUM or HIGH.
6.  The application displays the customer information.
7.  The user can consult the customer's historical complaints.
8.  The application creates a retention request.
9.  The application sends the customer context to the Foundry Agent.
10. The AI analysis is displayed when Foundry is available.
11. If Foundry is unavailable, the retention request is still preserved
    and marked as failed.

------------------------------------------------------------------------

## 3. Architecture

The application has four main layers.

``` text
┌──────────────────────────────────────────────┐
│              Streamlit UI                    │
│                                              │
│  Customer search                             │
│  Complaint input                             │
│  Customer information                        │
│  Churn and risk                              │
│  Complaint history                           │
│  AI analysis                                 │
└──────────────────────┬───────────────────────┘
                       │ HTTP
                       ▼
┌──────────────────────────────────────────────┐
│                FastAPI API                   │
│                                              │
│  /customers/{customer_id}                    │
│  /analyze                                    │
│  /retention/action                           │
└──────────────┬───────────────────┬───────────┘
               │                   │
               ▼                   ▼
┌────────────────────────┐   ┌─────────────────┐
│      SQLite DB         │   │ Microsoft       │
│                        │   │ Foundry Agent   │
│  customers             │   │                 │
│  complaints            │   │ AI analysis     │
│  retention_requests    │   │                 │
└────────────────────────┘   └─────────────────┘
```

### Streamlit

The Streamlit application is the user interface.

It is responsible for:

-   Customer ID input
-   Complaint input
-   Customer information
-   Churn prediction display
-   Risk visualisation
-   Complaint history
-   AI analysis display
-   User friendly error messages

The interface uses `st.session_state` to preserve the last searched
customer and its complaint history when Streamlit reruns the
application.

### FastAPI

FastAPI is the backend layer.

It is responsible for:

-   Retrieving customers
-   Retrieving complaint history
-   Executing the churn prediction
-   Creating retention requests
-   Calling the Foundry Agent
-   Updating the retention request status

### SQLite

SQLite is the current application database.

The database file is:

``` text
data/bank.db
```

SQLAlchemy is used for database access.

### Microsoft Foundry

Microsoft Foundry provides the AI Agent used for customer analysis.

The Agent receives customer information, churn information and the
current complaint.

------------------------------------------------------------------------

## 4. Database

The current database contains three relevant tables.

### 4.1 customers

Stores the customer information used by the application and the churn
model.

Main fields:

``` text
customer_id
age
gender
tenure_months
num_products
balance
credit_score
is_active_member
estimated_salary
has_credit_card
num_transactions_last_month
support_tickets_last_6m
churn
```

### 4.2 complaints

Stores historical customer complaints.

Main fields:

``` text
complaint_id
customer_id
date
channel
text
sentiment
```

The `customer_id` connects complaints with the corresponding customer.

### 4.3 retention_requests

Stores requests created as part of the customer retention process.

Fields:

``` text
id
customer_id
action_type
description
currency
reason
status
segment
risk
created_at
aproved_at
executed_at
```

The table is important because the retention request is considered a
business event that must not be lost when an external AI service is
unavailable.

------------------------------------------------------------------------

## 5. Main analysis flow

The complete flow is:

``` text
User
 │
 │ Customer ID + Complaint
 ▼
Streamlit
 │
 │ GET /customers/{customer_id}
 ▼
FastAPI
 │
 ├──────────────► SQLite
 │                 │
 │                 ├── Customer
 │                 └── Complaint history
 │
 └── Churn model
       │
       ▼
   Churn probability
   Risk classification
       │
       ▼
Streamlit displays result
       │
       │ POST /analyze
       ▼
FastAPI
       │
       ├── Create retention request
       │
       └── Call Foundry Agent
                │
          ┌─────┴─────┐
          │           │
       Success      Failure
          │           │
          ▼           ▼
      COMPLETED     FAILED
```

------------------------------------------------------------------------

## 6. Churn prediction

The backend calculates the churn prediction using:

``` python
predict_churn(customer_data)
```

The prediction provides:

``` text
churn_probability
risk
```

The risk is displayed in the user interface using three levels:

``` text
LOW       Green
MEDIUM    Orange
HIGH      Red
```

For HIGH risk, the interface also displays a warning icon.

The purpose is to make the risk immediately visible to the user.

------------------------------------------------------------------------

## 7. Complaint history

The customer endpoint retrieves the customer's historical complaints
from the `complaints` table.

The Streamlit interface stores this information in `session_state`.

After a customer has been searched, the user can select:

``` text
View Complaint History
```

The history displays:

-   Date
-   Channel
-   Sentiment
-   Complaint text

The Customer ID does not need to be entered again.

This approach is necessary because Streamlit reruns the script when the
history button is pressed.

------------------------------------------------------------------------

## 8. Retention request design decision

One of the main architectural decisions is that the retention request is
created independently of the Foundry Agent response.

The sequence is:

``` text
Analyse Customer
       │
       ▼
Create retention request
       │
       ▼
Call Foundry Agent
```

This means that the business request is preserved even when the AI
service is unavailable.

The request can therefore move through states such as:

``` text
PENDING
   │
   ├──► COMPLETED
   │
   └──► FAILED
```

The important distinction is:

-   `retention_requests` represents the business request.
-   Foundry represents the AI processing component.
-   The availability of Foundry must not determine whether the original
    request is stored.

This provides better resilience and traceability.

------------------------------------------------------------------------

## 9. Foundry failure handling

If Foundry is unavailable, the application does not discard the request.

Instead:

``` text
Request created
      ↓
Foundry call fails
      ↓
Request status = FAILED
```

The user receives a message indicating that the request was saved but
that the Foundry Agent is currently unavailable.

This behaviour allows the application to distinguish between:

-   A business request being created
-   AI processing succeeding
-   AI processing failing

------------------------------------------------------------------------

## 10. User interface

The current interface uses a dark theme provided by Streamlit.

The main visual decisions are:

-   Centred `Bank Churn AI` title
-   Blue title colour
-   Centred subtitle
-   Customer input section
-   Customer identification using `Customer - <ID>`
-   Three customer metrics
-   Churn probability and risk displayed together
-   Risk colour based on risk level
-   Warning icon for HIGH risk
-   Complaint section
-   Complaint history button
-   AI Analysis section

The interface was deliberately kept simple so that the main business
flow remains easy to understand.

------------------------------------------------------------------------

## 11. Streamlit state management

Streamlit reruns the application whenever the user interacts with a
widget.

For this reason, the application stores the following information in
`st.session_state`:

``` text
current_customer_id
current_customer
current_churn
current_complaints
analysis_data
show_history
customer_loaded
```

This allows the application to:

-   Clear the input form after a search
-   Keep the selected customer available
-   Show complaint history without losing the customer
-   Keep the churn result visible
-   Keep the AI result visible

------------------------------------------------------------------------

## 12. API endpoints

### GET /customers/{customer_id}

Retrieves:

``` text
Customer data
Complaint history
Churn prediction
```

### POST /analyze

Receives:

``` text
customer_id
complaint
```

It then:

1.  Retrieves the customer.
2.  Calculates churn.
3.  Creates the retention request.
4.  Calls the Foundry Agent.
5.  Updates the request status.
6.  Returns the result to Streamlit.

### POST /retention/action

Provides the endpoint for retention actions.

The current implementation keeps this functionality separate from the
initial analysis process.

------------------------------------------------------------------------

## 13. Project structure

The current project is organised approximately as follows:

``` text
project/
│
├── streamlit_app.py
├── api.py
├── models.py
│
├── app/
│   ├── churn.py
│   ├── retention.py
│   └── foundry_agent.py
│
└── data/
    └── bank.db
```

The separation allows each main responsibility to evolve independently.

------------------------------------------------------------------------

## 14. Technology stack

``` text
Python
Streamlit
FastAPI
SQLAlchemy
SQLite
Pydantic
Requests
Machine Learning
Microsoft Foundry
```

------------------------------------------------------------------------

## 15. Running the application

### Start the FastAPI backend

``` bash
uvicorn api:app --reload --port 8000
```

### Start the Streamlit interface

``` bash
streamlit run streamlit_app.py
```

The Streamlit application communicates with:

``` text
http://127.0.0.1:8000
```

The database is located at:

``` text
data/bank.db
```

------------------------------------------------------------------------

## 16. Current status

The current version provides a functional end to end prototype.

The main functionality is:

``` text
Customer search
       ↓
Customer information
       ↓
Churn prediction
       ↓
Risk classification
       ↓
Complaint history
       ↓
Retention request
       ↓
Foundry AI analysis
```

The application also handles Foundry unavailability without losing the
retention request.

------------------------------------------------------------------------

## 17. Design principles

The current implementation follows five main principles.

### Simplicity

The architecture should remain small and understandable.

### Separation of responsibilities

The UI, API, database, churn model, retention logic and AI Agent have
different responsibilities.

### Persistence

Business requests should be stored independently of external AI
availability.

### Resilience

An external service failure should not cause loss of application data.

### Controlled evolution

New components should only be introduced when they provide a clear
benefit.

------------------------------------------------------------------------

## 18. Possible future extensions

The current version is intentionally limited to the core scenario.

Possible future improvements include:

-   Retention request management
-   Approval workflow
-   Execution of retention actions
-   Management dashboard
-   Customer risk overview
-   AI evaluation
-   Authentication and authorisation
-   Production database
-   Azure deployment
-   Monitoring and logging

These are future extensions and are not required for the current
prototype.
