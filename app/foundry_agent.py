import os

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
AGENT_NAME = "BankChurnRetentionAgent"

project = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential(),
    allow_preview=True,
)

openai = project.get_openai_client(
    agent_name=AGENT_NAME
)


def analyze_customer(
    customer: dict,
    churn: dict,
    complaint: str
) -> str:

    prompt = f"""
Analyse this banking customer complaint according to the Retention Policy.

CUSTOMER DATA:
{customer}

CHURN:
{churn}

COMPLAINT:
{complaint}

Follow the Retention Policy available through File Search.

Analyse:
- tone
- main concern
- complaint summary
- sentiment consistency
- churn risk
- retention eligibility
- recommended action
- recommended voucher

If a policy-compliant retention action is permitted and does not require human approval,
invoke the trigger_retention_action tool.

Return the actual tool result when a tool is invoked.
Do not invent customer data, policy rules or tool results.
"""

    response = openai.responses.create(
        input=prompt
    )

    return response.output_text