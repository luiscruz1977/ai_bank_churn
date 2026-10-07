import os
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

load_dotenv()

FOUNDRY_PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
FOUNDRY_AGENT_NAME = "BankChurnRetentionAgent"


project = AIProjectClient(
    endpoint=FOUNDRY_PROJECT_ENDPOINT,
    credential=DefaultAzureCredential(),
)

openai = project.get_openai_client(
    agent_name=FOUNDRY_AGENT_NAME
)

response = openai.responses.create(
    input="""
For customer C00327, invoke the trigger_retention_action tool now.

Use:
customer_id = C00327
action_type = voucher

details:
amount = 50
currency = EUR
reason = Customer retention

Actually invoke the tool and return its real result.
""",
    tool_choice="required",
)

print("\n===== AGENT RESPONSE =====\n")
print(response.output_text)

print("\n===== OUTPUT ITEMS =====\n")

for item in response.output:
    print(item)