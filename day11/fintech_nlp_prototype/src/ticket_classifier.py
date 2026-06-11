from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import json

llm = OpenAI(temperature=0)

classifier_prompt = PromptTemplate(
    input_variables=["ticket"],
    template="""
You are a support ticket classifier.

Step-by-step reasoning:
1. Identify the issue type
2. Check urgency or risk
3. Map to category

Return JSON with:
- category
- reasoning
- confidence (1-5)
- sla_hours

Ticket:
{ticket}
"""
)

def classify_ticket(ticket):
    prompt = classifier_prompt.format(ticket=ticket)
    raw = llm(prompt)
    return json.loads(raw)

def assign_sla(category):
    return {"Billing": 24, "Refund": 48, "Tech": 12, "General": 72, "Escalate": 4}[category]