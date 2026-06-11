from groq_utils import chat_completion, extract_json


def _rule_based_classification(ticket):
    lowered = ticket.lower()
    category = "General"
    reasoning = "No strong signal matched, so the ticket was routed to the general queue."
    confidence = 2

    if any(keyword in lowered for keyword in ["refund", "chargeback", "money back"]):
        category = "Refund"
        reasoning = "The ticket mentions a refund-related issue."
        confidence = 5
    elif any(keyword in lowered for keyword in ["bill", "billing", "charge", "invoice"]):
        category = "Billing"
        reasoning = "The ticket references billing or payment charges."
        confidence = 5
    elif any(keyword in lowered for keyword in ["crash", "login", "error", "bug", "issue", "down", "failed"]):
        category = "Tech"
        reasoning = "The ticket describes a technical problem affecting usage."
        confidence = 5
    elif any(keyword in lowered for keyword in ["urgent", "asap", "immediately", "escalate", "critical"]):
        category = "Escalate"
        reasoning = "The ticket indicates urgency or escalation risk."
        confidence = 5

    return {
        "category": category,
        "reasoning": reasoning,
        "confidence": confidence,
        "sla_hours": assign_sla(category),
    }

def classify_ticket(ticket):
    try:
        raw = chat_completion(
            prompt=f"""
Return valid JSON with keys: category, reasoning, confidence, sla_hours.

Ticket:
{ticket}
""",
            system="Classify support tickets into Billing, Refund, Tech, General, or Escalate.",
        )
        parsed = extract_json(raw)
        parsed.setdefault("sla_hours", assign_sla(parsed.get("category", "General")))
        return parsed
    except Exception:
        pass

    return _rule_based_classification(ticket)

def assign_sla(category):
    return {"Billing": 24, "Refund": 48, "Tech": 12, "General": 72, "Escalate": 4}.get(category, 72)