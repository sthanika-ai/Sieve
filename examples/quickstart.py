"""Three typed questions about one support ticket, answered in one pass. Needs a CUDA GPU with about 20 GB."""
import json

from sieve import decide, load_sieve

m = load_sieve("sthanika-ai/Sieve-9B")

state = {"subject": "Duplicate charge on invoice 4411",
         "body": "Billed twice for March. Refund today or we cancel the annual contract."}
questions = {
    "department": {"type": "choice", "instructions": "Which team handles this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs and outages",
                                "sales": "pricing and contracts"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["can wait", "this week", "today"]},
    "churn_risk": {"type": "noul", "instructions": "Does the customer threaten to cancel?"},
}

print(json.dumps(decide(m, state, questions), indent=2))
