"""Example: bring-your-own harness. An inbox agent with three decisions and one generation.

    jevcontrol run examples/email_triage/experiment.yaml     # from the repo root, after editing the endpoints

or, in the app: New experiment -> "My harness" -> path to this file.
"""

META = {
    "name": "Email triage agent",
    "description": "Spam gate -> category -> urgency -> draft a reply for support mail.",
    "sites": {
        "spam": "Is this message spam or a phishing attempt?",
        "category": "Which team should handle it?",
        "urgency": "How urgent is it (0 = whenever, 3 = right now)?",
    },
}

CATEGORIES = {
    "support": "A customer needs help with a product or an order",
    "sales": "A prospect asks about pricing, plans or a purchase",
    "billing": "Invoices, payments, refunds or charges",
    "other": "Anything else: newsletters, recruiting, partnerships",
}
URGENCY = {
    "0": "No deadline; can wait days",
    "1": "Should be answered within a working day",
    "2": "Needs an answer within hours",
    "3": "Something is broken or lost right now",
}


def run(task, ctx):
    mail = {"from": task["from"], "subject": task["subject"], "body": task["body"]}
    if ctx.decide.noul("spam", mail, "The message is spam, a scam, or a phishing attempt").is_true:
        return {"action": "discard"}
    cat = ctx.decide.choice("category", mail, "Which team should handle this message?", CATEGORIES).selected
    urgency = int(ctx.decide.score("urgency", mail, "How urgent is this message?", URGENCY).selected)
    out = {"action": "route", "team": cat, "urgent": urgency >= 2}
    if cat == "support":
        out["draft"] = ctx.llm.chat(
            [{"role": "system", "content": "Write a two-sentence, friendly first reply to this customer. Do not promise refunds."},
             {"role": "user", "content": f"Subject: {mail['subject']}\n\n{mail['body']}"}], max_tokens=120)
    return out


def score(task, out):
    exp = task["expected"]
    if out.get("action") != exp["action"]:
        return 0.0
    if exp["action"] == "discard":
        return 1.0
    return float(out.get("team") == exp["team"] and out.get("urgent") == exp["urgent"])
