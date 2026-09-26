"""The demo's data: a small knowledge base, an order table, and a deterministic keyword retriever."""

from __future__ import annotations

import math
import re

# id, category, title, body
ARTICLES = [
    ("returns", "returns", "Return policy",
     "Items can be returned within 30 days of delivery in their original packaging. Refunds go to the original payment method within 5-7 business days after we receive the item."),
    ("shipping-standard", "shipping", "Standard shipping",
     "Standard shipping takes 3-5 business days and is free on orders over $50. Orders under $50 pay a flat $4.99 for standard shipping."),
    ("shipping-express", "shipping", "Express shipping",
     "Express shipping takes 1-2 business days and costs $14.99. Orders placed before 2 pm ET ship the same day."),
    ("shipping-intl", "shipping", "International shipping",
     "We ship to 40 countries. International orders take 7-14 business days. Duties and taxes are paid by the recipient."),
    ("password-reset", "account", "Resetting your password",
     "To reset your password choose 'Forgot password' on the sign-in page. The reset link is valid for 30 minutes."),
    ("two-factor", "account", "Two-factor authentication",
     "Two-factor authentication can be turned on under Settings > Security. We support authenticator apps and SMS codes."),
    ("warranty", "products", "Warranty",
     "All electronics carry a 12-month limited warranty that covers manufacturing defects. Accidental damage is not covered."),
    ("price-match", "payments", "Price matching",
     "We match a lower price from an authorised retailer within 14 days of purchase. Marketplace sellers and clearance items are excluded."),
    ("cancel-order", "orders", "Cancelling an order",
     "Orders can be cancelled free of charge within 1 hour of placing them. After that the order enters fulfilment and must be returned instead."),
    ("gift-cards", "payments", "Gift cards",
     "Gift cards never expire and can be used on any order. Gift cards cannot be redeemed for cash."),
    ("loyalty", "account", "Loyalty points",
     "Members earn 1 point per $1 spent. Every 500 points can be redeemed for a $10 reward."),
    ("subscription", "orders", "Managing subscriptions",
     "You can pause or cancel a subscription any time before the 5th of the month to avoid the next charge."),
    ("invoice", "orders", "Invoices",
     "Invoices are emailed after shipment and can be downloaded as a PDF from Account > Orders."),
    ("damaged", "returns", "Damaged items",
     "If an item arrives damaged, report it within 48 hours with a photo. We send a free replacement."),
    ("support-hours", "support", "Support hours",
     "Our support team is available Monday to Friday, 9 am to 6 pm ET. Chat replies typically arrive in under 5 minutes."),
    ("address-change", "orders", "Changing the shipping address",
     "The shipping address can be changed before the order ships, under Account > Orders > Edit address."),
    ("payment-methods", "payments", "Accepted payment methods",
     "We accept Visa, Mastercard, American Express, PayPal and Apple Pay. We do not accept cheques or cash on delivery."),
    ("bulk", "payments", "Bulk orders",
     "Orders of 50 or more units qualify for a 12% business discount. Contact our sales team for a quote."),
]
DOCS = {a[0]: {"id": a[0], "category": a[1], "title": a[2], "text": a[3]} for a in ARTICLES}

ORDERS = {
    "A1001": {"status": "shipped", "carrier": "UPS", "eta": "Sep 30", "total": "$84.20"},
    "A1002": {"status": "delivered", "carrier": "FedEx", "eta": "Sep 22", "total": "$19.99"},
    "A1003": {"status": "processing", "carrier": "pending", "eta": "Oct 3", "total": "$129.00"},
    "A1004": {"status": "cancelled", "carrier": "none", "eta": "none", "total": "$54.50"},
    "A1005": {"status": "shipped", "carrier": "DHL", "eta": "Oct 2", "total": "$210.75"},
    "A1006": {"status": "returned", "carrier": "UPS", "eta": "none", "total": "$33.10"},
    "A1007": {"status": "delivered", "carrier": "USPS", "eta": "Sep 25", "total": "$47.00"},
    "A1008": {"status": "processing", "carrier": "pending", "eta": "Oct 5", "total": "$76.40"},
    "A1009": {"status": "shipped", "carrier": "FedEx", "eta": "Oct 1", "total": "$15.25"},
    "A1010": {"status": "delivered", "carrier": "UPS", "eta": "Sep 19", "total": "$302.00"},
    "A1011": {"status": "shipped", "carrier": "USPS", "eta": "Sep 29", "total": "$64.80"},
    "A1012": {"status": "processing", "carrier": "pending", "eta": "Oct 4", "total": "$22.30"},
}

_STOP = set(["a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for", "and", "or", "i", "my", "me", "you", "your", "it", "do", "does", "did", "can", "could", "how", "what", "when", "where", "why", "which", "who", "this", "that", "with", "at", "as", "by", "from", "if", "so", "we", "our", "us", "please", "hi", "hello", "thanks", "thank", "any", "about", "there", "have", "has", "had", "not", "no", "yes", "get", "got", "will", "would", "just"])


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9$%]+", text.lower()) if t not in _STOP]


_DF: dict[str, int] = {}
for _d in DOCS.values():
    for _t in set(tokens(_d["title"] + " " + _d["text"])):
        _DF[_t] = _DF.get(_t, 0) + 1


def kb_search(query: str, k: int = 5) -> list[dict]:
    """Deterministic idf-weighted keyword overlap. Deliberately crude: it returns near-misses too,
    which is exactly what the relevance-scoring decision has to sort out."""
    q = set(tokens(query))
    scored = []
    for d in DOCS.values():
        dt = set(tokens(d["title"] + " " + d["text"]))
        s = sum(math.log(1 + len(DOCS) / _DF[t]) for t in q & dt)
        scored.append((-s, d["id"]))
    scored.sort()
    return [dict(DOCS[i]) for _, i in scored[:k]]


def order_lookup(order_id: str) -> dict:
    o = ORDERS.get(order_id)
    return {"order_id": order_id, **o} if o else {"order_id": order_id, "error": "not found"}
