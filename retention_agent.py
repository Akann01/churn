"""
retention_agent.py - turns Churn Radar's SHAP output into ready-to-send retention outreach.

Setup:
    pip install groq pandas
    set GROQ_API_KEY=your_key          (Windows)

Run:
    python retention_agent.py
    python retention_agent.py --limit 5
    python retention_agent.py --send-test-to you@gmail.com --send-limit 3
"""

import argparse
import json
import os
import smtplib
import time
from email.message import EmailMessage

import pandas as pd

MODEL = "openai/gpt-oss-20b"  # a chat model available on this Groq account
SIGN_OFF = "The Support Team"

FRIENDLY = {
    "Tenure": "months as a customer",
    "SatisfactionScore": "satisfaction score (1-5)",
    "Complain": "raised a complaint recently (1 = yes)",
    "DaySinceLastOrder": "days since last order",
    "CashbackAmount": "average cashback received",
    "WarehouseToHome": "warehouse-to-home distance",
    "HourSpendOnApp": "hours spent on the app",
    "NumberOfDeviceRegistered": "devices registered",
    "NumberOfAddress": "saved delivery addresses",
    "OrderAmountHikeFromlastYear": "% increase in order amount vs last year",
    "CouponUsed": "coupons used",
    "OrderCount": "orders placed",
    "CityTier": "city tier",
}

ALLOWED_ACTIONS = [
    "Personal call from support",
    "Resolve complaint and follow up",
    "Targeted cashback or discount offer",
    "Loyalty perk or early access",
    "Re-engagement reminder",
    "Delivery or service improvement outreach",
]

SYSTEM_PROMPT = f"""You are a retention specialist for an e-commerce company.
You get one at-risk customer: their churn probability and the top factors behind it.
Return ONLY a JSON object with these keys:
  "action": exactly one of {json.dumps(ALLOWED_ACTIONS)}
  "reason": one sentence linking the action to the customer's drivers
  "subject": email subject line, under 9 words
  "message": email body, 60-90 words, warm and specific, ending with "Warm regards," and then "The Support Team" on its own line
Rules:
- Use the placeholder {{first_name}} for the customer's name. Never invent a name.
- Read each driver's value literally. "days since last order" = 1 means they ordered yesterday, so never say they have been inactive. "months as a customer" = 1 means they are brand new, not long-time.
- In the email, mention (a) that they recently raised a concern ONLY if the driver "raised a complaint recently (1 = yes)" is in the list with value 1, and never guess what it was about. Mention (b) that they are new ONLY if the driver "months as a customer" is in the list with a value under 3. If a driver is not in the list, do not mention it or imply it, in the email or in the "reason" field. Do not mention any other driver in the email. Other drivers are for the "reason" field only.
- If there is no complaint driver, never say "thank you for reaching out" or imply they contacted us. Open with a friendly check-in instead.
- Do not hint that they might leave: no "stay", "keep you with us" or "before you go".
- Never claim an action has already been taken. Do not say "we have escalated", "we reviewed your case", or "we added an offer".
- Never invent details: no phone numbers, deadlines such as "within 24 hours", order numbers, or product names.
- Do not mention churn, risk scores, or that a model flagged them.
- Do not give specific percentages or amounts. You may mention a special offer once, only as something they can get by replying.
- Never use {{first_name}} in the subject line, only in the email body.
- Do not promise speed or timing: no "promptly", "quickly", "soon" or "right away".
- Mention the offer in exactly one sentence: "Reply to this email to receive a special offer." Never write "offer a special offer", and do not give the reply instruction anywhere else, including the subject.
- The email must not refer to orders, purchases, satisfaction, ratings, saved addresses, cashback, distance, or any driver other than a complaint or being new. Keep it a friendly, general check-in plus the offer.
- Put "The Support Team" on its own line after a blank line, with "Warm regards," above it.
- The one next step is always: reply to this email."""


def load_data(top_path, shap_path):
    top = pd.read_csv(top_path, index_col=0)
    shap = pd.read_csv(shap_path)
    overlap = set(top.index) & set(shap["customer_id"])
    if not overlap:
        raise ValueError(
            "No matching IDs between top_20_at_risk_customers.csv (first column) and "
            "shap_reasons.csv (customer_id). Check how IDs are assigned in churn.py."
        )
    return top, shap


def build_drivers(customer_id, row, shap_df, k=3):
    s = shap_df[(shap_df["customer_id"] == customer_id) & (shap_df["shap_value"] > 0)]
    s = s.sort_values("shap_value", ascending=False)
    drivers = []
    for _, r in s.iterrows():
        feat = r["feature"]
        if feat not in FRIENDLY:
            continue
        val = row[feat] if feat in row.index else r["feature_value"]
        drivers.append(
            {"factor": FRIENDLY[feat], "value": round(float(val), 1), "impact": round(float(r["shap_value"]), 3)}
        )
        if len(drivers) == k:
            break
    return drivers


def fallback_campaign(drivers):
    names = " ".join(d["factor"] for d in drivers).lower()
    if "complaint" in names:
        action = "Resolve complaint and follow up"
    elif "satisfaction" in names:
        action = "Personal call from support"
    elif "days since last order" in names:
        action = "Re-engagement reminder"
    else:
        action = "Targeted cashback or discount offer"
    return {
        "action": action,
        "reason": "Rule-based fallback from the customer's top risk drivers.",
        "subject": "We'd love to hear from you",
        "message": (
            "Hi {first_name},\n\nWe wanted to check in and make sure you are happy with your experience. "
            "Your feedback matters to us. "
            "Reply to this email to receive a special offer, and our team will take care of the rest.\n\n"
            f"Warm regards,\n{SIGN_OFF}"
        ),
    }


def clean_output(out):
    """Code-level guardrails: don't rely on the LLM alone for formatting."""
    # Subject must never contain the placeholder.
    out["subject"] = out["subject"].replace("{first_name}", "").strip(" ,")

    # Normalise the sign-off: strip any existing one, then append it cleanly.
    body = out["message"].strip()
    for tail in (SIGN_OFF, "Warm regards,", "Warm regards"):
        if body.endswith(tail):
            body = body[: -len(tail)].rstrip()
    # Put the greeting on its own line.
    prefix = "Hi {first_name},"
    if body.startswith(prefix):
        body = f"{prefix}\n\n{body[len(prefix):].lstrip()}"

    out["message"] = f"{body}\n\nWarm regards,\n{SIGN_OFF}"
    return out


BANNED_PHRASES = [
    "promptly", "quickly", "right away", "within 24", "within 48", "within a day",
    "escalated", "we reviewed", "offer you a special offer", "offer a special offer",
    "haven't ordered", "churn", "1-800",
    "help you stay", "keep you with us", "before you go", "leaving",
]

# Only allowed when the matching driver is actually in the customer's list.
COMPLAINT_WORDS = ["concern", "complain", "reaching out", "reached out", "your issue", "the issue"]
NEW_WORDS = ["new member", "new customer", "new to us", "brand new", "just joined", "welcome"]

# Other drivers are for the internal "reason" field only, never the customer email.
OTHER_DRIVER_WORDS = [
    "satisfaction", "rated us", "rating", "address", "cashback", "purchase",
    "your order", "recent order", "last order", "orders", "warehouse", "distance", "coupon",
]


def validate_output(out, drivers):
    """Raise ValueError if the email breaks a rule, so generate_campaign retries."""
    text = out["message"].replace("\u2019", "'").lower()
    for phrase in BANNED_PHRASES:
        if phrase in text:
            raise ValueError(f"Banned phrase in email: {phrase!r}")
    if text.count("special offer") > 1:
        raise ValueError("'special offer' mentioned more than once")

    # Facts must be backed by the customer's actual drivers.
    has_complaint = any(
        d["factor"].startswith("raised a complaint") and d["value"] >= 1 for d in drivers
    )
    is_new = any(d["factor"] == "months as a customer" and d["value"] < 3 for d in drivers)
    full = (out["subject"] + " " + out["message"] + " " + out["reason"]).replace("\u2019", "'").lower()
    if not has_complaint:
        for word in COMPLAINT_WORDS:
            if word in full:
                raise ValueError(f"Mentions a complaint with no complaint driver: {word!r}")
    if not is_new:
        for word in NEW_WORDS:
            if word in full:
                raise ValueError(f"Calls customer new with no new-customer driver: {word!r}")

    # The customer-facing email must not reveal other drivers or repeat the reply instruction.
    email_text = (out["subject"] + " " + out["message"]).replace("\u2019", "'").lower()
    for word in OTHER_DRIVER_WORDS:
        if word in email_text:
            raise ValueError(f"Email mentions a driver it should not: {word!r}")
    if email_text.count("reply") > 1:
        raise ValueError("Reply instruction repeated")
    if "{first_name}" not in out["message"]:
        raise ValueError("Missing {first_name} placeholder")
    words = len(out["message"].split())
    if not 35 <= words <= 110:
        raise ValueError(f"Email length out of range: {words} words")


def extract_json(text):
    """Pull the JSON object out of the model's reply instead of relying on JSON mode."""
    if not text:
        raise ValueError("Empty response from model")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in response")
    return json.loads(text[start : end + 1])


def generate_campaign(client, probability, drivers, retries=3):
    payload = {"churn_probability": round(float(probability), 2), "top_drivers": drivers}
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                temperature=0.4,
                max_tokens=2000,
                extra_body={"reasoning_effort": "low"},
            )
            out = extract_json(resp.choices[0].message.content)
            if out.get("action") not in ALLOWED_ACTIONS:
                raise ValueError(f"Action not in allowed list: {out.get('action')}")
            if not all(k in out for k in ("reason", "subject", "message")):
                raise ValueError("Missing keys in LLM response")
            out = clean_output(out)
            validate_output(out, drivers)
            return out, "llm"
        except Exception as e:
            if "invalid_api_key" in str(e) or "Error code: 401" in str(e):
                raise SystemExit(
                    "Groq rejected the API key (401). Set it again in this terminal with:\n"
                    "    set GROQ_API_KEY=gsk_your_real_key\n"
                    "No quotes, no spaces around '='. Nothing was saved."
                )
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(1.5)
    return fallback_campaign(drivers), "fallback"


def send_test_emails(rows, to_addr, limit):
    user, pwd = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASS")
    if not (user and pwd):
        print("Set SMTP_USER and SMTP_PASS (Gmail App Password) to send test emails.")
        return
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(user, pwd)
        for r in rows[:limit]:
            msg = EmailMessage()
            msg["From"], msg["To"] = user, to_addr
            msg["Subject"] = f"[TEST] {r['subject']}".replace("{first_name}", "Alex")
            msg.set_content(r["message"].replace("{first_name}", "Alex"))
            server.send_message(msg)
            print(f"  sent test email for customer {r['customer_id']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", default="top_20_at_risk_customers.csv")
    ap.add_argument("--shap", default="shap_reasons.csv")
    ap.add_argument("--out", default="retention_campaign.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--send-test-to", default=None)
    ap.add_argument("--send-limit", type=int, default=3)
    args = ap.parse_args()

    top, shap = load_data(args.top, args.shap)
    if args.limit:
        top = top.head(args.limit)

    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    results = []
    for cid, row in top.iterrows():
        drivers = build_drivers(cid, row, shap)
        prob = row["Churn_Probability"]
        print(f"Customer {cid} (p={prob:.2f}) drivers: {[d['factor'] for d in drivers]}")
        out, source = generate_campaign(client, prob, drivers)
        results.append(
            {
                "customer_id": cid,
                "churn_probability": round(float(prob), 3),
                "top_drivers": json.dumps(drivers),
                "action": out["action"],
                "reason": out["reason"],
                "subject": out["subject"],
                "message": out["message"],
                "source": source,
            }
        )
        time.sleep(0.5)

    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"\nSaved {len(results)} campaign rows to {args.out}")

    if args.send_test_to:
        send_test_emails(results, args.send_test_to, args.send_limit)


if __name__ == "__main__":
    main()