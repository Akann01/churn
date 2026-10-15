# Churn Radar

An end-to-end customer churn system: it predicts who is likely to leave, explains why with SHAP, drafts a retention email for each at-risk customer, and puts all of it in an interactive Streamlit dashboard. It was built to move past a static model and into something a business user could actually act on.

## What it does

Predicts which e-commerce customers are likely to churn and shows the top reasons behind each score. A second stage turns those reasons into a short outreach email, with automatic checks so the email never says anything untrue. The dashboard also lets you try what-if scenarios by changing a customer's key risk factors and watching the score update.

## Demo

https://github.com/Akann01/churn/raw/main/dashboard_demo.mp4

## Dataset

E-Commerce Customer Churn dataset (Excel), with customer tenure, satisfaction scores, order behavior, complaints, and payment and device info.

## Approach

1. **Data cleaning.** Dropped the non-predictive ID column, filled missing numeric values with the median, and removed 558 exact duplicate rows (5,630 down to 5,072) so identical customers can't appear in both the training and test sets.
2. **EDA.** Churn rate broken down by login device, payment mode, gender, order category and marital status.
3. **Modeling.** Compared Logistic Regression (baseline, class-weighted) against Random Forest.
4. **Explainability.** SHAP (TreeExplainer) gives the top features driving each customer's churn risk.
5. **Retention agent.** An LLM (Groq) reads each customer's top drivers, picks an action and writes a short email. A validator checks every email before it is accepted.
6. **Dashboard.** A Streamlit app to explore at-risk customers, see the reasons, read the suggested email and run what-if scenarios.

## Results

| Model | ROC AUC |
|---|---|
| Logistic Regression | 0.874 |
| Random Forest | 0.984 |

Scores are on a held-out test set of 1,015 customers, after duplicates were removed. Before removing them, the Random Forest scored 0.999, which was inflated because about 17% of test rows had an identical twin in the training set.

The Random Forest is the model behind the dashboard and the SHAP explanations. On churners it has precision 0.97 and recall 0.76. Logistic Regression catches slightly more churners (recall 0.80) but raises many more false alarms (precision 0.43). That is a reasonable trade in retention, where missing a churner costs more than a false alarm, so both are worth keeping in view.

## Retention agent

The agent takes the 20 most at-risk customers and writes one email each. The risky part of using an LLM here is that it will say untrue things, so every email goes through a validator and is regenerated if it fails (up to four attempts, then a safe generic fallback is used).

The validator rejects an email if it:

- mentions a complaint, or says "thank you for reaching out", when the customer has no complaint driver
- calls someone a new customer when their tenure isn't under three months
- brings up drivers the customer shouldn't see, such as orders, satisfaction, ratings, addresses, cashback or distance
- hints that the customer might leave
- is outside the word range or repeats the offer line

**What it caught.** A manual review of five emails looked fine. A full run of 20 showed cashback customers being thanked for "raising a concern" they never raised. The prompt said to mention a concern but not to do it only when the complaint driver was present. After fixing the prompt and adding a validator rule, the next run flagged and regenerated the bad emails on its own. In the final run all 20 emails were written by the model, with no fallbacks needed. The validator still caught two slips along the way (a mention of a purchase and of saved addresses) and regenerated both. A scan of all 20 for banned words, including marital status, came back empty, and the emails average about 45 words.

Example, for a new customer with no complaint:

> **Subject:** Welcome! Enjoy a special offer
>
> Hi {first_name},
>
> We're excited to have you with us and want to make your first experience even better. As a new member, we'd love to give you a special incentive to explore our range.
>
> Reply to this email to receive a special offer.
>
> Warm regards,
> The Support Team

`{first_name}` is a placeholder filled in when the campaign is sent. Emails are drafts for review and are never sent automatically.

## Dashboard features

- Filter customers by minimum churn probability
- Sort by risk level
- Click into any customer to see their profile and predicted risk
- SHAP breakdown of the top factors behind that customer's risk (red raises it, blue lowers it)
- Suggested outreach: the action, the reason, and the email, with a tag showing whether it was written by the model or is the fallback
- What-if sliders that re-score a customer live using the trained model

## Tech stack

Python, pandas, scikit-learn, SHAP, Streamlit, Plotly, joblib, Groq API

## Running it locally

```bash
pip install -r requirements.txt
python churn.py                # trains the model, generates SHAP values and CSVs

set GROQ_API_KEY=your_key_here   # Windows; use export on Mac/Linux
python retention_agent.py      # writes retention_campaign.csv (--limit 3 to test)

streamlit run churn_radar.py   # launches the dashboard
```

The API key is read from the environment and is never stored in the code. If the key is rejected, the agent stops straight away instead of saving fallback emails. The dashboard still works if you skip the agent step; the outreach section just shows a short note.

## Files

- `churn.py`: data cleaning, model training, SHAP analysis
- `retention_agent.py`: LLM email agent and validator
- `churn_radar.py`: Streamlit dashboard
- `churn_model.pkl`, `scaler.pkl`: saved model artifacts
- `top_20_at_risk_customers.csv`, `shap_reasons.csv`: model outputs used by the dashboard
- `retention_campaign.csv`: generated emails, one row per customer, used by the dashboard

## Known limits

- Customer IDs in the dashboard are row numbers from the dataset, because the original ID column is dropped before training.
- The top 20 at-risk customers share similar drivers, so only a few actions get used.
- Fallback emails are generic, and their action label doesn't always match the text.
- Emails are drafts and are not sent automatically.
