# AI Phishing Email Detection Project

This project fine-tunes a Hugging Face text classification model to classify emails as:

- `legitimate`
- `phishing`

NOTE: MODEL FOLDER NOT INCLUDED (WILL HAVE TO BE RAN ON HOST MACHINE DUE TO UPLOAD LIMITS)

The dataset is synthetic and uses `[safe-link-placeholder]` instead of live URLs.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python train_model.py --dataset phishing_emails.csv
```

## Test Predictions

```bash
python predict_email.py
```

Or test one email:

```bash
python predict_email.py --text "Subject: Security alert Body: Verify your account at [safe-link-placeholder]."
```

## Dataset Columns

| Column | Description |
|---|---|
| `text` | Email subject and body |
| `label` | `legitimate` or `phishing` |

## Notes

This is a proof-of-concept model. A real production phishing detector would need:
- a larger dataset
- real-world email examples
- adversarial phishing samples
- URL/domain reputation checks
- attachment analysis
- sender metadata
- human review for high-risk decisions
