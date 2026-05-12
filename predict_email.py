"""
Use the fine-tuned phishing detector model.

Run:
    python predict_email.py --text "Urgent: verify your account now at [safe-link-placeholder]"

Or:
    python predict_email.py
"""

import argparse
from transformers import pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="phishing_detector_model")
    parser.add_argument("--text", default=None)
    args = parser.parse_args()

    classifier = pipeline(
        "text-classification",
        model=args.model_dir,
        tokenizer=args.model_dir,
    )

    if args.text:
        examples = [args.text]
    else:
        examples = [
            "Subject: Meeting update\nBody: The project sync has been moved to 2 PM tomorrow.",
            "Subject: URGENT account action required\nBody: Your password expires today. Verify your account at [safe-link-placeholder].",
            "Subject: Vendor invoice\nBody: The attached invoice was routed through the standard approval workflow.",
            "Subject: CEO confidential request\nBody: Purchase gift cards immediately and reply with the codes.",
        ]

    for email in examples:
        result = classifier(email)[0]
        print("\n--- Email ---")
        print(email)
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['score']:.4f}")


if __name__ == "__main__":
    main()
