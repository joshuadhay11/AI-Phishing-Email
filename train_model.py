"""
Dataset format:
    text,label
    "Subject: Example\nBody: Example email text" , legitimate
    "Subject: Example\nBody: Suspicious email text" , phishing

Output:
    ./phishing_detector_model/
"""

import argparse
import inspect
import os
from typing import Dict

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


LABEL2ID = {
    "legitimate": 0,
    "phishing": 1,
}

ID2LABEL = {
    0: "legitimate",
    1: "phishing",
}


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = (predictions == labels).mean().item()

    # Binary metrics for phishing as the positive class.
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision_phishing": precision,
        "recall_phishing": recall,
        "f1_phishing": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="phishing_emails.csv", help="Path to CSV dataset.")
    parser.add_argument("--model", default="distilbert-base-uncased", help="Base Hugging Face model.")
    parser.add_argument("--output_dir", default="phishing_detector_model", help="Directory to save trained model.")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=192)
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    raw_dataset = load_dataset("csv", data_files=args.dataset)["train"]

    def encode_label(example):
        label = str(example["label"]).strip().lower()
        if label not in LABEL2ID:
            raise ValueError(f"Unknown label '{example['label']}'. Expected one of {list(LABEL2ID)}")
        example["labels"] = LABEL2ID[label]
        return example

    raw_dataset = raw_dataset.map(encode_label)
    split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = split_dataset.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(["text", "label"])
    tokenized.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Transformers renamed evaluation_strategy to eval_strategy in newer versions.
    training_args_signature = inspect.signature(TrainingArguments.__init__)
    eval_arg_name = "eval_strategy" if "eval_strategy" in training_args_signature.parameters else "evaluation_strategy"

    training_kwargs = {
        "output_dir": args.output_dir,
        eval_arg_name: "epoch",
        "save_strategy": "epoch",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "logging_dir": "logs",
        "logging_steps": 10,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_phishing",
        "greater_is_better": True,
        "report_to": "none",
    }

    trainer_kwargs = {
        "model": model,
        "args": TrainingArguments(**training_kwargs),
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["test"],
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    
    trainer_signature = inspect.signature(Trainer.__init__)
    
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    
    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    results = trainer.evaluate()

    print("\nEvaluation results:")
    for key, value in results.items():
        print(f"{key}: {value}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nSaved model to: {args.output_dir}")


if __name__ == "__main__":
    main()
