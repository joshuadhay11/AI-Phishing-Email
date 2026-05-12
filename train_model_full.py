"""
Dataset format:
    text,label
    "Subject: Example\nBody: Example email text",legitimate
    "Subject: Example\nBody: Suspicious email text",phishing

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
    """
    Compute binary classification metrics.

    Positive class:
        phishing = 1

    Negative class:
        legitimate = 0
    """

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = (predictions == labels).mean().item()

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

    parser.add_argument(
        "--dataset",
        default="phishing_emails.csv",
        help="Path to CSV dataset.",
    )

    parser.add_argument(
        "--model",
        default="distilbert-base-uncased",
        help="Base Hugging Face model.",
    )

    parser.add_argument(
        "--output_dir",
        default="phishing_detector_model",
        help="Directory to save trained model.",
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=3,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training and evaluation batch size.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=192,
        help="Maximum token length for each email.",
    )

    parser.add_argument(
        "--train_all",
        action="store_true",
        help="Train on the entire dataset without a reserved test split.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    print(f"\nLoading dataset from: {args.dataset}")

    raw_dataset = load_dataset("csv", data_files=args.dataset)["train"]

    print(f"Loaded {len(raw_dataset)} total examples.")

    def encode_label(example):
        label = str(example["label"]).strip().lower()

        if label not in LABEL2ID:
            raise ValueError(
                f"Unknown label '{example['label']}'. "
                f"Expected one of {list(LABEL2ID.keys())}"
            )

        example["labels"] = LABEL2ID[label]
        return example

    raw_dataset = raw_dataset.map(encode_label)

    if args.train_all:
        print("\nTraining mode: FULL DATASET")
        print("No reserved test set will be used.")
        print("Do not report this run as validation accuracy.\n")
        split_dataset = raw_dataset
    else:
        print("\nTraining mode: 80/20 TRAIN/TEST SPLIT")
        print("80% of examples used for training.")
        print("20% of examples reserved for evaluation.\n")
        split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)

        print(f"Training examples: {len(split_dataset['train'])}")
        print(f"Testing examples:  {len(split_dataset['test'])}")

    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    print("\nTokenizing dataset...")
    tokenized = split_dataset.map(tokenize, batched=True)

    tokenized = tokenized.remove_columns(["text", "label"])
    tokenized.set_format("torch")

    print(f"\nLoading model: {args.model}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Hugging Face changed `evaluation_strategy` to `eval_strategy`
    # in some newer versions. This checks what your installed version expects.
    training_args_signature = inspect.signature(TrainingArguments.__init__)
    eval_arg_name = (
        "eval_strategy"
        if "eval_strategy" in training_args_signature.parameters
        else "evaluation_strategy"
    )

    training_kwargs = {
        "output_dir": args.output_dir,
        "save_strategy": "epoch",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "logging_dir": "logs",
        "logging_steps": 10,
        "report_to": "none",
    }

    if args.train_all:
        training_kwargs[eval_arg_name] = "no"
        training_kwargs["load_best_model_at_end"] = False
    else:
        training_kwargs[eval_arg_name] = "epoch"
        training_kwargs["load_best_model_at_end"] = True
        training_kwargs["metric_for_best_model"] = "f1_phishing"
        training_kwargs["greater_is_better"] = True

    trainer_kwargs = {
        "model": model,
        "args": TrainingArguments(**training_kwargs),
        "train_dataset": tokenized if args.train_all else tokenized["train"],
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
    }

    if not args.train_all:
        trainer_kwargs["eval_dataset"] = tokenized["test"]
        trainer_kwargs["compute_metrics"] = compute_metrics

    trainer_signature = inspect.signature(Trainer.__init__)

    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    print("\nStarting training...")
    trainer.train()

    if args.train_all:
        print("\nTraining completed on the full dataset.")
        print("No evaluation metrics were calculated because no reserved test set was used.")
    else:
        results = trainer.evaluate()

        print("\nEvaluation results:")
        for key, value in results.items():
            print(f"{key}: {value}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nSaved model to: {args.output_dir}")


if __name__ == "__main__":
    main()