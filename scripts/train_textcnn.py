from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from news_sentinel.data.ag_news import read_jsonl
from news_sentinel.evaluation.metrics import classification_report_dict
from news_sentinel.model_registry import (
    append_registry_entry,
    build_run_id,
    get_git_snapshot,
    sha256_file,
    write_latest_snapshot,
)
from news_sentinel.models.textcnn import TextCNN
from news_sentinel.models.torch_text_data import build_vocabulary, create_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TextCNN on AG News.")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/textcnn"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract(rows: List[dict], max_samples: int) -> tuple[List[str], List[int]]:
    if max_samples > 0:
        rows = rows[:max_samples]
    return [r["text"] for r in rows], [int(r["label"]) for r in rows]


def resolve_device(mode: str) -> str:
    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_epoch(
    model: TextCNN,
    loader,
    optimizer,
    criterion,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_steps = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_steps += 1

    return total_loss / max(total_steps, 1)


def evaluate(model: TextCNN, loader, criterion, device: str) -> tuple[float, List[int], List[int]]:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            total_loss += float(loss.item())
            total_steps += 1

    return total_loss / max(total_steps, 1), all_labels, all_preds


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_rows = read_jsonl(args.train_file)
    test_rows = read_jsonl(args.test_file)

    train_texts, train_labels = _extract(train_rows, args.max_train_samples)
    test_texts, test_labels = _extract(test_rows, args.max_test_samples)

    vocab = build_vocabulary(train_texts)

    train_loader = create_dataloader(
        train_texts,
        train_labels,
        vocab=vocab,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = create_dataloader(
        test_texts,
        test_labels,
        vocab=vocab,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = resolve_device(args.device)
    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        num_filters=args.num_filters,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, y_true, y_pred = evaluate(model, test_loader, criterion, device)
        macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_macro_f1": round(macro_f1, 4),
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_macro_f1={macro_f1:.4f}"
        )

    final_loss, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    report = classification_report_dict(y_true, y_pred)
    report["loss"] = float(final_loss)
    report["model_type"] = "textcnn"
    report["device"] = device
    report["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    report["epochs"] = args.epochs
    report["batch_size"] = args.batch_size
    report["max_length"] = args.max_length
    report["history"] = history

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "textcnn.pt"
    report_path = args.output_dir / "train_report.json"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab": vocab.idx_to_token,
        "config": {
            "embedding_dim": args.embedding_dim,
            "num_filters": args.num_filters,
            "max_length": args.max_length,
            "num_classes": 4,
        },
    }

    torch.save(checkpoint, ckpt_path)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    repo_root = Path(__file__).resolve().parents[1]
    git = get_git_snapshot(repo_root)
    run_id = build_run_id("textcnn", git.commit)

    entry = {
        "run_id": run_id,
        "timestamp_utc": report["timestamp_utc"],
        "model_name": "textcnn",
        "artifact_path": str(ckpt_path),
        "artifact_sha256": sha256_file(ckpt_path),
        "report_path": str(report_path),
        "metrics": {
            "accuracy": report["accuracy"],
            "macro_f1": report["macro_f1"],
            "loss": report["loss"],
        },
        "dataset": {
            "train_file": str(args.train_file),
            "test_file": str(args.test_file),
            "n_train": len(train_texts),
            "n_test": len(test_texts),
        },
        "params": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "embedding_dim": args.embedding_dim,
            "num_filters": args.num_filters,
            "learning_rate": args.learning_rate,
        },
        "device": device,
        "git": {
            "commit": git.commit,
            "branch": git.branch,
            "dirty": git.dirty,
        },
    }

    registry_path = Path("artifacts/model_registry.jsonl")
    latest_snapshot_path = Path("artifacts/model_registry_latest.json")
    append_registry_entry(registry_path, entry)
    write_latest_snapshot(latest_snapshot_path, entry)

    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved report: {report_path}")
    print(f"Updated registry: {registry_path}")
    print(f"Updated latest snapshot: {latest_snapshot_path}")
    print(json.dumps({"macro_f1": report["macro_f1"], "accuracy": report["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
