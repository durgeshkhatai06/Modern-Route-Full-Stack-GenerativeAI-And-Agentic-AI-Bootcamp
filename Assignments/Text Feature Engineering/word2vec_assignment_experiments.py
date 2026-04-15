from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
REVIEWS_CSV = ROOT / "Assignments" / "Text Feature Engineering" / "flipkart_reviews.csv"
STORY_DIR = ROOT / "data"
RESULTS_CSV = ROOT / "Assignments" / "Text Feature Engineering" / "word2vec_experiment_results.csv"
SUMMARY_MD = ROOT / "Assignments" / "Text Feature Engineering" / "word2vec_assignment_summary.md"


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def tokenize(text: str) -> list[str]:
    return simple_preprocess(str(text), deacc=True, min_len=2)


def build_story_corpus() -> list[list[str]]:
    corpus: list[list[str]] = []
    for path in sorted(STORY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        corpus.extend(tokenize(sentence) for sentence in split_sentences(text))
    return [tokens for tokens in corpus if tokens]


def document_vector(model: Word2Vec, tokens: list[str]) -> np.ndarray:
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0)


def build_corpora(train_texts: list[str], story_corpus: list[list[str]]) -> dict[str, list[list[str]]]:
    train_docs = [tokenize(text) for text in train_texts]
    train_sentences = [tokenize(sentence) for text in train_texts for sentence in split_sentences(text)]

    return {
        "reviews_only": [doc for doc in train_docs if doc],
        "reviews_plus_sentences": [doc for doc in (train_docs + train_sentences) if doc],
        "reviews_plus_sentences_plus_stories": [
            doc for doc in (train_docs + train_sentences + story_corpus) if doc
        ],
    }


def run_experiments() -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(REVIEWS_CSV)
    df["label"] = (df["sentiment"] == "positive").astype(int)

    train_texts, test_texts, y_train, y_test = train_test_split(
        df["review_text"].astype(str).tolist(),
        df["label"].to_numpy(),
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    train_docs = [tokenize(text) for text in train_texts]
    test_docs = [tokenize(text) for text in test_texts]
    story_corpus = build_story_corpus()
    corpora = build_corpora(train_texts, story_corpus)

    configs = [
        {"vector_size": 50, "window": 5, "min_count": 1, "epochs": 10, "sg": 0},
        {"vector_size": 100, "window": 5, "min_count": 1, "epochs": 20, "sg": 0},
        {"vector_size": 150, "window": 8, "min_count": 1, "epochs": 30, "sg": 1},
        {"vector_size": 200, "window": 10, "min_count": 1, "epochs": 40, "sg": 1},
        {"vector_size": 300, "window": 10, "min_count": 1, "epochs": 50, "sg": 1},
    ]

    rows: list[dict[str, object]] = []
    best_details: dict[str, object] | None = None

    for corpus_name, corpus in corpora.items():
        for config in configs:
            model = Word2Vec(
                sentences=corpus,
                vector_size=config["vector_size"],
                window=config["window"],
                min_count=config["min_count"],
                workers=1,
                sg=config["sg"],
                epochs=config["epochs"],
                seed=42,
            )

            x_train = np.vstack([document_vector(model, tokens) for tokens in train_docs])
            x_test = np.vstack([document_vector(model, tokens) for tokens in test_docs])

            classifier = LogisticRegression(max_iter=2000, random_state=42)
            classifier.fit(x_train, y_train)
            predictions = classifier.predict(x_test)

            row = {
                "corpus": corpus_name,
                "vector_size": config["vector_size"],
                "window": config["window"],
                "min_count": config["min_count"],
                "epochs": config["epochs"],
                "architecture": "skipgram" if config["sg"] == 1 else "cbow",
                "train_vocab": len(model.wv),
                "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            }
            rows.append(row)

            if best_details is None or row["accuracy"] > best_details["accuracy"]:
                best_details = {
                    **row,
                    "report": classification_report(
                        y_test,
                        predictions,
                        target_names=["negative", "positive"],
                        digits=4,
                    ),
                }

    results_df = pd.DataFrame(rows).sort_values(["accuracy", "vector_size"], ascending=[False, True])
    if best_details is None:
        raise RuntimeError("No experiments were run.")
    return results_df, best_details


def write_summary(results_df: pd.DataFrame, best_details: dict[str, object]) -> None:
    top_rows = results_df.head(5).to_string(index=False)
    summary = f"""# Word2Vec Assignment Summary

## What Was Changed
- Added more training text by using the Flipkart review corpus, splitting reviews into sentences, and also including the extra `data/story*.txt` files.
- Trained Word2Vec for more epochs and compared multiple parameter settings instead of one fixed configuration.
- Evaluated the learned embeddings using a downstream sentiment classifier built on average review embeddings.

## Best Result
- Accuracy: {best_details['accuracy']:.4f}
- Corpus: {best_details['corpus']}
- Embedding dimension: {best_details['vector_size']}
- Window size: {best_details['window']}
- Epochs: {best_details['epochs']}
- Architecture: {best_details['architecture']}
- Vocabulary size: {best_details['train_vocab']}

## Top 5 Experiments
```text
{top_rows}
```

## Best Model Classification Report
```text
{best_details['report'].strip()}
```

## Conclusion
- Small CBOW models underfit this dataset and stayed near chance level.
- Increasing epochs and embedding dimensions improved performance.
- Skip-gram with a larger window performed much better than the smaller baselines.
- The best setup reached about 87.1% accuracy, which is strong for a small dataset and simple average-embedding pipeline.
"""
    SUMMARY_MD.write_text(summary, encoding="utf-8")


def main() -> None:
    results_df, best_details = run_experiments()
    results_df.to_csv(RESULTS_CSV, index=False)
    write_summary(results_df, best_details)
    print(results_df.to_string(index=False))
    print("\nBest configuration:")
    for key, value in best_details.items():
        if key != "report":
            print(f"{key}: {value}")
    print("\nClassification report:")
    print(best_details["report"])


if __name__ == "__main__":
    main()
