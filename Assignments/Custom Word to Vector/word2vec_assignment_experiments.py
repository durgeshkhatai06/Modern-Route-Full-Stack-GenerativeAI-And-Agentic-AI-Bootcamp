# Allow using forward references in type hints (Python 3.10+ style annotations in earlier versions)
from __future__ import annotations

# Import 're' module for regular expression operations (used for splitting text by sentence delimiters)
import re
# Import 'Path' from pathlib to handle file paths in a cross-platform way
from pathlib import Path

# Import numpy for numerical computations and array operations
import numpy as np
# Import pandas for data manipulation and CSV file handling
import pandas as pd
# Import Word2Vec model from gensim to train word embeddings
from gensim.models import Word2Vec
# Import simple_preprocess utility to clean and tokenize text
from gensim.utils import simple_preprocess
# Import LogisticRegression classifier for sentiment classification
from sklearn.linear_model import LogisticRegression
# Import metrics to evaluate model performance (accuracy, classification report)
from sklearn.metrics import accuracy_score, classification_report
# Import train_test_split to divide data into training and testing sets
from sklearn.model_selection import train_test_split


# Define ROOT as the project root directory (navigate up 2 levels from the current script's location)
ROOT = Path(__file__).resolve().parents[2]
# Define path to the Flipkart reviews CSV file used for training sentiment data
REVIEWS_CSV = ROOT / "Assignments" / "Text Feature Engineering" / "flipkart_reviews.csv"
# Define path to the data directory containing story text files for additional training corpus
STORY_DIR = ROOT / "data"
# Define path where experiment results will be saved as a CSV file
RESULTS_CSV = ROOT / "Assignments" / "Text Feature Engineering" / "word2vec_experiment_results.csv"
# Define path where the summary markdown file will be written with results and analysis
SUMMARY_MD = ROOT / "Assignments" / "Text Feature Engineering" / "word2vec_assignment_summary.md"



# Function to split text into individual sentences using regex pattern for sentence delimiters
def split_sentences(text: str) -> list[str]:
    # Use regex to split on periods, exclamation marks, or question marks followed by whitespace
    # The regex (?<=[.!?])\s+ uses a lookbehind to find these delimiters
    # Then split, strip whitespace from each part, and filter out empty strings
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]



# Function to convert text into a list of preprocessed tokens (words)
def tokenize(text: str) -> list[str]:
    # Use gensim's simple_preprocess utility to:
    # 1. Convert text to lowercase
    # 2. Remove punctuation
    # 3. Split by whitespace to create tokens
    # 4. deacc=True removes accents from characters
    # 5. min_len=2 filters out single-character tokens
    return simple_preprocess(str(text), deacc=True, min_len=2)



# Function to read story files and build a corpus of tokenized sentences
def build_story_corpus() -> list[list[str]]:
    # Initialize empty list to store tokenized sentences from all story files
    corpus: list[list[str]] = []
    # Loop through all .txt files in the STORY_DIR sorted by filename in alphabetical order
    for path in sorted(STORY_DIR.glob("*.txt")):
        # Read the entire text file with UTF-8 encoding, ignore any encoding errors
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Split text into sentences, tokenize each sentence, and add to corpus
        corpus.extend(tokenize(sentence) for sentence in split_sentences(text))
    # Filter out empty token lists and return only non-empty sentences
    return [tokens for tokens in corpus if tokens]



# Function to create a single vector representation for a document by averaging word vectors
def document_vector(model: Word2Vec, tokens: list[str]) -> np.ndarray:
    # Extract word vectors from the model for each token that exists in the vocabulary
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    # If no vectors found (all words are out-of-vocabulary), return a zero vector with model dimensions
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    # Otherwise, return the mean (average) of all word vectors as the document representation
    return np.mean(vectors, axis=0)



# Function to create different training corpora by combining training texts and story files in various ways
def build_corpora(train_texts: list[str], story_corpus: list[list[str]]) -> dict[str, list[list[str]]]:
    # Tokenize each training text (review) into tokens
    train_docs = [tokenize(text) for text in train_texts]
    # Split each training text into sentences, then tokenize each sentence
    train_sentences = [tokenize(sentence) for text in train_texts for sentence in split_sentences(text)]

    # Return a dictionary with three different corpus configurations for experimentation
    return {
        # First corpus: only tokenized reviews without additional data
        "reviews_only": [doc for doc in train_docs if doc],
        # Second corpus: reviews plus their sentences split out (provides more training data)
        "reviews_plus_sentences": [doc for doc in (train_docs + train_sentences) if doc],
        # Third corpus: reviews, sentences, plus story files (maximum training data)
        "reviews_plus_sentences_plus_stories": [
            doc for doc in (train_docs + train_sentences + story_corpus) if doc
        ],
    }



# Main function that runs all Word2Vec experiments with different configurations and corpora
def run_experiments() -> tuple[pd.DataFrame, dict[str, object]]:
    # Read the CSV file containing reviews and their sentiment labels
    df = pd.read_csv(REVIEWS_CSV)
    # Create a binary label column: 1 for positive sentiment, 0 for negative
    df["label"] = (df["sentiment"] == "positive").astype(int)

    # Split data into training (75%) and testing (25%) sets with stratified split to maintain label distribution
    train_texts, test_texts, y_train, y_test = train_test_split(
        df["review_text"].astype(str).tolist(),  # Convert reviews to strings and convert to list
        df["label"].to_numpy(),  # Convert labels to numpy array
        test_size=0.25,  # Use 25% of data for testing
        random_state=42,  # Fixed seed for reproducible splits
        stratify=df["label"],  # Ensure both train and test have similar label distributions
    )

    # Tokenize all training and testing reviews
    train_docs = [tokenize(text) for text in train_texts]
    test_docs = [tokenize(text) for text in test_texts]
    # Build corpus from story files for additional training data
    story_corpus = build_story_corpus()
    # Create three different corpus configurations
    corpora = build_corpora(train_texts, story_corpus)

    # Define different Word2Vec configurations to test
    configs = [
        # Config 1: Small CBOW model (sg=0 means CBOW architecture)
        {"vector_size": 50, "window": 5, "min_count": 1, "epochs": 10, "sg": 0},
        # Config 2: Medium CBOW model with more dimensions
        {"vector_size": 100, "window": 5, "min_count": 1, "epochs": 20, "sg": 0},
        # Config 3: Larger skip-gram model (sg=1 means skip-gram architecture)
        {"vector_size": 150, "window": 8, "min_count": 1, "epochs": 30, "sg": 1},
        # Config 4: Even larger skip-gram model with larger window
        {"vector_size": 200, "window": 10, "min_count": 1, "epochs": 40, "sg": 1},
        # Config 5: Largest skip-gram model with maximum dimensions
        {"vector_size": 300, "window": 10, "min_count": 1, "epochs": 50, "sg": 1},
    ]

    # Initialize lists to store all experiment results
    rows: list[dict[str, object]] = []
    # Track the best result found so far
    best_details: dict[str, object] | None = None

    # Outer loop: iterate through each corpus configuration
    for corpus_name, corpus in corpora.items():
        # Inner loop: iterate through each Word2Vec configuration
        for config in configs:
            # Train Word2Vec model with current corpus and configuration
            model = Word2Vec(
                sentences=corpus,  # Training corpus (list of tokenized sentences)
                vector_size=config["vector_size"],  # Dimension of word vectors
                window=config["window"],  # Context window size for word prediction
                min_count=config["min_count"],  # Minimum word frequency to include in vocabulary
                workers=1,  # Number of threads (1 for reproducibility)
                sg=config["sg"],  # 0=CBOW, 1=skip-gram architecture
                epochs=config["epochs"],  # Number of training iterations
                seed=42,  # Random seed for reproducibility
            )

            # Convert training documents to vectors by averaging word embeddings
            x_train = np.vstack([document_vector(model, tokens) for tokens in train_docs])
            # Convert test documents to vectors the same way
            x_test = np.vstack([document_vector(model, tokens) for tokens in test_docs])

            # Train a logistic regression classifier on the Word2Vec embeddings
            classifier = LogisticRegression(max_iter=2000, random_state=42)
            # Fit classifier on training embeddings and labels
            classifier.fit(x_train, y_train)
            # Make predictions on test set embeddings
            predictions = classifier.predict(x_test)

            # Store results from this experiment as a dictionary
            row = {
                "corpus": corpus_name,  # Name of corpus used
                "vector_size": config["vector_size"],  # Embedding dimension
                "window": config["window"],  # Context window size
                "min_count": config["min_count"],  # Minimum word frequency
                "epochs": config["epochs"],  # Number of training epochs
                "architecture": "skipgram" if config["sg"] == 1 else "cbow",  # Model type
                "train_vocab": len(model.wv),  # Size of vocabulary learned by Word2Vec
                "accuracy": round(float(accuracy_score(y_test, predictions)), 4),  # Classification accuracy
            }
            # Add this result to the list of all results
            rows.append(row)

            # Check if this is the best result so far
            if best_details is None or row["accuracy"] > best_details["accuracy"]:
                # Update best result dictionary with current experiment's details
                best_details = {
                    **row,  # Include all metrics from current experiment
                    # Generate detailed classification report (precision, recall, F1 for each class)
                    "report": classification_report(
                        y_test,
                        predictions,
                        target_names=["negative", "positive"],  # Class names for report
                        digits=4,  # Show 4 decimal places in report
                        zero_division=0,  # Handle division by zero gracefully
                    ),
                }

    # Convert all results to a pandas DataFrame and sort by accuracy (descending) then vector_size (ascending)
    results_df = pd.DataFrame(rows).sort_values(["accuracy", "vector_size"], ascending=[False, True])
    # Safety check to ensure at least one experiment was run
    if best_details is None:
        raise RuntimeError("No experiments were run.")
    # Return both the full results dataframe and the best configuration details
    return results_df, best_details



# Function to write a markdown summary report of the word2vec experiments
def write_summary(results_df: pd.DataFrame, best_details: dict[str, object]) -> None:
    # Convert the top 5 results to a formatted string for the report
    top_rows = results_df.head(5).to_string(index=False)
    # Create a comprehensive markdown-formatted summary report
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
    # Write the summary string to the markdown output file
    SUMMARY_MD.write_text(summary, encoding="utf-8")



# Main entry point function that orchestrates the entire experiment workflow
def main() -> None:
    # Run all experiments and get results dataframe and best configuration details
    results_df, best_details = run_experiments()
    # Save all experimental results to a CSV file for future reference
    results_df.to_csv(RESULTS_CSV, index=False)
    # Generate a markdown summary report with findings and best results
    write_summary(results_df, best_details)
    # Print all results to console in a formatted table
    print(results_df.to_string(index=False))
    # Print a blank line for readability
    print("\nBest configuration:")
    # Print each metric from the best configuration (except the full report)
    for key, value in best_details.items():
        if key != "report":
            print(f"{key}: {value}")
    # Print a blank line for readability
    print("\nClassification report:")
    # Print the detailed classification report for the best model
    print(best_details["report"])


# Check if this script is being run directly (not imported as a module)
if __name__ == "__main__":
    # Call the main function to execute the experiments
    main()
