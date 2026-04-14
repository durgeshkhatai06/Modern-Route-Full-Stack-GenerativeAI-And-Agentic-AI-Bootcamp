# Text Feature Engineering Assignment — Observations & Conclusions
### Dataset: Flipkart Product Reviews | 124 Reviews | Python + scikit-learn

---

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| Total reviews | 124 |
| Positive reviews | 64 (51.6%) |
| Negative reviews | 60 (48.4%) |
| Source | Flipkart-style product reviews (electronics / gadgets) |

The dataset was collected with balanced sentiment distribution to avoid class bias in classification.

---

## 2. Preprocessing Observations

Preprocessing pipeline applied in order:

1. **Lowercase conversion** — eliminates case-based duplicate tokens (e.g., "Product" vs "product")
2. **Tokenization** — `re.findall(r'\b[a-z]+\b', text)` cleanly extracts only alphabetic tokens
3. **Punctuation removal** — handled implicitly by the regex above
4. **Stopword removal** — removed 49 common English stopwords; reduced noise significantly
5. **Lemmatization** (rule-based) — mapped ~40 common inflections (e.g., "delivered" → "deliver")

**Key observation:** After preprocessing, vocabulary shrank from ~600 raw tokens to **472 meaningful tokens**, improving signal-to-noise ratio for all downstream tasks.

---

## 3. Vocabulary Analysis

- **Vocabulary size:** 472 unique tokens
- **Top 5 words:** `product` (93), `quality` (52), `good` (48), `work` (44), `deliver` (38)
- **Observation:** High-frequency words span both sentiments ("product" appears in all reviews). This demonstrates why raw frequency alone (BoW) is insufficient — TF-IDF is needed to suppress such globally common terms.

---

## 4. Feature Engineering Comparison

| Aspect | OHE | Bag of Words | TF-IDF |
|--------|-----|--------------|--------|
| Value type | Binary (0/1) | Integer count | Float (0.0–1.0) |
| Captures frequency? | ❌ No | ✅ Yes | ✅ Yes (via TF) |
| Penalizes common words? | ❌ No | ❌ No | ✅ Yes (via IDF) |
| Matrix shape | 124 × 472 | 124 × 472 | 124 × 472 |
| Sparsity | 98.1% | 98.1% | 98.1% |
| Best for | Presence detection | Short texts, spam | Search, keyword extraction |

**Key observation on TF-IDF weighting:**
The word `"product"` appears in 93/124 documents → IDF ≈ 1.25 (low weight).
The word `"outstanding"` appears in only 6/124 documents → IDF ≈ 3.77 (high weight).
This correctly reflects that "outstanding" is a stronger signal for positive sentiment than the generic word "product."

---

## 5. Sparse Matrix Analysis

All three matrices achieved **98.1% sparsity** — meaning only ~1.9% of entries are non-zero.

**Why sparsity is a problem at scale:**
- A realistic corpus of 1 million documents with a 100,000-word vocabulary would require **800 GB** as a dense float64 matrix
- Sparse CSR storage reduces this to manageable sizes, but matrix operations (dot products, cosine similarity) become cache-inefficient
- Linear algebra libraries optimised for dense matrices cannot be used directly

**Solution in industry:** Replace sparse BoW/TF-IDF vectors with dense word embeddings (Word2Vec: 300 dims, BERT: 768 dims), which capture semantics in a compact, cache-friendly format.

---

## 6. Sentiment Classification Results

| Model | Features | Accuracy | F1 (Positive) |
|-------|----------|----------|---------------|
| Logistic Regression | BoW | 84.00% | 84.2% |
| Logistic Regression | TF-IDF | 84.00% | 84.2% |
| **Naive Bayes** | **BoW** | **88.00%** | **88.1%** |
| **Naive Bayes** | **TF-IDF** | **88.00%** | **87.8%** |

**Key observations:**

1. **Naive Bayes outperformed Logistic Regression** on this dataset. This is expected on small datasets — NB's probabilistic word-frequency model is well-matched to text data and requires less training data to converge.

2. **TF-IDF did not significantly outperform BoW** on this particular task. This is common for **sentiment classification** because the presence/frequency of opinion words ("excellent", "terrible") matters more than their corpus-level rarity. TF-IDF's strength shows more in **information retrieval** tasks.

3. **Top positive features (LR + TF-IDF):** excellent, outstanding, superb, brilliant, fantastic — semantically strong positive adjectives
   **Top negative features:** terrible, disappoint, pathetic, useless, worst — strong negative descriptors

---

## 7. Answers to Real-world Questions (Summary)

**Why BoW fails at semantics:**
BoW cannot recognise that "excellent" and "outstanding" mean the same thing. Each word gets a separate column with no notion of proximity in meaning. Negation ("not good") is also mishandled.

**When to use BoW vs TF-IDF:**
- **BoW:** Spam filters, Naive Bayes classifiers, very short text, quick baselines
- **TF-IDF:** Search engines, keyword extraction, document clustering, recommendation systems

**Limitations of TF-IDF:**
No semantic understanding, ignores word order, corpus-dependent IDF, poor handling of OOV words, ineffective on very short text, no negation awareness.

---

## 8. Conclusions

1. Preprocessing is the most impactful step — proper tokenisation and stopword removal directly determines model quality.
2. For sentiment classification on product reviews, **Naive Bayes with BoW** is a surprisingly competitive baseline (88% accuracy) and should always be tried first before heavier models.
3. TF-IDF's main advantage over BoW is **interpretability** and **keyword extraction**, not always classification accuracy.
4. Sparse matrices work for small datasets but are fundamentally unscalable — dense embeddings (Word2Vec, BERT) are the industry standard for corpora above 100K documents.
5. The next natural step would be to apply **BERT-based sentence transformers** for dense, semantically-aware embeddings and compare against these baselines.

---
*Assignment completed using Python 3, scikit-learn 1.8, pandas 3.0, numpy 2.4, matplotlib*
