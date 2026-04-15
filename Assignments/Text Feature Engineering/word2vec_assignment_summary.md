# Word2Vec Assignment Summary

## What Was Changed
- Added more training text by using the Flipkart review corpus, splitting reviews into sentences, and also including the extra `data/story*.txt` files.
- Trained Word2Vec for more epochs and compared multiple parameter settings instead of one fixed configuration.
- Evaluated the learned embeddings using a downstream sentiment classifier built on average review embeddings.

## Best Result
- Accuracy: 0.8710
- Corpus: reviews_plus_sentences
- Embedding dimension: 300
- Window size: 10
- Epochs: 50
- Architecture: skipgram
- Vocabulary size: 463

## Top 5 Experiments
```text
                             corpus  vector_size  window  min_count  epochs architecture  train_vocab  accuracy
reviews_plus_sentences_plus_stories          200      10          1      40     skipgram          665    0.8710
             reviews_plus_sentences          300      10          1      50     skipgram          463    0.8710
reviews_plus_sentences_plus_stories          300      10          1      50     skipgram          665    0.8710
             reviews_plus_sentences          200      10          1      40     skipgram          463    0.8387
             reviews_plus_sentences          150       8          1      30     skipgram          463    0.7419
```

## Best Model Classification Report
```text
precision    recall  f1-score   support

    negative     0.9231    0.8000    0.8571        15
    positive     0.8333    0.9375    0.8824        16

    accuracy                         0.8710        31
   macro avg     0.8782    0.8688    0.8697        31
weighted avg     0.8768    0.8710    0.8702        31
```

## Conclusion
- Small CBOW models underfit this dataset and stayed near chance level.
- Increasing epochs and embedding dimensions improved performance.
- Skip-gram with a larger window performed much better than the smaller baselines.
- The best setup reached about 87.1% accuracy, which is strong for a small dataset and simple average-embedding pipeline.
