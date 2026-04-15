# Word2Vec Assignment Summary

## Aim
The goal of this assignment was to improve the Word2Vec model by:
- adding more data
- training for more epochs
- checking accuracy
- testing different embedding dimensions
- trying different parameters

## What I Did
I used the `flipkart_reviews.csv` dataset.

To add more data, I:
- used all review texts
- split reviews into smaller sentences
- also used the story files inside the `data` folder

Then I trained Word2Vec with different values of:
- embedding dimension: 50, 100, 150, 200, 300
- epochs: 10, 20, 30, 40, 50
- window size: 5, 8, 10
- model type: CBOW and Skip-gram

After that, I checked the accuracy using a simple sentiment classification model.

## Best Result
- Accuracy: `87.10%`
- Embedding dimension: `300`
- Epochs: `50`
- Window size: `10`
- Model type: `Skip-gram`

## Simple Observations
- Small dimensions like 50 and 100 gave low accuracy.
- Increasing the embedding dimension improved the result.
- Training for more epochs also improved the model.
- Skip-gram performed better than CBOW.

## Final Conclusion
The model improved when I:
- added more data
- increased the number of epochs
- used a larger embedding dimension

Best parameters:
- `vector_size = 300`
- `epochs = 50`
- `window = 10`
- `sg = 1` which means Skip-gram

Final accuracy:
- `87.10%`
