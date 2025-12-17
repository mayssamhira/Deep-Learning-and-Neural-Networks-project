# IMDb Movie Review Sentiment Analysis

Classify IMDb movie reviews as **positive** or **negative** using a neural network with TF-IDF features.

---

## Tools & Libraries
- Python, Pandas, Numpy  
- NLTK (text preprocessing)  
- Scikit-learn (TF-IDF, metrics)  
- TensorFlow / Keras (neural network)  
- Matplotlib & Seaborn (visualization)

---

## Dataset
- `IMDB Dataset.csv` (50,000 reviews, labels: positive/negative)  
- Source: [Kaggle IMDb Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## Steps
1. Load and explore dataset  
2. Clean text (lowercase, remove HTML/URLs, tokenize, remove stopwords)  
3. TF-IDF vectorization  
4. Train/test split  
5. Build neural network (Dense + Dropout layers)  
6. Train with early stopping  
7. Evaluate (accuracy, confusion matrix, ROC)  
8. Predict on new reviews

---

## Example

```python
sample = "This movie was amazing!"
label, prob = predict_review(sample)
print(f"Sentiment: {label}, probability: {prob:.4f}")



👩‍💻 Author

Mayssa Mhira
