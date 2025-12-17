# IMDb Movie Review Sentiment Classification

## 📌 Project Overview
This project builds a neural network to classify IMDb movie reviews as **positive** or **negative**.  
It uses **TF-IDF** for text vectorization and a **dense neural network** for classification.

The notebook is designed to run in **Google Colab** or **locally in VS Code**.

---

## 🛠 Tools & Libraries
- Python 3
- Pandas
- Numpy
- Matplotlib & Seaborn (visualization)
- NLTK (text preprocessing)
- Scikit-learn (TF-IDF, train/test split, metrics)
- TensorFlow / Keras (neural network)

---

## 💾 Dataset
- File: `IMDB Dataset.csv`
- Contains **50,000 movie reviews** with corresponding sentiments (`positive`/`negative`).

**Columns:**
1. `review` → text of the movie review
2. `sentiment` → label (positive or negative)

**Source:** [IMDb Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## 🚀 Steps
1. **Install dependencies** (`pip install pandas scikit-learn tensorflow nltk matplotlib seaborn wordcloud`)
2. **Load dataset**
   - In Colab: upload `IMDB Dataset.csv` using `files.upload()`
   - Locally: place the CSV in your project folder
3. **Data exploration**
   - Check sentiment distribution
   - Explore review lengths
4. **Preprocessing**
   - Lowercase conversion
   - Remove HTML tags & URLs
   - Tokenization & stopword removal
5. **TF-IDF vectorization**
6. **Train/test split**
7. **Build neural network**
   - Dense layers + Dropout
   - Sigmoid output for binary classification
8. **Train the model**
   - Use early stopping and learning rate scheduler
9. **Evaluate model**
   - Accuracy, classification report, confusion matrix, ROC curve
10. **Make predictions** on new reviews

---

## 📊 Results
- Test Accuracy: ~0.87 (depends on preprocessing and hyperparameters)
- Confusion matrix and ROC curve provide insights into performance.

---

## ⚡ Usage Example

```python
# Predict sentiment for a new review
sample = "The movie was fantastic and thrilling!"
label, prob = predict_review(sample)
print(f"Predicted sentiment: {label}, probability: {prob:.4f}")
