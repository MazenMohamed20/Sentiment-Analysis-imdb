# 🎬 Sentiment Analysis — IMDB Movie Reviews

A machine learning model that classifies movie reviews as **Positive** or **Negative** using NLP techniques.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | ~90% |
| Dataset | IMDB (50,000 reviews) |
| Model | Logistic Regression |

---

## 🛠️ Tech Stack

- **Python 3.x**
- **scikit-learn** — Model & TF-IDF
- **HuggingFace Datasets** — IMDB Data
- **Matplotlib** — Visualization

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-imdb.git
cd sentiment-analysis-imdb
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the model**
```bash
python app.py
```

---

## 🔍 How It Works

```
Raw Text → TF-IDF Vectorizer → Logistic Regression → Positive / Negative
```

1. **TF-IDF** converts text into numerical features
2. **Logistic Regression** learns patterns from 25,000 training reviews
3. Model predicts sentiment on 25,000 unseen test reviews

---

## 📁 Project Structure

```
sentiment-analysis-imdb/
│
├── app.py # Main model code
├── requirements.txt # Dependencies
└── README.md # You are here
```

---

## 💬 Example Predictions

```python
predict_sentiment("This movie was absolutely fantastic!")
# 😊 Positive — 94.2%

predict_sentiment("Terrible movie, complete waste of time.")
# 😞 Negative — 97.1%
```

---

## 👤 Author

Made by **Mazen Mohamed Fayez**
- GitHub: [@MazenMohamed20](https://github.com/MazenMohamed20)

---

⭐ If you found this useful, give it a star!
