# ============================================================
# Sentiment Analysis on IMDB — 50,000 Reviews
# ============================================================



# ===== STEP 1: Imports =====
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model, metrics, preprocessing, model_selection
import matplotlib.pyplot as plt
import numpy as np

print("✅ Libraries imported!")


# ===== STEP 2: Load the Data =====
print("\n⏳ Uploading dataset...")

dataset = load_dataset("imdb")

#print(dataset)  # shows the structure of the dataset

# train/test split
train_data = dataset["train"]  # 25,000 reviews
test_data  = dataset["test"]   # 25,000 reviews

# lists of reviews and labels
X_train = train_data["text"]
y_train = train_data["label"]  # 1 = Positive, 0 = Negative

X_test = test_data["text"]
y_test = test_data["label"]

print(f"✅ Dataset loaded!")
print(f"   Training: {len(X_train):,} reviews")
print(f"   Testing:  {len(X_test):,} reviews")

# Show a sample review
print(f"\n📝 Sample review:")
print(f"   Text:  {X_train[0][:100]}...")
print(f"   Label: {'Positive 😊' if y_train[0] == 1 else 'Negative 😞'}")


# ===== STEP 3: TF-IDF — Convert Text to Numbers =====
print("\n⏳ Converting texts to numbers...")

vectorizer = TfidfVectorizer(
    max_features=10000,   # keep only the top 10,000 most important words
    stop_words='english', # remove common words like "the, is, a"
    ngram_range=(1, 2),   # use single words + every pair of consecutive words
                          # e.g. "not good" is treated as one feature, not two
    min_df=5              # ignore words that appear fewer than 5 times
)

X_train_tfidf = vectorizer.fit_transform(X_train)  # learn vocabulary + transform
X_test_tfidf  = vectorizer.transform(X_test)       # transform only (no learning)

print(f"✅ Conversion done!")
print(f"   Training Matrix shape: {X_train_tfidf.shape}")
print(f"   Meaning: {X_train_tfidf.shape[0]:,} sentences × {X_train_tfidf.shape[1]:,} features")


# ===== STEP 4: Train the Model =====
print("\n⏳ Training... (approximately 2 minutes)")

model = linear_model.LogisticRegression(
    max_iter=1000, # number of learning iterations
    C=1.0,         # regularization strength (lower = stronger regularization)
    solver='lbfgs',# mathematical solver method
    n_jobs=-1      # use all CPU cores for faster training
)

model.fit(X_train_tfidf, y_train)
print("✅ Model trained successfully!")


# ===== STEP 5: Evaluation =====
print("\n📊 Evaluating the model...")

y_pred   = model.predict(X_test_tfidf)
accuracy = metrics.accuracy_score(y_test, y_pred)

print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")
print("\n📋 Detailed Report:")
print(metrics.classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))


# ===== STEP 6: Confusion Matrix =====
cm = metrics.confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Greens')

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Negative', 'Positive'], fontsize=12)
ax.set_yticklabels(['Negative', 'Positive'], fontsize=12)
ax.set_xlabel('Predicted', fontsize=13)
ax.set_ylabel('Actual', fontsize=13)
ax.set_title('Confusion Matrix', fontsize=15, fontweight='bold')

for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i,j]:,}',
                ha='center', va='center',
                fontsize=14, fontweight='bold',
                color='white' if cm[i,j] > cm.max()/2 else 'black')

plt.colorbar(im)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("✅ Confusion Matrix saved!")


# ===== STEP 7: Most Important Words per Category =====
print("\n🔑 Top 15 keywords:")

feature_names  = vectorizer.get_feature_names_out()
coefficients   = model.coef_[0]

# Highest-scoring Positive words
top_positive_idx = np.argsort(coefficients)[-15:][::-1]
top_negative_idx = np.argsort(coefficients)[:15]

print("\n😊 Positive keywords:")
for idx in top_positive_idx:
    print(f"   {feature_names[idx]:<20} {coefficients[idx]:.3f}")

print("\n😞 Negative keywords:")
for idx in top_negative_idx:
    print(f"   {feature_names[idx]:<20} {coefficients[idx]:.3f}")


# ===== STEP 8: Try it yourself! =====
def predict_sentiment(text):
    """
    Enter any English sentence and the model will tell you
    whether it is Positive or Negative.
    """
    text_tfidf  = vectorizer.transform([text])
    prediction  = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    confidence  = max(probability) * 100

    label = "😊 Positive" if prediction == 1 else "😞 Negative"
    print(f"\nText:       '{text}'")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.1f}%")
    print("-" * 50)

# ===== Test examples =====
predict_sentiment("This movie was absolutely fantastic! Best film of the year.")
predict_sentiment("Terrible movie, complete waste of time and money.")
predict_sentiment("It was okay, nothing special but not bad either.")
predict_sentiment("The acting was great but the story was disappointing.")
predict_sentiment("I loved my family very much.")