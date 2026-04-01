# ============================================================
#  CSC 309 Mini Project #3 — Spam Email Detector
#  Concepts : Supervised Learning, Text Classification
#  Tools     : Scikit-learn
#  Algorithms: Naive Bayes, Logistic Regression
# ============================================================

import os
import re
import pickle
import urllib.request

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  STEP 0 — Download NLTK data (first run only)
# ─────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)


# ─────────────────────────────────────────────
#  STEP 1 — Load / Download Dataset
# ─────────────────────────────────────────────
DATASET_URL = (
    "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
)
DATASET_PATH = "emails.csv"


def load_data():
    """Load the SMS Spam Collection dataset.
    Downloads it automatically if not present locally."""
    if not os.path.exists(DATASET_PATH):
        print("📥 Downloading dataset …")
        urllib.request.urlretrieve(DATASET_URL, "sms_raw.tsv")
        df = pd.read_csv("sms_raw.tsv", sep="\t", header=None, names=["label", "message"])
        df.to_csv(DATASET_PATH, index=False)
        print("✅ Dataset saved to emails.csv")
    else:
        df = pd.read_csv(DATASET_PATH)

    print(f"\n📊 Dataset loaded — {len(df)} rows")
    print(df["label"].value_counts())
    return df


# ─────────────────────────────────────────────
#  STEP 2 — Preprocessing
# ─────────────────────────────────────────────
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    """Clean and normalize a single email/SMS text."""
    text = text.lower()                          # lowercase
    text = re.sub(r"http\S+|www\S+", " ", text)  # remove URLs
    text = re.sub(r"\W", " ", text)              # remove special chars
    text = re.sub(r"\s+", " ", text).strip()     # collapse whitespace
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)


# ─────────────────────────────────────────────
#  STEP 3 — Feature Extraction (TF-IDF)
# ─────────────────────────────────────────────
def extract_features(df: pd.DataFrame):
    """Preprocess text and vectorize with TF-IDF."""
    print("\n🔧 Preprocessing text …")
    df["clean_text"] = df["message"].apply(preprocess)

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["clean_text"]).toarray()
    y = df["label"].map({"spam": 1, "ham": 0}).values

    print(f"✅ Feature matrix shape: {X.shape}")
    return X, y, vectorizer


# ─────────────────────────────────────────────
#  STEP 4 — Train & Evaluate Both Models
# ─────────────────────────────────────────────
def train_and_evaluate(X, y):
    """Train Naive Bayes and Logistic Regression, print metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
    }

    results = {}
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"  🤖  {name}")
        print("=" * 50)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

        print(f"  Test Accuracy    : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"  CV Accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print()
        print(classification_report(y_test, preds, target_names=["Ham", "Spam"]))

        results[name] = {
            "model": model,
            "accuracy": acc,
            "cv_mean": cv_scores.mean(),
            "preds": preds,
            "y_test": y_test,
        }

    return results, X_test, y_test


# ─────────────────────────────────────────────
#  STEP 5 — Visualisations
# ─────────────────────────────────────────────
def plot_results(results):
    """Plot accuracy comparison and confusion matrices side by side."""
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] * 100 for n in names]
    cv_means = [results[n]["cv_mean"] * 100 for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("CSC 309 — Spam Email Detector Results", fontsize=14, fontweight="bold")

    # Bar chart — accuracy comparison
    x = np.arange(len(names))
    width = 0.35
    axes[0].bar(x - width / 2, accuracies, width, label="Test Acc", color="#4C72B0")
    axes[0].bar(x + width / 2, cv_means, width, label="CV Acc (5-fold)", color="#DD8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=10)
    axes[0].set_ylim(90, 100)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Model Accuracy Comparison")
    axes[0].legend()
    for bar in axes[0].patches:
        axes[0].annotate(
            f"{bar.get_height():.2f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=8,
        )

    # Confusion matrices
    for ax, name in zip(axes[1:], names):
        cm = confusion_matrix(results[name]["y_test"], results[name]["preds"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix\n{name}")

    plt.tight_layout()
    plt.savefig("results.png", dpi=150, bbox_inches="tight")
    print("\n📈 Chart saved → results.png")
    plt.show()


# ─────────────────────────────────────────────
#  STEP 6 — Save Best Model
# ─────────────────────────────────────────────
def save_best_model(results, vectorizer):
    """Pickle the model with the highest test accuracy."""
    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_model = results[best_name]["model"]

    pickle.dump(best_model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    print(f"\n💾 Best model saved  → model.pkl  ({best_name})")
    print("💾 Vectorizer saved  → vectorizer.pkl")
    return best_name


# ─────────────────────────────────────────────
#  STEP 7 — Prediction Function
# ─────────────────────────────────────────────
def predict_email(text: str) -> str:
    """Predict whether a given email/SMS text is spam or ham."""
    model = pickle.load(open("model.pkl", "rb"))
    vec = pickle.load(open("vectorizer.pkl", "rb"))

    cleaned = preprocess(text)
    features = vec.transform([cleaned]).toarray()
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    label = "🚨 SPAM" if prediction == 1 else "✅ HAM (Not Spam)"
    confidence = proba[prediction] * 100
    return f"{label}  —  Confidence: {confidence:.1f}%"


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load data
    df = load_data()

    # 2. Extract features
    X, y, vectorizer = extract_features(df)

    # 3. Train & evaluate
    results, X_test, y_test = train_and_evaluate(X, y)

    # 4. Plot results
    plot_results(results)

    # 5. Save best model
    best = save_best_model(results, vectorizer)
    print(f"\n🏆 Best model: {best}")

    # 6. Demo predictions
    print("\n" + "=" * 50)
    print("  🔍  DEMO PREDICTIONS")
    print("=" * 50)

    test_emails = [
        "Congratulations! You have won a FREE iPhone. Click here NOW to claim your prize!!!",
        "Hey, are we still meeting tomorrow for the project discussion?",
        "URGENT: Your account has been suspended. Verify immediately or lose access!",
        "Don't forget to submit your CSC 309 mini project by Friday.",
        "You have been selected for a $1,000,000 lottery. Reply with your bank details.",
        "Please review the attached report and let me know your thoughts.",
    ]

    for email in test_emails:
        result = predict_email(email)
        print(f"\n📧 \"{email[:60]}{'…' if len(email) > 60 else ''}\"")
        print(f"   → {result}")

    print("\n✅ All done! Run predict_email('your text here') to test new messages.")