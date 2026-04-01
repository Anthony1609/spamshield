from flask import Flask, render_template, request, jsonify
import pickle, re, nltk, os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)

app = Flask(__name__)

# ── Load model & vectorizer ──────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
model      = pickle.load(open(os.path.join(BASE, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE, "vectorizer.pkl"), "rb"))

stemmer    = PorterStemmer()
stop_words = set(stopwords.words("english"))

# ── Preprocessing (must match training) ─────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # remove URLs
    text = re.sub(r"\W", " ", text)                # remove special chars
    text = re.sub(r"\s+", " ", text).strip()       # collapse whitespace
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# ── Stats (from training run) ────────────────────────────────
STATS = {
    "total_emails"   : 5572,
    "ham_count"      : 4825,
    "spam_count"     : 747,
    "nb_accuracy"    : 97.67,
    "lr_accuracy"    : 97.31,
    "nb_cv"          : 97.68,
    "lr_cv"          : 96.72,
    "nb_precision"   : 100.0,
    "nb_recall"      : 83.0,
    "best_model"     : "Naive Bayes",
}

# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", stats=STATS)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    cleaned  = preprocess(text)
    features = vectorizer.transform([cleaned]).toarray()

    # ── Spam probability with lowered threshold ──────────────
    proba          = model.predict_proba(features)[0]
    spam_prob      = proba[1]

    # Flag as spam if >30% probability (catches long marketing emails)
    pred           = 1 if spam_prob >= 0.30 else 0
    confidence     = round(float(spam_prob if pred == 1 else proba[0]) * 100, 2)

    # ── Top spam keywords detected ───────────────────────────
    feature_names = vectorizer.get_feature_names_out()
    scores        = features[0]
    top_idx       = scores.argsort()[::-1][:5]
    keywords      = [feature_names[i] for i in top_idx if scores[i] > 0]

    return jsonify({
        "label"     : "SPAM" if pred == 1 else "HAM",
        "confidence": confidence,
        "keywords"  : keywords,
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
