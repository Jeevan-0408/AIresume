pip install scikit-learn pandas nltk joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Example training data
data = {
    "text": [
        "Python developer with machine learning experience",
        "Experienced data scientist with AI background",
        "No technical skills listed",
        "Beginner looking for first job",
        "Strong skills in Java, SQL, and APIs"
    ],
    "label": [1, 1, 0, 0, 1]  # 1 = pass, 0 = reject
}

df = pd.DataFrame(data)

# Build pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

# Train
model.fit(df["text"], df["label"])

# Save model
joblib.dump(model, "screening_model.pkl")

print("Model trained and saved.")
import joblib

# Load model
model = joblib.load("screening_model.pkl")

def screen_application(text):
    score = model.predict_proba([text])[0][1]

    if score > 0.7:
        decision = "PASS"
    elif score > 0.4:
        decision = "REVIEW"
    else:
        decision = "REJECT"

    return {
        "score": round(score, 2),
        "decision": decision
    }

# Example usage
application_text = "I have experience with Python, AI models, and data analysis"
result = screen_application(application_text)

print(result)
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("screening_model.pkl")

@app.route("/screen", methods=["POST"])
def screen():
    data = request.json
    text = data.get("text", "")

    score = model.predict_proba([text])[0][1]
    decision = "PASS" if score > 0.7 else "REVIEW" if score > 0.4 else "REJECT"

    return jsonify({
        "score": round(score, 2),
        "decision": decision
    })

if __name__ == "__main__":
    app.run(debug=True)
