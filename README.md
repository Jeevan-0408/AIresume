1️⃣ Install requirements

pip install scikit-learn pandas nltk joblib

2️⃣ Train a simple screening model

(This uses text classification – realistic and easy to extend)

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

3️⃣ Run the screening system

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

4️⃣ Example Output

{
  "score": 0.86,
  "decision": "PASS"
}

5️⃣ If you want a web-based AI screening API

(Backend for React / Lovable / frontend)

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

6️⃣ You can easily extend this to:

    Resume screening

    Student application screening

    Form moderation

    Skill matching

    AI pre-interview filtering

Next step (optional)

Tell me:

    🔹 Is this for resumes, students, content moderation, or something else?

    🔹 Do you want OpenAI / LLM-based screening instead of classic ML?

    🔹 Frontend (React / Lovable) or backend only?

I can customize the code exactly for your project 🚀
You said:
explain file contents
ChatGPT said:

Sure 🙂 I’ll explain what each file / section does in the AI-based screening system I shared, line by line and in simple terms.
1️⃣ Training File (Model Creation)

This part creates and trains the AI model.
Imports

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

What this does:

    pandas → handles data in table form

    TfidfVectorizer → converts text into numbers (AI can’t read text directly)

    LogisticRegression → the AI classifier (decides pass/reject)

    Pipeline → connects text processing + model into one flow

    joblib → saves the trained model to a file

Training Data

data = {
    "text": [
        "Python developer with machine learning experience",
        "Experienced data scientist with AI background",
        "No technical skills listed",
        "Beginner looking for first job",
        "Strong skills in Java, SQL, and APIs"
    ],
    "label": [1, 1, 0, 0, 1]
}

Meaning:

    text → example applications or resumes

    label

        1 = ACCEPT / PASS

        0 = REJECT

👉 This is teaching the AI what good and bad applications look like
Convert to DataFrame

df = pd.DataFrame(data)

Turns the data into a table format that ML libraries can use.
Model Pipeline

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

This is the AI brain:

    TfidfVectorizer

        Breaks text into words

        Assigns importance scores to each word

    LogisticRegression

        Learns patterns

        Predicts probability of PASS or REJECT

Train the Model

model.fit(df["text"], df["label"])

The AI learns from examples here.
Save the Model

joblib.dump(model, "screening_model.pkl")

    Saves the trained AI to a file

    You don’t need to retrain every time

    File created: screening_model.pkl

2️⃣ Screening File (Using the Model)

This part uses the trained model to screen new input.
Load the Model

model = joblib.load("screening_model.pkl")

Loads the saved AI model from disk.
Screening Function

def screen_application(text):

This function:

    Takes application text

    Returns score + decision

Predict Score

score = model.predict_proba([text])[0][1]

    AI gives a probability between 0 and 1

    Example:

        0.85 → very strong application

        0.20 → weak application

Decision Logic

if score > 0.7:
    decision = "PASS"
elif score > 0.4:
    decision = "REVIEW"
else:
    decision = "REJECT"

Rules:

    > 70% → PASS

    40–70% → REVIEW

    < 40% → REJECT

(You can change these thresholds)
Return Result

return {
    "score": round(score, 2),
    "decision": decision
}

Output example:

{
  "score": 0.86,
  "decision": "PASS"
}

3️⃣ API File (Flask Web Server)

This makes the AI usable by frontend apps (React, Lovable, etc.)
Flask Setup

from flask import Flask, request, jsonify
app = Flask(__name__)

Creates a small web server.
API Endpoint

@app.route("/screen", methods=["POST"])
def screen():

    URL: /screen

    Method: POST

    Accepts JSON data

Read Input

data = request.json
text = data.get("text", "")

Gets text sent from frontend.
Return JSON Response

return jsonify({
    "score": round(score, 2),
    "decision": decision
})

Frontend receives structured results.
4️⃣ File Summary
File	Purpose
train_model.py	Trains the AI
screening_model.pkl	Saved AI model
screen.py	Screens new data
app.py	API for frontend
5️⃣ What You Can Do Next

