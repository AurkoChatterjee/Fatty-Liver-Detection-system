from flask import Flask, render_template, request
import sys
import os

# Fix imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
sys.path.append(PROJECT_ROOT)

from ml.predict import predict_risk
from llm.explanation_engine import generate_explanation

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    # Get form data
    patient_data = [
        float(request.form["age"]),
        float(request.form["male"]),
        float(request.form["weight"]),
        float(request.form["height"]),
        float(request.form["bmi"]),
        float(request.form["futime"]),
        float(request.form["chol"]),
        float(request.form["dbp"]),
        float(request.form["fib4"]),
        float(request.form["hdl"]),
        float(request.form["sbp"])
    ]

    # Prediction
    risk = predict_risk(patient_data)

    # Explanation
    explanation = generate_explanation(patient_data, risk)

    return render_template(
        "result.html",
        risk=round(risk*100, 2),
        explanation=explanation
    )

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)