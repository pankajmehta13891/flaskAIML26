from flask import Flask, request, render_template
import joblib
import os
import numpy as np
from sklearn.datasets import load_iris

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(MODEL_PATH)

iris = load_iris()
target_names = iris.target_names


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        f1 = float(request.form["f1"])
        f2 = float(request.form["f2"])
        f3 = float(request.form["f3"])
        f4 = float(request.form["f4"])

        features = [f1, f2, f3, f4]

        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]

        result = target_names[prediction]
        confidence = round(np.max(probabilities) * 100, 2)

        return render_template(
            "result.html",
            result=result,
            confidence=confidence,
            f1=f1,
            f2=f2,
            f3=f3,
            f4=f4
        )

    except Exception:
        return render_template("result.html", result="Invalid Input")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)