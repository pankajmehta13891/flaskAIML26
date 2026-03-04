from flask import Flask, request, render_template
import joblib
import os
from sklearn.datasets import load_iris

app = Flask(__name__)

model = joblib.load("model.pkl")

iris = load_iris()
target_names = iris.target_names

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        try:
            features = [
                float(request.form["f1"]),
                float(request.form["f2"]),
                float(request.form["f3"]),
                float(request.form["f4"])
            ]

            prediction = model.predict([features])[0]
            result = target_names[prediction]

        except Exception:
            result = "Invalid input. Please enter numeric values."

    return render_template("index.html", result=result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)