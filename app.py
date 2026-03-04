from flask import Flask, render_template, request, redirect, url_for
import os
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "model.pkl"

app = Flask(__name__)


def train_and_save_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe.fit(X, y)
    joblib.dump(pipe, MODEL_PATH)
    return pipe


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return train_and_save_model()


model = load_model()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        f1 = float(request.form.get("sepal_length"))
        f2 = float(request.form.get("sepal_width"))
        f3 = float(request.form.get("petal_length"))
        f4 = float(request.form.get("petal_width"))
    except Exception:
        return redirect(url_for("index"))

    X = np.array([[f1, f2, f3, f4]])
    pred = model.predict(X)[0]

    # map target to name
    iris = load_iris()
    species = iris.target_names[pred]

    return render_template("result.html", species=species, features=X.tolist()[0])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)