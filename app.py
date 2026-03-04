from flask import Flask, request, render_template_string
import joblib
import os
import numpy as np
from sklearn.datasets import load_iris

app = Flask(__name__)

# Load model safely
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("model.pkl not found. Make sure it is in the root directory.")

model = joblib.load(MODEL_PATH)

# Load iris labels once
iris = load_iris()
target_names = iris.target_names

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Iris Predictor</title>
</head>
<body>
    <h2>Iris Predictor 🌸</h2>
    <form method="post">
        Sepal Length: <input name="f1" required><br><br>
        Sepal Width: <input name="f2" required><br><br>
        Petal Length: <input name="f3" required><br><br>
        Petal Width: <input name="f4" required><br><br>
        <input type="submit" value="Predict">
    </form>

    {% if result %}
        <h3>Prediction: {{ result }}</h3>
    {% endif %}
</body>
</html>
"""

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

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template_string(HTML, result=result)


# Important for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)