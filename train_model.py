# Optional: train and save model locally before deployment
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

iris = load_iris()
X, y = iris.data, iris.target
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])
pipe.fit(X, y)
joblib.dump(pipe, "model.pkl")
print("Saved model.pkl")