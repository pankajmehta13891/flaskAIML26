# 🌸 Iris Flower Predictor (Flask + Machine Learning)

A simple Machine Learning web application built using **Flask** and **Scikit-Learn** that predicts the species of an Iris flower based on user input features.

Deployed-ready for platforms like Render 🚀

---

## 📌 Features

- 🌸 Predict Iris species (Setosa, Versicolor, Virginica)
- 📊 Displays prediction confidence %
- ✅ Input validation with range checks
- 🎨 Clean UI with separate CSS
- 🏗 Professional Flask project structure
- ☁ Ready for deployment (Render compatible)

---

## 🧠 Machine Learning Model

- Dataset: Iris Dataset
- Algorithm: Scikit-learn Classifier
- Method used:
  - `predict()` → Predict class
  - `predict_proba()` → Get confidence probability

---

## 📂 Project Structure


project/
│
├── app.py
├── model.pkl
├── requirements.txt
├── Procfile
├── .python-version
│
├── templates/
│ ├── index.html
│ └── result.html
│
└── static/
└── style.css



---

## 🚀 How to Run Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

Create Virtual Environment


conda create -n iris python=3.11
conda activate iris


Install Requirements

run application 
python app.py


Deployment (Render)

Uses .python-version → Python 3.11.9

Procfile:
web: gunicorn app:app

| Feature      | Allowed Range |
| ------------ | ------------- |
| Sepal Length | 4 – 8         |
| Sepal Width  | 2 – 5         |
| Petal Length | 1 – 7         |
| Petal Width  | 0.1 – 3       |



🛠 Tech Stack

Python 3.11

Flask

Scikit-learn

NumPy

HTML

CSS

Gunicorn

pip install -r requirements.txt


