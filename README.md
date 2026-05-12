# 🎓 Student Exam Score Predictor

> **Live App:** [Student-Performance-Predictor](https://ipycharmer-student-performance-predictor.streamlit.app/)
> **Project Page:** [ipycharmer.github.io/student-performance-predictor](https://ipycharmer.github.io/student-performance-predictor)  
> **Dataset:** [Student Performance Factors — Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)

Predict a student's exam score based on 19 real-world factors and calculate the **minimum weekly study hours** needed to achieve a passing score (≥ 60). Trained on 6,607 student records using a full `sklearn` Pipeline.

---

## 🚀 Live Demo

Click the sliders, fill in your details, get your predicted score instantly.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ipycharmer-student-performance-predictor.streamlit.app/)

---

## 🧠 Key Finding

> **Attendance (38.5%) matters more than hours studied (24.7%).**  
> Most students grind study hours while skipping class. The data says that's the wrong trade-off.

| Factor | Importance |
|---|---|
| Attendance | 38.5% |
| Hours Studied | 24.7% |
| Previous Scores | 8.8% |
| Tutoring Sessions | 3.6% |
| Access to Resources | 3.0% |

---

## 🏗️ Pipeline Architecture

Instead of manually looping encoders (which causes *unseen data* errors at prediction time), the entire preprocessing + model is a single `sklearn` Pipeline:

```
Raw DataFrame (strings + numbers)
       ↓
ColumnTransformer
  ├── OrdinalEncoder  → 8 ordinal cols  (Low/Med/High → 0/1/2)
  ├── OrdinalEncoder  → 5 binary cols   (Yes/No → 0/1)
  └── MinMaxScaler    → 6 numeric cols  (→ 0 to 1 range)
       ↓
RandomForestRegressor (200 trees)
       ↓
Predicted Exam Score
```

One `pipeline.joblib` file contains everything. The Streamlit app calls:
```python
pipeline = joblib.load("pipeline.joblib")
score = pipeline.predict(raw_df)[0]  # raw strings — no manual encoding needed
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| R² Score | 0.68 |
| MAE | 2.03 points |
| RMSE | ~3.1 points |
| Training set | 5,285 students |
| Test set | 1,322 students |

---

## 🗂️ Repository Structure

```
student-performance-predictor/
│
├── app.py                              ← Streamlit web app
├── pipeline.joblib                     ← trained pipeline (preprocessing + model)
├── requirements.txt
│
├── student_performance_analysis.ipynb  ← full EDA + pipeline training notebook
├── StudentPerformanceFactors.csv       ← raw dataset
│
└── project.html                        ← project page (matches portfolio aesthetics)
```

---

## ⚙️ Run Locally

```bash
git clone https://github.com/ipycharmer/student-performance-predictor
cd student-performance-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

No `category-encoders` needed — pure `sklearn` only.

---

## 🔄 Retrain the Model

Open `student_performance_analysis.ipynb` and run all cells.  
The last cell saves a fresh `pipeline.joblib` automatically.

---

## 📬 Contact

**Ameer Hamza Nasir**  
[Email](mailto:hamza6700@gmail.com) · [Portfolio](https://ipycharmer.github.io) · [Github](https://github.com/ipycharmer)

---

<p align="center">Built with Python, Scikit-learn, and Streamlit · Deployed on Streamlit Cloud</p>
