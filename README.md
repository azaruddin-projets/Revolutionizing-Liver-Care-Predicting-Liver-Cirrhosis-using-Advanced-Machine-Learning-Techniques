# Revolutionizing Liver Care 
## Predicting Liver Cirrhosis using Advanced Machine Learning Techniques

A web-based tool to predict the presence of liver cirrhosis using machine learning and a simple Flask web interface. This project aims to assist in early detection of liver disease.

---

## Project Objective

Liver cirrhosis is a serious, irreversible liver condition. Early detection allows for timely medical intervention. This app helps medical professionals (or users) to quickly assess risk using patient data.

---

## Dataset

- **Indian Liver Patient Dataset (ILPD)** from Kaggle
- Features include: Age, Gender, Bilirubin levels, Proteins, Enzymes, etc.
- Target: `is_patient` (0 = Healthy, 1 = Liver Disease)

---

## Machine Learning Model

- **Random Forest Classifier**
- Features normalized using `StandardScaler`
- Trained using 80-20 split
- Accuracy: ~68% (baseline version)

---

## Tech Stack

- Python (Pandas, NumPy, Scikit-learn, Seaborn, Joblib)
- Flask (Backend)
- HTML + CSS (Responsive, animated UI)

---

## How to Run Locally

```bash
pip install flask numpy pandas scikit-learn joblib
python app.py
