# 💳 Credit Card Approval Prediction

## 📌 Overview

**Credit Card Approval Prediction** is a machine learning web application that predicts whether an applicant is likely to be **Approved** or **Rejected** based on personal, financial, and credit-related information.

The application is built using **Python, Flask, Pandas, NumPy, and Scikit-learn** with a **Random Forest Classifier**.

> **Note:** The dataset contains historical credit-status information rather than actual bank approval/rejection labels. Therefore, the approval/rejection result is derived from credit-risk classification and is intended for educational purposes.

---

## 🚀 Features

* Credit card approval prediction
* Random Forest machine learning model
* Risk probability display
* Applicant information form
* Flask web application
* Responsive user interface
* Real-time prediction

---

## 🛠️ Technologies

* **Language:** Python
* **Backend:** Flask
* **Machine Learning:** Scikit-learn, Random Forest
* **Data Processing:** Pandas, NumPy
* **Frontend:** HTML, CSS, JavaScript
* **Tools:** VS Code, Git, GitHub

---

## 📂 Project Structure

```text
credit-card-approval-prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── database/
│   └── credit_card_dataset.csv
│
├── model/
│   ├── credit_card_model.pkl
│   ├── feature_columns.pkl
│   ├── label_encoders.pkl
│   └── scaler.pkl
│
├── static/
│   ├── style.css
│   └── script.js
│
└── templates/
    ├── home.html
    ├── index.html
    └── result.html
```

---

## 🔄 How It Works

1. User enters applicant information.
2. Flask receives and processes the data.
3. Categorical data is encoded.
4. The trained Random Forest model makes a prediction.
5. The application displays the credit-risk result.

```text
Lower Credit Risk  →  APPROVED
Higher Credit Risk →  REJECTED
```

---

## 📊 Model Performance

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 79.68% |
| Precision | 29.42% |
| Recall    | 51.98% |
| F1 Score  | 37.57% |

---

## 💻 Run Locally

### Clone the repository

```bash
git clone https://github.com/Kanaka-durga57/credit-card-approval-prediction.git
cd credit-card-approval-prediction
```

### Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 🔮 Future Enhancements

* Prediction history
* User authentication
* Database integration
* Model explainability
* Hyperparameter tuning
* Cloud deployment
* Training with actual approval/rejection labels

---

## ⚠️ Disclaimer

This project is developed for **educational purposes**. The prediction is not an actual bank approval or rejection decision.

---

## 👩‍💻 Author

**Kanaka Durga**
Computer Science and Engineering

GitHub: https://github.com/Kanaka-durga57
