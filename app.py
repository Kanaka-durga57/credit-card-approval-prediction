from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ==============================
# Model Files
# ==============================

MODEL_PATH = "model/credit_card_model.pkl"
ENCODER_PATH = "model/label_encoders.pkl"
SCALER_PATH = "model/scaler.pkl"
FEATURE_PATH = "model/feature_columns.pkl"


# ==============================
# Load Model Files
# ==============================

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_PATH)


# ==============================
# Home Page
# ==============================

@app.route("/")
def home():
    return render_template("home.html")


# ==============================
# Prediction Page
# ==============================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    # Show prediction form
    if request.method == "GET":
        return render_template("index.html")

    try:

        # ==============================
        # Get Form Data
        # ==============================

        form_data = request.form

        input_data = {
            "CODE_GENDER": form_data["CODE_GENDER"],
            "FLAG_OWN_CAR": form_data["FLAG_OWN_CAR"],
            "FLAG_OWN_REALTY": form_data["FLAG_OWN_REALTY"],
            "CNT_CHILDREN": int(form_data["CNT_CHILDREN"]),
            "AMT_INCOME_TOTAL": float(form_data["AMT_INCOME_TOTAL"]),
            "NAME_INCOME_TYPE": form_data["NAME_INCOME_TYPE"],
            "NAME_EDUCATION_TYPE": form_data["NAME_EDUCATION_TYPE"],
            "NAME_FAMILY_STATUS": form_data["NAME_FAMILY_STATUS"],
            "NAME_HOUSING_TYPE": form_data["NAME_HOUSING_TYPE"],
            "DAYS_BIRTH": int(form_data["DAYS_BIRTH"]),
            "DAYS_EMPLOYED": int(form_data["DAYS_EMPLOYED"]),
            "FLAG_MOBIL": int(form_data["FLAG_MOBIL"]),
            "FLAG_WORK_PHONE": int(form_data["FLAG_WORK_PHONE"]),
            "FLAG_PHONE": int(form_data["FLAG_PHONE"]),
            "FLAG_EMAIL": int(form_data["FLAG_EMAIL"]),
            "OCCUPATION_TYPE": form_data["OCCUPATION_TYPE"],
            "CNT_FAM_MEMBERS": float(form_data["CNT_FAM_MEMBERS"])
        }


        # ==============================
        # Convert Input to DataFrame
        # ==============================

        input_df = pd.DataFrame([input_data])


        # ==============================
        # Encode Categorical Features
        # ==============================

        for column, encoder in label_encoders.items():

            value = input_df[column].astype(str)

            if value.iloc[0] in encoder.classes_:

                input_df[column] = encoder.transform(value)

            else:

                # Unknown category
                input_df[column] = 0


        # ==============================
        # Arrange Features
        # ==============================

        input_df = input_df[feature_columns]


        # ==============================
        # Apply Scaling If Required
        # ==============================

        if scaler is not None:

            input_data_scaled = scaler.transform(input_df)

        else:

            input_data_scaled = input_df


        # ==============================
        # Make Prediction
        # ==============================

        prediction = model.predict(input_data_scaled)[0]

        probability = model.predict_proba(input_data_scaled)[0]


        # Probability of risky class
        risk_probability = probability[1] * 100


        # ==============================
        # Convert Prediction
        # ==============================

        if prediction == 1:

            # Higher risk
            result = "REJECTED"
            risk_status = "Higher Credit Risk"

        else:

            # Lower risk
            result = "APPROVED"
            risk_status = "Lower Credit Risk"


        # ==============================
        # Show Result Page
        # ==============================

        return render_template(
            "result.html",
            result=result,
            risk_status=risk_status,
            probability=round(risk_probability, 2)
        )


    except Exception as e:

        return f"""
        <!DOCTYPE html>

        <html>

        <head>

            <title>Prediction Error</title>

            <style>

                body {{
                    font-family: Arial, sans-serif;
                    background: #f5f5f5;
                    text-align: center;
                    padding: 50px;
                }}

                .error-box {{
                    background: white;
                    padding: 30px;
                    max-width: 700px;
                    margin: auto;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}

                h2 {{
                    color: #d32f2f;
                }}

                a {{
                    display: inline-block;
                    margin-top: 20px;
                    padding: 10px 20px;
                    background: #2563eb;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                }}

            </style>

        </head>

        <body>

            <div class="error-box">

                <h2>Prediction Error</h2>

                <p>{str(e)}</p>

                <a href="/predict">Go Back</a>

            </div>

        </body>

        </html>
        """


# ==============================
# Run Flask Application
# ==============================

if __name__ == "__main__":
    app.run(debug=True)