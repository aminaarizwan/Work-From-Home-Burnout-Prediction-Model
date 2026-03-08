from flask import Flask, render_template, request
import joblib
import streamlit as st
import pandas as pd

app = Flask(__name__)
import streamlit as st

st.set_page_config(page_title="Burnout Predictor", layout="centered")

# Custom CSS to center everything inside the app
st.markdown("""
    <style>
    .centered {
        text-align: center;
        color: #6C63FF;  /* Title color */
    }
    .subtitle {
        text-align: center;
        font-size:16px;
        color: #ffffff;  /* Subtitle color */
    }
    </style>
""", unsafe_allow_html=True)

# Use the classes
st.markdown('<h2 class="centered">Work From Home Burnout Predictor</h2>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter your work details</p>', unsafe_allow_html=True)
# Load trained models
dt_model = joblib.load("models/decision_tree_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")
knn_model = joblib.load("models/knn_model.pkl")
mlr_model = joblib.load("models/mlr_model.pkl")
dtc_model = joblib.load("models/decision_tree_risk_model.pkl")
le_risk = {0: "Low", 1: "Medium", 2: "High"}  # Label mapping

work_hours = st.number_input("Work Hours")
screen_time_hours = st.number_input("Screen Time Hours")
meetings_count = st.number_input("Meetings Count")
breaks_taken = st.number_input("Breaks Taken")
after_hours_work = st.number_input("After Hours Work")
sleep_hours = st.number_input("Sleep Hours")
task_completion_rate = st.number_input("Task Completion Rate")

if st.button("Predict Burnout"):

    data = pd.DataFrame([[
        work_hours,
        screen_time_hours,
        meetings_count,
        breaks_taken,
        after_hours_work,
        sleep_hours,
        task_completion_rate
    ]],
    columns=[
        'work_hours',
        'screen_time_hours',
        'meetings_count',
        'breaks_taken',
        'after_hours_work',
        'sleep_hours',
        'task_completion_rate'
    ])

    prediction = dt_model.predict(data)

    st.success(f"Predicted Burnout Score: {prediction[0]}")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input
        features = ['work_hours', 'screen_time_hours', 'meetings_count', 'breaks_taken', 
                    'after_hours_work', 'sleep_hours', 'task_completion_rate']
        input_data = []
        for f in features:
            val = float(request.form.get(f))
            input_data.append(val)

        df_input = pd.DataFrame([input_data], columns=features)

        # Predict burnout_score using all regressors
        dt_score = dt_model.predict(df_input)[0]
        rf_score = rf_model.predict(df_input)[0]
        knn_score = knn_model.predict(df_input)[0]
        mlr_score = mlr_model.predict(df_input)[0]

        # Average of all regression models
        avg_score = (dt_score + rf_score + knn_score + mlr_score)/4

        # Predict burnout_risk using classifier
        risk_pred = dtc_model.predict(df_input)[0]
        risk_label = le_risk[risk_pred]

        return render_template("index.html", score=round(avg_score,2), risk=risk_label)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False)