import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide"
)

# Load model and encoders
model = joblib.load("best_model/xgb_model.pkl")
encoders = {
    col: joblib.load(f"encoders/{col}_encoder.pkl")
    for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
}

# Header
st.title("Credit Risk Prediction App")
st.markdown("### Enter applicant information to assess credit risk")
st.divider()

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    age = st.number_input("Age", min_value=18, max_value=80, value=30, help="Applicant's age")
    sex = st.selectbox("Sex", ["male", "female"])
    job = st.selectbox(
        "Job Level",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"Level {x}",
        help="Job skill level: 0 (unskilled) to 3 (highly skilled)"
    )
    housing = st.selectbox("Housing", ["own", "rent", "free"])

with col2:
    st.subheader("Financial Information")
    saving_accounts = st.selectbox(
        "Saving Accounts",
        ["little", "moderate", "quite rich", "rich"]
    )
    checking_account = st.selectbox(
        "Checking Account",
        ["little", "moderate", "rich"]
    )
    credit_amount = st.number_input(
        "Credit Amount ($)",
        min_value=0,
        value=1000,
        step=100,
        help="Requested credit amount"
    )
    duration = st.number_input(
        "Duration (months)",
        min_value=1,
        max_value=72,
        value=12,
        help="Loan duration in months"
    )

st.divider()

# Prediction button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_button = st.button("Predict Credit Risk", use_container_width=True, type="primary")

if predict_button:
    # Prepare input data
    input_df = pd.DataFrame({
        "Age": [age],
        "Sex": [encoders["Sex"].transform([sex])[0]],
        "Job": [job],
        "Housing": [encoders["Housing"].transform([housing])[0]],
        "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
        "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
        "Credit amount": [credit_amount],
        "Duration": [duration],
    })

    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    st.divider()

    # Display result with probability
    if prediction == 1:
        st.success("### Prediction: GOOD Credit Risk")
        st.metric(
            label="Confidence",
            value=f"{prediction_proba[1] * 100:.1f}%",
            help="Model's confidence in this prediction"
        )
        st.info("💡 This applicant shows good indicators for credit approval.")
    else:
        st.error("### Prediction: BAD Credit Risk")
        st.metric(
            label="Confidence",
            value=f"{prediction_proba[0] * 100:.1f}%",
            help="Model's confidence in this prediction"
        )
        st.warning("⚠️ This applicant may pose a higher credit risk.")

    # Show input summary
    with st.expander("View Input Summary"):
        summary_data = {
            "Feature": ["Age", "Sex", "Job Level", "Housing", "Saving Accounts",
                        "Checking Account", "Credit Amount", "Duration"],
            "Value": [str(age), str(sex), str(job), str(housing), str(saving_accounts),
                      str(checking_account), f"${credit_amount:,}", f"{duration} months"]
        }
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, hide_index=True, width='stretch')

# Footer
st.divider()
st.caption("This prediction is for informational purposes only. Final credit decisions should involve human review.")