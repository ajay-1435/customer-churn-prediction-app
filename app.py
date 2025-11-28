import streamlit as st
import pandas as pd
import joblib
from src.preprocess import preprocess, load_data

st.title("📊 Customer Churn Prediction App")

uploaded_file = st.file_uploader("Upload customer CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data", df.head())

    model = joblib.load("models/random_forest_model.pkl")
    X, _ = preprocess(df)

    # Add both predictions and probabilities
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    df['Predicted_Churn'] = predictions
    df['Churn_Probability'] = probabilities.round(2)
    
    st.write("### Predictions", df[['customerID', 'Predicted_Churn']].head())
    st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")


    import matplotlib.pyplot as plt

    st.write("### Churn Probability Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['Churn_Probability'], bins=10, color='skyblue', edgecolor='black')
    ax.set_xlabel("Probability of Churn")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)