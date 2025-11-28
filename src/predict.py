import pandas as pd
import joblib
from preprocess import preprocess, load_data

def predict_new_data(csv_path):
    model = joblib.load('models/random_forest_model.pkl')
    df = load_data(csv_path)
    X, _ = preprocess(df)  # We ignore y here

    predictions = model.predict(X)
    df['Predicted_Churn'] = predictions
    print(df[['customerID', 'Predicted_Churn']].head())

if __name__ == "__main__":
    predict_new_data('C:/Users/kanam/OneDrive/Desktop/customer_churn_prediction/data/telco_churn.csv')