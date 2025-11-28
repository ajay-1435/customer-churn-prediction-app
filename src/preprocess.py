import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)

    # Clean blank strings and convert to numeric
    for col in ['TotalCharges', 'MonthlyCharges', 'tenure']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Map churn to binary
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    return df
def preprocess(df):
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    bin_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaymentMethod']

    # Clean blank strings and convert to numeric
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values
    df.dropna(inplace=True)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=[col for col in cat_cols if col in df.columns], drop_first=True)

    # Map binary columns
    for col in bin_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

    # Scale numeric columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Separate features and target
    X = df.drop(columns=[col for col in ['customerID', 'Churn'] if col in df.columns])
    y = df['Churn'] if 'Churn' in df.columns else None

    return X, y