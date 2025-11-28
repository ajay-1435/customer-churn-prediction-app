import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from preprocess import load_data, preprocess

def train_model():
    df = load_data('C:/Users/kanam/OneDrive/Desktop/customer_churn_prediction/data/telco_churn.csv')
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    print("✅ Model trained")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


    import joblib
    joblib.dump(model, 'models/random_forest_model.pkl')
    print("✅ Model saved to models/random_forest_model.pkl")

if __name__ == "__main__":
    train_model()


