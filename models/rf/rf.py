import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import joblib


file_path = "../cleaned_loan_approval_dataset.csv"
df = pd.read_csv(file_path)

df.drop(columns=['loan_id'], inplace=True)


label_encoders = {}
for col in ['education', 'self_employed', 'loan_status']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le


X = df.drop(columns=['loan_status'])
y = df['loan_status']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)


param_grid = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}


rf = RandomForestClassifier()
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=20, cv=5, verbose=2, n_jobs=-1)
random_search.fit(X_train, y_train)


best_rf = random_search.best_estimator_
final_acc = accuracy_score(y_test, best_rf.predict(X_test))


model_path = "model.pkl"
joblib.dump(best_rf, model_path)


scaler_path = "scaler.pkl"
encoders_path = "label_encoders.pkl"
joblib.dump(scaler, scaler_path)
joblib.dump(label_encoders, encoders_path)

print(f"Best Parameters: {random_search.best_params_}, Final Accuracy: {final_acc}")
