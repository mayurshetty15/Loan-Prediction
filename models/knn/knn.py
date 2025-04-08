import pandas as pd

file_path = "../cleaned_loan_approval_dataset.csv"
df = pd.read_csv(file_path)

df.drop(columns=['loan_id'], inplace=True)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

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

best_k, best_acc = 1, 0
for k in range(1, 21):  
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    if acc > best_acc:
        best_k, best_acc = k, acc

final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train, y_train)

final_acc = accuracy_score(y_test, final_knn.predict(X_test))
model_path = "model.pkl"
joblib.dump(final_knn, model_path)

scaler_path = "scaler.pkl"
encoders_path = "label_encoders.pkl"
joblib.dump(scaler, scaler_path)
joblib.dump(label_encoders, encoders_path)

