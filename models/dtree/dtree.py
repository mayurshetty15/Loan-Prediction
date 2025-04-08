import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
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

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2 )

best_depth, best_acc = 1, 0
for depth in range(1, 21): 
    tree = DecisionTreeClassifier(max_depth=depth )
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    if acc > best_acc:
        best_depth, best_acc = depth, acc

final_tree = DecisionTreeClassifier(max_depth=best_depth, )
final_tree.fit(X_train, y_train)

final_acc = accuracy_score(y_test, final_tree.predict(X_test))
model_path = "model.pkl"
joblib.dump(final_tree, model_path)

scaler_path = "scaler.pkl"
encoders_path = "label_encoders.pkl"
joblib.dump(scaler, scaler_path)
joblib.dump(label_encoders, encoders_path)

print(f"Best Depth: {best_depth}, Final Accuracy: {final_acc}")
