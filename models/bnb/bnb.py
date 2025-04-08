import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Binarizer
from sklearn.naive_bayes import BernoulliNB
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

binarizer = Binarizer()
X_binarized = binarizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_binarized, y, test_size=0.2)

bnb = BernoulliNB()
bnb.fit(X_train, y_train)

y_pred = bnb.predict(X_test)
final_acc = accuracy_score(y_test, y_pred)

model_path = "model.pkl"
joblib.dump(bnb, model_path)

binarizer_path = "scaler.pkl"
encoders_path = "label_encoders.pkl"
joblib.dump(binarizer, binarizer_path)
joblib.dump(label_encoders, encoders_path)

print(f"Final Accuracy: {final_acc}")
