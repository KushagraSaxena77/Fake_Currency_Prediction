import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv('fake_bills.csv')
print(df.head())

if df.isnull().any().any():
    df = df.dropna()


features = df[['diagonal','height_left','height_right','margin_up','length']]
target = df['is_genuine']

label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.3, random_state=42)

joblib.dump((X_train, X_test, y_train, y_test), 'processed_data.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Preprocessing successfully completed.")


