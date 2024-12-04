import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = joblib.load('processed_data.pkl')
label_encoder = joblib.load('label_encoder.pkl')

log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)


pred = log_reg.predict(X_test)

print("Logistic Regression accuracy: ", accuracy_score(y_test, pred))
print("Classification report:\n ", classification_report(y_test,pred))
