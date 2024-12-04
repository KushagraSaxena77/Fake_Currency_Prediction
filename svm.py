import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = joblib.load('processed_data.pkl')
label_encoder = joblib.load('label_encoder.pkl')

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf'],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(SVC(), param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

svm = grid_search.best_estimator_

pred = svm.predict(X_test)

print("SVM accuracy: ", accuracy_score(y_test, pred))
print("Classification report:\n", classification_report(y_test, pred, zero_division=0))
