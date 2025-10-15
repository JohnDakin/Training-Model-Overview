import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Create a sample dataset
data = {
    'hours_studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'hours_slept': [5, 6, 6, 7, 8, 7, 8, 9, 9, 10],
    'passed': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['hours_studied', 'hours_slept']]  # features
y = df['passed']                           # target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Predictions:", y_pred)
print("Accuracy:", accuracy)


# Predict if a student who studies 6 hours and sleeps 8 hours will pass
print(model.predict([[6, 8]]))


import joblib
joblib.dump(model, 'student_model.pkl')
print("Model saved successfully!")


model = joblib.load('student_model.pkl')
print(model.predict([[3, 6]]))  # 0 or 1

