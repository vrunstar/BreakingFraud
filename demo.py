import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Alpha\OneDrive\Desktop\suv_data.csv")

df["Gender"] = df["Gender"].replace({"Male":1, "Female":0})

X = df.drop(columns=["EstimatedSalary"])
y = df["EstimatedSalary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# ============ Linear Regression ====================
linreg = LinearRegression()
linreg.fit(X_train, y_train)

linreg_pred = linreg.predict(X_test)

print("R² Score:", r2_score(y_test, linreg_pred))
print("MSE:", mean_squared_error(y_test, linreg_pred))

plt.figure(figsize=(6,6))
plt.scatter(y_test, linreg_pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.show()

# ============ Logistic Regression ====================
Xlog = df.drop(columns=["Purchased"])
ylog = df["Purchased"]

Xlog_train, Xlog_test, ylog_train, ylog_test = train_test_split(
    Xlog, ylog, test_size=0.2, random_state=42
)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(Xlog_train, ylog_train)

logreg_pred = logreg.predict(Xlog_test)

print("Accuracy:", accuracy_score(ylog_test, logreg_pred))
print("Confusion Matrix:")
print(confusion_matrix(ylog_test, logreg_pred))

# ============ Logistic Regression ====================
Xlog = df.drop(columns=["Purchased"])
ylog = df["Purchased"]

Xlog_train, Xlog_test, ylog_train, ylog_test = train_test_split(
    Xlog, ylog, test_size=0.2, random_state=42
)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(Xlog_train, ylog_train)

logreg_pred = logreg.predict(Xlog_test)

print("Accuracy:", accuracy_score(ylog_test, logreg_pred))
print("Confusion Matrix:")
print(confusion_matrix(ylog_test, logreg_pred))