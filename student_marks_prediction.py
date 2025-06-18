# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load Dataset
data = pd.read_csv('data.csv')
print(data.head())

# Visualize Data
plt.scatter(data['Hours'], data['Marks'])
plt.title('Study Hours vs Marks')
plt.xlabel('Hours Studied')
plt.ylabel('Marks')
plt.show()

# Split Data
X = data[['Hours']]  # Features
y = data['Marks']    # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Compare Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)

# Model Evaluation
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

# Predict for custom input
hours = float(input("Enter number of study hours: "))
predicted_marks = model.predict([[hours]])
print(f"Predicted Marks = {predicted_marks[0]:.2f}")
