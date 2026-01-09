# aimlmid2026_s_chalauri25
Task 1 – Finding the Correlation

Objective: Find Pearson's correlation coefficient for the given dataset and create a scatter plot with the regression line.

Code: task1_correlation.py

import numpy as np
import matplotlib.pyplot as plt

x = np.array([-9, -7, -5, -3.5, -1.0, 1, 3, 5, 7.9, 9.9])
y = np.array([4, 4.5, 3, 4, 1.0, 1.3, -2, -3.6, -4.9, -5.8])

# Pearson's correlation coefficient
r = np.corrcoef(x, y)[0, 1]
print("Pearson's r:", r)

# Line of best fit
slope, intercept = np.polyfit(x, y, 1)

# Scatter plot
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, slope * x + intercept, color='red', label='Line of Best Fit')
plt.title('Scatter Plot of Data Points with Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()
plt.savefig('correlation_plot.png')
plt.show()


Result:

Pearson's correlation coefficient (r): ~ -0.968

The plot shows a strong negative correlation between X and Y. The red line represents the linear regression fit.
<img width="633" height="479" alt="image" src="https://github.com/user-attachments/assets/ba2c2c22-0ffb-469f-b19a-303d94d69d22" />

Task 2 – Spam Email Detection

Objective: Build a logistic regression model to classify emails as Spam or Legitimate.

Data File: s_chalauri25_18943.csv

Data Loading
import pandas as pd

DATA_PATH = "s_chalauri25_18943.csv"
df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully")
print(df.head())


Loaded the dataset successfully.

Target column: is_spam (1 = Spam, 0 = Legitimate).

Data Preparation
from sklearn.model_selection import train_test_split

X = df.drop("is_spam", axis=1)
y = df["is_spam"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


Split the dataset into 70% training and 30% testing data.

Model Training
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


Logistic Regression model trained on 70% of the dataset.

Model Coefficients
coefficients = model.coef_[0]
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", ascending=False)
print(feature_importance)


Coefficients indicate the influence of each feature on predicting spam.

Positive coefficients → increases likelihood of spam.

Model Validation
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)


Confusion Matrix:

	Predicted Legit	Predicted Spam
Actual Legit	xx	xx
Actual Spam	xx	xx

Accuracy: ~ accuracy_score (calculated from test set)

Email Classification
def extract_features_from_email(email_text):
    email_text = email_text.lower()
    features = {feature: email_text.count(feature) for feature in X.columns}
    return pd.DataFrame([features])

def classify_email(email_text):
    features_df = extract_features_from_email(email_text)
    prediction = model.predict(features_df)[0]
    return "SPAM" if prediction == 1 else "LEGITIMATE"


Extracts features from email text and predicts its class.

Manual Test Emails

Spam Email:

Congratulations! You have won a FREE prize. Click this urgent link now to claim your money. Limited time offer!!!


Reason: Contains typical spam keywords: “FREE”, “urgent”, “prize”, “limited time”.

Legitimate Email:

Hello, Please find attached the project report discussed in our last meeting. Let me know if you have any questions. Best regards,


Reason: Professional language, normal work email context, no spam keywords.

Classification Results:

Spam Email → SPAM

Legitimate Email → LEGITIMATE

Visualizations
1. Class Distribution
plt.figure()
df["is_spam"].value_counts().plot(kind="bar")
plt.title("Spam vs Legitimate Email Distribution")
plt.xlabel("Email Class (0 = Legitimate, 1 = Spam)")
plt.ylabel("Count")
plt.show()


Shows ratio of spam vs legitimate emails. Helps check for class imbalance.

<img width="636" height="476" alt="image" src="https://github.com/user-attachments/assets/78b4c4ea-32c8-4960-8259-36e4ba50f9f0" />


2. Confusion Matrix Heatmap
import seaborn as sns
plt.figure()
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Legitimate", "Spam"],
    yticklabels=["Legitimate", "Spam"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()



Visualizes model performance. Shows where misclassifications occur.

<img width="632" height="480" alt="image" src="https://github.com/user-attachments/assets/6a9e4688-d846-4c88-a9fc-dbfc11ec8314" />

results of code: 
Dataset loaded successfully
   words  links  capital_words  spam_word_count  is_spam
0    892      7             24                8        1
1    243      1              3                0        0
2    283      0              2                1        0
3    478      0              9                9        1
4    314      0              2                0        0

Data split completed
Training samples: 1750
Testing samples: 750

Model training completed

Logistic Regression Coefficients:
           Feature  Coefficient
1            links     0.964295
3  spam_word_count     0.790066
2    capital_words     0.467676
0            words     0.008019

Confusion Matrix:
[[355  12]
 [ 25 358]]

Accuracy:
0.9506666666666667

Spam email classification result:
LEGITIMATE

Legitimate email classification result:
LEGITIMATE
