import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# =========================
# 1. LOAD DATA
# =========================

DATA_PATH = r"C:\Users\Shota\Desktop\s_chalauri25_18943.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print(df.head())

# =========================
# 2. DATA PREPARATION
# =========================

X = df.drop("is_spam", axis=1)
y = df["is_spam"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("\nData split completed")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# =========================
# 3. TRAIN LOGISTIC REGRESSION MODEL
# =========================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel training completed")

# =========================
# 4. MODEL COEFFICIENTS
# =========================

coefficients = model.coef_[0]

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", ascending=False)

print("\nLogistic Regression Coefficients:")
print(feature_importance)

# =========================
# 5. MODEL VALIDATION
# =========================

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nAccuracy:")
print(accuracy)

# =========================
# 6. EMAIL FEATURE EXTRACTION
# =========================

def extract_features_from_email(email_text):
    email_text = email_text.lower()
    features = {}

    for feature in X.columns:
        features[feature] = email_text.count(feature)

    return pd.DataFrame([features])

# =========================
# 7. EMAIL CLASSIFICATION
# =========================

def classify_email(email_text):
    features_df = extract_features_from_email(email_text)
    prediction = model.predict(features_df)[0]
    return "SPAM" if prediction == 1 else "LEGITIMATE"

# =========================
# 8. TEST MANUAL EMAILS
# =========================

spam_email = """
Congratulations! You have won a FREE prize.
Click this urgent link now to claim your money.
Limited time offer!!!
"""

legit_email = """
Hello,

Please find attached the project report discussed in our last meeting.
Let me know if you have any questions.

Best regards,
"""

print("\nSpam email classification result:")
print(classify_email(spam_email))

print("\nLegitimate email classification result:")
print(classify_email(legit_email))

# =========================
# 9. VISUALIZATION 1: CLASS DISTRIBUTION
# =========================

plt.figure()
df["is_spam"].value_counts().plot(kind="bar")
plt.title("Spam vs Legitimate Email Distribution")
plt.xlabel("Email Class (0 = Legitimate, 1 = Spam)")
plt.ylabel("Count")
plt.show()

# =========================
# 10. VISUALIZATION 2: CONFUSION MATRIX HEATMAP
# =========================

plt.figure()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Spam"],
    yticklabels=["Legitimate", "Spam"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()
