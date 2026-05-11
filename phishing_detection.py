
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset load
df = pd.read_csv("dataset_phishing.csv")
print("Total rows:", len(df))
print(df["status"].value_counts())

# Feature selection
selected_features = [
    "length_url", "length_hostname", "nb_dots", "nb_hyphens",
    "nb_at", "nb_qm", "nb_slash", "nb_www", "ratio_digits_url",
    "https_token", "ratio_digits_host", "nb_subdomains",
    "prefix_suffix", "shortening_service", "length_words_raw",
    "char_repeat", "shortest_word_host", "longest_word_path",
    "domain_age", "domain_registration_length",
    "web_traffic", "google_index", "page_rank"
]

X = df[selected_features].fillna(df[selected_features].median())
y = df["status"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Results
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"ACCURACY: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate","Phishing"],
            yticklabels=["Legitimate","Phishing"])
plt.title("Confusion Matrix - Phishing URL Detection")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# Feature Importance
feat_imp = pd.Series(model.feature_importances_,
           index=selected_features).sort_values(ascending=False)
plt.figure(figsize=(11,5))
feat_imp.plot(kind="bar", color="steelblue", edgecolor="white")
plt.title("Top Features for Phishing Detection")
plt.xlabel("Feature"); plt.ylabel("Importance Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
