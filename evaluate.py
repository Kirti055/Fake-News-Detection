import torch
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import BertTokenizer, BertModel

# File paths
model_path = "models/bert_svm_fake_news.pkl"
test_data_path = "data/test_news.csv"
test_embeddings_path = "models/test_embeddings.npy"
test_labels_path = "models/test_labels.npy"
results_path = "models/evaluation_results.txt"

# Load trained SVM model
if not os.path.exists(model_path):
    print("❌ Model file not found! Train the model first using train.py")
    exit()

svm_model = joblib.load(model_path)
print("✅ Model loaded successfully!")

# Load test dataset
if not os.path.exists(test_data_path):
    print("❌ Test dataset not found! Please ensure you have a test dataset in 'data/'")
    exit()

df_test = pd.read_csv(test_data_path)

# Ensure necessary columns exist
if "Text" not in df_test.columns or "label" not in df_test.columns:
    print("❌ Required columns ('Text', 'label') missing in test dataset!")
    exit()

df_test["content"] = df_test["Text"].fillna('')
df_test = df_test[["content", "label"]]

# Convert labels to numeric format
df_test["label"] = df_test["label"].astype(str).str.lower().map({"fake": 1, "real": 0})
df_test = df_test.dropna(subset=["label"])
y_test = df_test["label"].values

# Load tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

# Function to extract BERT embeddings
def get_bert_embedding(text):
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = bert_model(**tokens)
    return output.last_hidden_state[:, 0, :].numpy().flatten()

# Load or generate test embeddings
if os.path.exists(test_embeddings_path) and os.path.exists(test_labels_path):
    print("🔄 Loading saved test embeddings...")
    X_test = np.load(test_embeddings_path)
    y_test = np.load(test_labels_path)
else:
    print("🆕 Generating embeddings for test data...")
    X_test = np.array([get_bert_embedding(text) for text in df_test["content"]])

    # Save embeddings for future runs
    np.save(test_embeddings_path, X_test)
    np.save(test_labels_path, y_test)
    print("✅ Test embeddings saved!")

# Predict using the trained SVM model
print("🔍 Making predictions...")
y_pred = svm_model.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print results
print(f"\n📊 Confusion Matrix:\n{cm}")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"🔹 Precision: {precision:.4f}")
print(f"🔸 Recall: {recall:.4f}")
print(f"⭐ F1-Score: {f1:.4f}")

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

# Save results to a text file
with open(results_path, "w") as f:
    f.write(f"Confusion Matrix:\n{cm}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")

print(f"\n📁 Evaluation results saved in {results_path}")
