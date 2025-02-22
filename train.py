import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

# Load and Merge Datasets
df_fake = pd.read_csv("data/Fake.csv")
df_real = pd.read_csv("data/True.csv")

# Assign labels (1 = Fake, 0 = Real)
df_fake["label"] = 1
df_real["label"] = 0

# Combine both datasets
df = pd.concat([df_fake, df_real], ignore_index=True)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Combine title and text to form the final content
df["content"] = df["title"].fillna('') + " " + df["text"].fillna('')

# Split dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["content"].values, df["label"].values, test_size=0.2, random_state=42
)

# Function to get BERT embeddings
def get_bert_embedding(text):
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = bert_model(**tokens)
    return output.last_hidden_state[:, 0, :].numpy().flatten()

# Convert texts to embeddings
train_embeddings = np.array([get_bert_embedding(text) for text in train_texts])
test_embeddings = np.array([get_bert_embedding(text) for text in test_texts])

# Train SVM
svm_model = SVC(kernel="linear")
svm_model.fit(train_embeddings, train_labels)

# Create model directory if not exists
os.makedirs("models", exist_ok=True)

# Save the trained model
joblib.dump(svm_model, "models/bert_svm_fake_news.pkl")
print("Model saved successfully!")

# Evaluate the model
predictions = svm_model.predict(test_embeddings)
print("Model Accuracy:", accuracy_score(test_labels, predictions))
