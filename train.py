import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os

# Ensure the models folder exists
os.makedirs("models", exist_ok=True)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

# Load dataset
df = pd.read_csv("data/news.csv")

# Ensure necessary columns exist
if "title" not in df.columns or "text" not in df.columns or "label" not in df.columns:
    raise ValueError("Dataset is missing required columns: 'title', 'text', or 'label'.")

# Combine title and text
df["content"] = df["title"].fillna('') + " " + df["text"].fillna('')

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

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

# Convert texts to embeddings in chunks (for periodic saving)
checkpoint_path = "models/train_embeddings_checkpoint.npy"

if os.path.exists(checkpoint_path):
    print("Resuming from previous checkpoint...")
    train_embeddings = np.load(checkpoint_path, allow_pickle=True).tolist()
    start_index = len(train_embeddings)
else:
    train_embeddings = []
    start_index = 0

# Process remaining embeddings
for i, text in enumerate(train_texts[start_index:], start=start_index):
    train_embeddings.append(get_bert_embedding(text))
    if i % 500 == 0:  # Save progress every 500 samples
        np.save(checkpoint_path, np.array(train_embeddings, dtype=object))
        print(f"Checkpoint saved at {i} samples...")

# Ensure embeddings are saved
np.save(checkpoint_path, np.array(train_embeddings, dtype=object))
train_embeddings = np.array(train_embeddings)

# Convert test texts to embeddings
test_embeddings = np.array([get_bert_embedding(text) for text in test_texts])

# Train or resume SVM model
model_path = "models/bert_svm_fake_news.pkl"

if os.path.exists(model_path):
    print("Loading existing model...")
    svm_model = joblib.load(model_path)

    if not hasattr(svm_model, "support_"):  # Check if the model was actually trained
        print("Model exists but is not trained. Retraining...")
        svm_model.fit(train_embeddings, train_labels)
        joblib.dump(svm_model, model_path)
else:
    print("No existing model found. Training a new one...")
    svm_model = SVC(kernel="linear")
    svm_model.fit(train_embeddings, train_labels)
    joblib.dump(svm_model, model_path)

print("Model training completed!")

# Evaluate the model
predictions = svm_model.predict(test_embeddings)
print("Model Accuracy:", accuracy_score(test_labels, predictions))
