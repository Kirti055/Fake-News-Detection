import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import joblib
import os

# Ensure necessary folders exist
os.makedirs("models", exist_ok=True)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

# File paths
checkpoint_embeddings = "models/train_embeddings.npy"
checkpoint_labels = "models/train_labels.npy"
model_path = "models/bert_svm_fake_news.pkl"

# Load datasets
df_old = pd.read_csv("data/old_news.csv") if os.path.exists("data/old_news.csv") else None
df_new = pd.read_csv("data/new_news.csv") if os.path.exists("data/new_news.csv") else None

# Process old dataset
if df_old is not None:
    df_old["content"] = df_old["title"].fillna('') + " " + df_old["text"].fillna('')
    df_old = df_old[["content", "label"]]
else:
    df_old = pd.DataFrame(columns=["content", "label"])

# Process new dataset
if df_new is not None:
    df_new["content"] = df_new["Text"].fillna('')
    df_new = df_new[["content", "label"]]
else:
    df_new = pd.DataFrame(columns=["content", "label"])

# Merge datasets and shuffle
df = pd.concat([df_old, df_new], ignore_index=True).sample(frac=1, random_state=42)

# Function to get BERT embeddings
def get_bert_embedding(text):
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = bert_model(**tokens)
    return output.last_hidden_state[:, 0, :].numpy().flatten()

# Load previous embeddings and labels
if os.path.exists(checkpoint_embeddings) and os.path.exists(checkpoint_labels):
    try:
        train_embeddings = np.load(checkpoint_embeddings, allow_pickle=True)
        train_labels = np.load(checkpoint_labels, allow_pickle=True)
        if len(train_embeddings) != len(train_labels):
            raise ValueError("Mismatch between embeddings and labels!")
        print(f"🔄 Loaded existing embeddings ({len(train_embeddings)}) and labels ({len(train_labels)})")
    except Exception as e:
        print(f"⚠️ Error loading checkpoints: {e}")
        train_embeddings = np.empty((0, 768), dtype=np.float32)
        train_labels = np.array([], dtype=np.int32)
else:
    train_embeddings = np.empty((0, 768), dtype=np.float32)
    train_labels = np.array([], dtype=np.int32)

# Find last processed index to resume
start_index = len(train_labels)
print(f"🔄 Resuming from index {start_index}")

# Convert new texts to embeddings from the last checkpoint
new_embeddings, new_labels = [], []
for i, row in enumerate(df.itertuples(index=False), start=1):
    if i <= start_index:
        continue  # Skip already processed data
    
    embedding = get_bert_embedding(row.content)
    new_embeddings.append(embedding)
    new_labels.append(row.label)
    
    if i % 500 == 0 or i == len(df):
        train_embeddings = np.vstack([train_embeddings, new_embeddings])
        train_labels = np.hstack([train_labels, new_labels])
        np.save(checkpoint_embeddings, train_embeddings)
        np.save(checkpoint_labels, train_labels)
        print(f"💾 Saved checkpoint at {i} samples...")
        new_embeddings, new_labels = [], []  # Reset buffer

# Train or load SVM model
if os.path.exists(model_path):
    print("🔄 Loading existing model...")
    svm_model = joblib.load(model_path)
else:
    print("🚀 Training new SVM model...")
    svm_model = SVC(kernel="linear")

# Ensure embeddings and labels match before training
if len(train_embeddings) == len(train_labels):
    print("🟢 Training SVM model...")
    svm_model.fit(train_embeddings, train_labels)
    joblib.dump(svm_model, model_path)
    print("✅ Model training completed!")
else:
    print("❌ Error: Mismatch between embeddings and labels!")
