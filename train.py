import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import joblib
import os

# Ensure necessary folders exist
os.makedirs("models", exist_ok=True)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

# File paths
checkpoint_embeddings = "models/train_embeddings.npy"
checkpoint_labels = "models/train_labels.npy"
model_path = "models/bert_svm_fake_news.pkl"

# Load datasets
df_old_path = "data/old_news.csv"
df_new_path = "data/new_news.csv"

df_old = pd.read_csv(df_old_path) if os.path.exists(df_old_path) else None
df_new = pd.read_csv(df_new_path) if os.path.exists(df_new_path) else None

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

# Standardize labels
df_old["label"] = df_old["label"].astype(str).str.lower().map({"fake": 1, "real": 0})
df_new["label"] = df_new["label"].astype(str).str.lower().map({"fake": 1, "real": 0})

# Remove invalid labels
df_old = df_old.dropna(subset=["label"])
df_new = df_new.dropna(subset=["label"])

# Merge datasets
df = pd.concat([df_old, df_new], ignore_index=True).sample(frac=1, random_state=42)

# Function to get BERT embeddings
def get_bert_embedding(text):
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = bert_model(**tokens)
    return output.last_hidden_state[:, 0, :].numpy().flatten()

# Load previous embeddings if available
if os.path.exists(checkpoint_embeddings) and os.path.exists(checkpoint_labels):
    train_embeddings = np.load(checkpoint_embeddings, allow_pickle=True)
    train_labels = np.load(checkpoint_labels, allow_pickle=True)
    print(f"🔄 Loaded existing embeddings ({len(train_embeddings)}) and labels ({len(train_labels)})")
else:
    train_embeddings = np.empty((0, 768))
    train_labels = np.array([])

# Convert new texts to embeddings (only if needed)
new_embeddings = []
new_labels = []
processed_count = len(train_labels)  # Start from the last saved count

for i, row in enumerate(df.itertuples(index=False)):
    if i < processed_count:  # Skip already processed data
        continue

    embedding = get_bert_embedding(row.content)
    new_embeddings.append(embedding)
    new_labels.append(row.label)

    # Save every 500 samples
    if (i + 1) % 500 == 0 or (i + 1) == len(df):
        train_embeddings = np.vstack([train_embeddings, new_embeddings])
        train_labels = np.hstack([train_labels, new_labels])
        np.save(checkpoint_embeddings, train_embeddings)
        np.save(checkpoint_labels, train_labels)
        print(f"💾 Saved checkpoint at {i + 1} samples...")
        new_embeddings, new_labels = [], []  # Reset buffer

# Load or train SVM model
if os.path.exists(model_path):
    print("🔄 Loading existing SVM model...")
    svm_model = joblib.load(model_path)
else:
    print("🚀 Training new SVM model...")
    svm_model = SVC(kernel="linear")

# Train or continue training SVM
print("📈 Training SVM model...")
svm_model.fit(train_embeddings, train_labels)
joblib.dump(svm_model, model_path)
print("✅ Model training completed!")
