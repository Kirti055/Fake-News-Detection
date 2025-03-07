import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import joblib
import os

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

# Load trained SVM model
model_path = "models/bert_svm_fake_news.pkl"

if not os.path.exists(model_path):
    print("❌ Model file not found! Train the model first using train.py")
    exit()

svm_model = joblib.load(model_path)
print("✅ Model loaded successfully!")

# Function to get BERT embeddings
def get_bert_embedding(text):
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = bert_model(**tokens)
    return output.last_hidden_state[:, 0, :].numpy().flatten()

# Take user input for news text
new_text = input("📰 Enter the news text: ").strip()

if not new_text:
    print("❌ No input provided. Please enter some text.")
    exit()

# Get embedding and make prediction
new_embedding = get_bert_embedding(new_text).reshape(1, -1)
prediction = svm_model.predict(new_embedding)

# Print result
print("📰 FAKE News ❌" if prediction[0] == 1 else "✅ REAL News ✅")
