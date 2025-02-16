import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import joblib

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

# Load trained SVM model
svm_model = joblib.load("models/bert_svm_fake_news.pkl")

# Function to get embeddings
def get_bert_embedding(text):
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = bert_model(**tokens)
    return output.last_hidden_state[:, 0, :].numpy().flatten()

# Test with a new input
new_text = input("Enter news text: ")
new_embedding = get_bert_embedding(new_text).reshape(1, -1)
prediction = svm_model.predict(new_embedding)
print("📰 Fake News" if prediction[0] == 1 else "✅ Real News")
