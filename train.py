import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

# Function to get embeddings
def get_bert_embedding(text):
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = bert_model(**tokens)
    return output.last_hidden_state[:, 0, :].numpy().flatten()

# Load dataset
df = pd.read_csv("data/fake_news_dataset.csv")  
df['label'] = df['label'].astype(int)

# Split dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'].values, df['label'].values, test_size=0.2, random_state=42)

# Convert texts to embeddings
train_embeddings = np.array([get_bert_embedding(text) for text in train_texts])
test_embeddings = np.array([get_bert_embedding(text) for text in test_texts])

# Train SVM
svm_model = SVC(kernel='linear')
svm_model.fit(train_embeddings, train_labels)

# Save model
joblib.dump(svm_model, "models/bert_svm_fake_news.pkl")

# Evaluate
predictions = svm_model.predict(test_embeddings)
print("Accuracy:", accuracy_score(test_labels, predictions))
