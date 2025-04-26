import torch
import numpy as np
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

import os

# Load tokenizer and model
model = RobertaForSequenceClassification.from_pretrained("./saved_models/multilabel_roberta_model")
tokenizer = RobertaTokenizer.from_pretrained("./saved_models/multilabel_roberta_model")
model.eval()

# Define categories (must match training labels)
classes = [
    "Arts, Culture, and Entertainment",
    "Business and Finance",
    "Health and Wellness",
    "Lifestyle and Fashion",
    "Politics",
    "Science and Technology",
    "Sports",
    "Crime"
]
mlb = MultiLabelBinarizer(classes=classes)
mlb.fit([classes])  # just to initialize it

# Input text
text = input("Enter your text: ")

# Tokenize and predict
inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)
    predicted = (probs > 0.5).int().cpu().numpy()

# Decode labels
predicted_labels = mlb.inverse_transform(predicted)

# Display
print("\nPredicted Categories:", predicted_labels[0] if predicted_labels else "None")
