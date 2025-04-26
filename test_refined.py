import torch
import pandas as pd
import numpy as np
from datasets import Dataset  
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

# Load the Pretrained RoBERTa Model
model_name = "dstefa/roberta-base_topic_classification_nyt_news"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name) 

# model_path = "./refined_roberta" 
# tokenizer = RobertaTokenizer.from_pretrained(model_path)
# model = RobertaForSequenceClassification.from_pretrained(model_path) 
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model.to(device).eval() 

# Load label mappings from the model
id2label = model.config.id2label
label2id = {v: k for k, v in id2label.items()}
 
# Load and Prepare the Test Dataset
test_df = pd.read_csv("test_set.csv")

# Convert labels to string format
test_df["label"] = test_df["label"].astype(str)
 
def preprocess_data(examples):
    encoding = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128) 
    return encoding

# Convert to Hugging Face Dataset
test_dataset = Dataset.from_pandas(test_df)
test_dataset = test_dataset.map(preprocess_data, batched=True)
test_dataset = test_dataset.remove_columns(["text", "label"])

# Convert dataset to PyTorch tensors
def collate_fn(batch):
    return {k: torch.tensor([dic[k] for dic in batch], dtype=torch.long) for k in batch[0]}

# Use DataLoader for Faster Inference
BATCH_SIZE = 32
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

# Perform Inference
all_predictions = []
with torch.no_grad():
    for batch in test_dataloader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        predicted_classes = torch.argmax(outputs.logits, dim=1)
        all_predictions.extend(predicted_classes.cpu().tolist())

# Map predictions back to labels
test_df["predicted_label"] = [id2label[pred] for pred in all_predictions]

# Convert predicted labels to string format
test_df["predicted_label"] = test_df["predicted_label"].astype(str)

# Compute Metrics
accuracy = accuracy_score(test_df["label"], test_df["predicted_label"])
report = classification_report(test_df["label"], test_df["predicted_label"], zero_division=0)

print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)

# Save results
test_df.to_csv("test_results.csv", index=False)
