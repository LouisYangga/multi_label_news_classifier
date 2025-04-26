import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
import ast

# Load dataset
df = pd.read_csv("dataset/Multi_Label_Dataset.csv", delimiter=";")

# Convert string labels to actual lists
df['labels'] = df['labels'].apply(ast.literal_eval)

# Multi-label binarization
mlb = MultiLabelBinarizer()
one_hot_encoded = mlb.fit_transform(df['labels'])

# Split data into train and validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), one_hot_encoded, test_size=0.2, random_state=42
)

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Tokenize the text data
train_encoding = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
val_encoding = tokenizer(val_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

# Custom Dataset
class MultiLabelDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset and dataloaders
train_dataset = MultiLabelDataset(train_encoding, train_labels)
val_dataset = MultiLabelDataset(val_encoding, val_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Load model
model = RobertaForSequenceClassification.from_pretrained(
    "./saved_models/refined_roberta",  # or use "roberta-base"
    num_labels=len(mlb.classes_),
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 15
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * epochs
)

# Training and Validation
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.cpu().numpy()
            predictions = (logits > 0).astype(int)
            labels = batch['labels'].cpu().numpy()

            all_preds.extend(predictions)
            all_true.extend(labels)

    print("Hamming Loss:", hamming_loss(all_true, all_preds))
    print("Subset Accuracy:", accuracy_score(all_true, all_preds))
    print(classification_report(all_true, all_preds, target_names=mlb.classes_))

# Save the model
model.save_pretrained("./saved_models/multilabel_roberta_model")
tokenizer.save_pretrained("./saved_models/multilabel_roberta_model")
