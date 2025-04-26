import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# Step 1: Load the pretrained model to get the correct topic mapping
model_name = "dstefa/roberta-base_topic_classification_nyt_news"
model = RobertaForSequenceClassification.from_pretrained(model_name)

# Get the original label mapping (Ensures correct topic-to-ID conversion)
id2label = model.config.id2label
label2id = {v: k for k, v in id2label.items()}  # Reverse mapping

print("Original Labels from Pretrained Model:", id2label)

# Step 2: Load each dataset and assign labels
class_files = {
    "business": "dataset/balanced_fine_tuning_business_dataset.csv",
    "entertainment": "dataset/balanced_fine_tuning_Entertainment_dataset.csv",
    "health": "dataset/balanced_fine_tuning_Health and Wellness_dataset.csv",
    "politics": "dataset/balanced_fine_tuning_politics_dataset.csv",
    "science": "dataset/balanced_fine_tuning_science and tech_dataset.csv",
    "sports": "dataset/balanced_fine_tuning_sports_dataset.csv",
    "crime": "dataset/balanced_fine_tuning_Crime_dataset.csv",
    "lifestyle and fashion": "dataset/balanced_fine_tuning_Lifestyle_and_Fashion_dataset.csv"
}

# Step 3: Convert dataset topic names to match model's expected labels
dataset_topic_mapping = {
    "business": "Business and Finance",
    "entertainment": "Arts, Culture, and Entertainment",
    "health": "Health and Wellness",
    "politics": "Politics",
    "science": "Science and Technology",
    "sports": "Sports",
    "crime": "Crime",
    "lifestyle and fashion": "Lifestyle and Fashion"
}

# Step 4: Load datasets and map topics correctly
dataframes = []
for label, filepath in class_files.items():
    df = pd.read_csv(filepath)
    df['topic'] = dataset_topic_mapping[label]  # Convert to correct topic name
    df['topic_id'] = df['topic'].map(label2id)  # Convert to correct topic ID
    dataframes.append(df)

# Combine all datasets
combined_df = pd.concat(dataframes, ignore_index=True)

# Shuffle the dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add an index column to combined_df before conversion
combined_df["original_index"] = combined_df.index  

# Convert to Hugging Face Dataset
hf_dataset = Dataset.from_pandas(combined_df)

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained(model_name)

# Tokenize data and add labels
def preprocess_data(examples):
    encoding = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    encoding['labels'] = examples['topic_id']  # Ensure labels are included
    return encoding

hf_dataset = hf_dataset.map(preprocess_data, batched=True)
# Remove unnecessary columns but KEEP 'labels'
hf_dataset = hf_dataset.remove_columns(['topic', 'topic_id'])  # Keep 'labels' and 'text'

# Split into train (80%) and test (20%) sets
train_test_split = hf_dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = train_test_split['train']
test_dataset = train_test_split['test']  # Save this separately

# Step 6: Fine-Tuning Setup
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(id2label))

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    save_total_limit=2,  # Keep only the latest checkpoints
    load_best_model_at_end=True,  # Load the best model based on evaluation metrics
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,  # Use test dataset for evaluation
    tokenizer=tokenizer
)

trainer.train()


# Save the fine-tuned model
model.save_pretrained("./refined_roberta")
tokenizer.save_pretrained("./refined_roberta")

print("Fine-tuning complete! Model saved to './refined_roberta'")

# Extract original test dataset indices
test_indices = train_test_split["test"].to_pandas()["original_index"]

# Fetch original text using these indices
test_texts = test_dataset["text"]  # Get text directly from the dataset

# Get predictions
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)

# Convert IDs back to labels
test_labels = [id2label[label] for label in test_dataset['labels']]
predicted_labels = [id2label[pred] for pred in preds]

# Save test results to CSV
test_results = pd.DataFrame({
    "text": test_dataset["text"],
    "actual_label": test_labels,
    "predicted_label": predicted_labels
})

test_results.to_csv("test_results.csv", index=False)
print("Test results saved to 'test_results.csv'")