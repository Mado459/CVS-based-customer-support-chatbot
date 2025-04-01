from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BioBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = BertForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Example input: User asks a health-related question
user_input = "What does it mean if my blood pressure is high?"

# Tokenize the input and prepare it for the model
inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)

# Get the model's response
outputs = model(**inputs)
logits = outputs.logits

# Convert logits to probabilities (this step will depend on how you want to use BioBERT)
probs = torch.softmax(logits, dim=-1)

