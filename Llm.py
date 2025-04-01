import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForTokenClassification, pipeline
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load and preprocess the cardiovascular dataset
file_path = 'Cardio1.csv'
data = pd.read_csv(file_path, delimiter=';')

# Select relevant columns and ensure numeric conversion in a single step
numeric_columns = ['age', 'height', 'weight', 'cholesterol', 'ap_hi', 'ap_lo', 'gluc']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Fill missing values and scale in one step
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns].fillna(data[numeric_columns].mean()))

# Handle class imbalance
y = data['cardio']
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
weights_dict = dict(enumerate(class_weights))

# Split data (train-test split)
X = data.drop(columns=['cardio'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training with hyperparameter tuning
rf_model = RandomForestClassifier(random_state=42, class_weight=weights_dict)
param_grid = {
    'n_estimators': [100, 200],  # Reduced search space for speedup
    'max_depth': [10, 20], 
    'min_samples_split': [2, 5]
}

# Use early stopping and parallelize grid search
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Report model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Load BioBERT model and tokenizer
bio_bert_tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
bio_bert_model = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Example text for cardiovascular analysis
example_text = "High blood pressure is a major risk factor for heart disease."

# Tokenize and run through the model (with no_grad for efficiency)
inputs = bio_bert_tokenizer(example_text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():  # Disable gradients for inference
    outputs = bio_bert_model(**inputs)
    
last_hidden_states = outputs.last_hidden_state
print("Shape of BioBERT hidden states:", last_hidden_states.shape)

# Load NER tokenizer and model
bio_bert_ner_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
bio_bert_ner_model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1")

# NER pipeline setup
ner_pipeline = pipeline("ner", model=bio_bert_ner_model, tokenizer=bio_bert_ner_tokenizer, aggregation_strategy="simple")

# Example NER
ner_results = ner_pipeline(example_text)
for entity in ner_results:
    print(f"Entity: {entity['word']}, Label: {entity['entity_group']}, Score: {entity['score']:.4f}")

# Define a custom Dataset class for Trainer compatibility
class CardiovascularDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.features.iloc[idx].values, dtype=torch.float32),
            'labels': torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        }

# Create datasets for Trainer
train_dataset = CardiovascularDataset(X_train, y_train)
eval_dataset = CardiovascularDataset(X_test, y_test)

# Define TrainingArguments with optimized settings
training_args = TrainingArguments(
    output_dir='./results',                   # Output directory
    evaluation_strategy="epoch",              # Evaluation strategy
    logging_dir='./logs',                     # Log directory
    per_device_train_batch_size=8,            # Train batch size
    per_device_eval_batch_size=8,             # Eval batch size
    num_train_epochs=3,                       # Number of epochs
    load_best_model_at_end=True,              # Load the best model at the end of training
    save_total_limit=2                        # Save only 2 checkpoints to save memory
)

# Trainer setup
trainer = Trainer(
    model=bio_bert_ner_model,                 # Pre-trained BioBERT model
    args=training_args,
    train_dataset=train_dataset,               # Cardiovascular training dataset
    eval_dataset=eval_dataset                   # Cardiovascular test dataset
)

# Fine-tune BioBERT model
trainer.train()

def query_biobert(user_input):
    # Efficient tokenization and inference
    inputs = bio_bert_tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    
    # Use no_grad for efficient inference
    with torch.no_grad():
        outputs = bio_bert_model(**inputs)
    
    # Extract and process logits (for sequence classification)
    logits = outputs.last_hidden_state.mean(dim=1)  # Average pooling for simplicity
    
    # For simplicity, returning a basic response
    return f"Cardiovascular risk based on input: {user_input}, Logits: {logits}"

# Sample query
user_query = "What are the risks of high cholesterol on heart health?"
response = query_biobert(user_query)
print(response)
