import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

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

# Define GAN components
latent_dim = 10

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Instantiate models
generator = Generator(latent_dim, X_train.shape[1])
discriminator = Discriminator(X_train.shape[1])

# Define loss function and optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

# GAN Training Loop
num_epochs = 10000
batch_size = 64
real_label = 1
fake_label = 0

for epoch in range(num_epochs):
    # Train Discriminator
    real_data = torch.tensor(X_train.sample(batch_size).values, dtype=torch.float32)
    real_labels = torch.full((batch_size, 1), real_label, dtype=torch.float32)
    optimizer_D.zero_grad()
    output = discriminator(real_data)
    d_loss_real = adversarial_loss(output, real_labels)
    
    # Train with fake data
    z = torch.randn(batch_size, latent_dim)
    fake_data = generator(z)
    fake_labels = torch.full((batch_size, 1), fake_label, dtype=torch.float32)
    output = discriminator(fake_data.detach())
    d_loss_fake = adversarial_loss(output, fake_labels)
    
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    optimizer_G.zero_grad()
    output = discriminator(fake_data)
    g_loss = adversarial_loss(output, real_labels)  # We want generator output to be classified as real
    g_loss.backward()
    optimizer_G.step()

    # Print losses every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}/{num_epochs} | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")

# Generate synthetic data
z = torch.randn(len(X_train), latent_dim)
synthetic_data = generator(z).detach().numpy()

# Combine synthetic and original data
synthetic_df = pd.DataFrame(synthetic_data, columns=X_train.columns)
augmented_X_train = pd.concat([X_train, synthetic_df], ignore_index=True)
augmented_y_train = pd.concat([y_train, y_train.sample(len(synthetic_df), replace=True)], ignore_index=True)

# Train model with augmented data
rf_model = RandomForestClassifier(random_state=42, class_weight=weights_dict)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20], 
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(augmented_X_train, augmented_y_train)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Report model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy after Data Augmentation: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
