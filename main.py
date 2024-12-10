# Import Spark libraries
from pyspark.sql import SparkSession
from pyspark import SparkConf

# Set up Spark session
conf = SparkConf().setAppName("LeafClassifier").setMaster("local[*]")
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# PyTorch and other libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import optuna
from collections import defaultdict

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to 128x128
    transforms.ToTensor(),          # Convert to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
])

# Load dataset
dataset = datasets.ImageFolder(root='/home/sat3812/Downloads/plants', transform=transform)

# Split data into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Record accuracies for plotting
trial_accuracies = defaultdict(list)

# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout1 = trial.suggest_float("dropout1", 0.1, 0.5)
    dropout2 = trial.suggest_float("dropout2", 0.3, 0.7)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Dataloader for this trial's batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the CNN model
    class LeafClassifierCNN(nn.Module):
        def __init__(self):
            super(LeafClassifierCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.dropout1 = nn.Dropout(dropout1)
            self.fc1 = nn.Linear(64 * 31 * 31, 128)
            self.dropout2 = nn.Dropout(dropout2)
            self.fc2 = nn.Linear(128, len(dataset.classes))
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = self.dropout1(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            return x

    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeafClassifierCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop (2 epochs for speed)
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Validation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    trial_accuracies["trial"].append(len(trial_accuracies["trial"]) + 1)
    trial_accuracies["accuracy"].append(accuracy)

    return accuracy

# Run Optuna study with fewer trials
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)  # Reduced to 5 trials

# Print best hyperparameters
print("Best hyperparameters:", study.best_params)

# Plot accuracies
plt.plot(trial_accuracies["trial"], trial_accuracies["accuracy"], marker='o', label="Validation Accuracy")
plt.xlabel("Trial")
plt.ylabel("Accuracy (%)")
plt.title("Hyperparameter Tuning: Trial vs Accuracy")
plt.legend()
plt.grid()
plt.show()

# Save the Spark session
spark.stop()
