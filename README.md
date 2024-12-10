# Leaf Disease Classification Using CNN and Hyperparameter Optimization
## Overview
This project utilizes a Convolutional Neural Network (CNN) to classify leaves as either diseased or healthy using a labeled dataset of plant leaf images. The project implements Optuna, a Python library, for efficient hyperparameter optimization to improve model performance. The application of CNNs and hyperparameter tuning ensures high accuracy in identifying plant health, aiding in agricultural diagnostics.

## Key Features
Convolutional Neural Network: A deep learning model specifically designed for image classification tasks.
Data Augmentation and Preprocessing: Images are resized, normalized, and transformed for improved model generalization.
Hyperparameter Tuning with Optuna: Automated search for the best combination of learning rate, dropout rates, and batch size to enhance model performance.
Training and Testing: The dataset is split into 80% training and 20% testing data.
Results Visualization: Includes simple and interpretable graphs of training and testing accuracy for model evaluation.
Technologies Used
Programming Language: Python
Libraries:
Deep Learning: PyTorch, Torchvision
Data Handling: NumPy, Matplotlib
Hyperparameter Tuning: Optuna
Distributed Computing: PySpark (for parallelization support)
Project Workflow
Data Loading: Images of plant leaves are loaded using PyTorch's ImageFolder.
Data Splitting: The dataset is split into training (80%) and testing (20%) subsets.
Data Augmentation: Applied transformations such as resizing, normalization, and conversion to tensors.
Model Definition: A CNN architecture is built with two convolutional layers, max-pooling, dropout layers, and fully connected layers.
Hyperparameter Optimization:
Search Parameters: Learning rate, dropout rates, and batch size.
Objective: Maximize testing accuracy.
Optimizer: Adam optimizer is used with tuned parameters from Optuna.
Training: The model is trained for multiple epochs, with training loss monitored.
Evaluation: The model's performance is evaluated on the test set, and metrics such as accuracy are reported.
Results: Graphs of training vs. testing accuracy are plotted to analyze model performance.
How to Run the Project
Clone the repository:
bash
Copy code
git clone https://github.com/codefosu/Diseased-Plant-leaf-detection
Navigate to the project directory:
bash
Copy code
cd leaf-disease-classification
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Run the script:
bash
Copy code
python main.py
View the results and graphs generated after training.
Dataset
The dataset consists of images of healthy and diseased plant leaves stored in directories. Each subdirectory represents a class (e.g., healthy, diseased).

Path to Dataset: [/path/to/dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
Structure:
Copy code
dataset/
├── healthy/
└── diseased/
## Results
Accuracy: Achieved an accuracy of ~90% on the test dataset after hyperparameter tuning.
Graphs: Accuracy vs trial graphs are available
Introduce a validation set to improve generalization and prevent overfitting.
Extend the dataset with more diverse leaf images for better robustness.
Deploy the model as a web application for real-time plant health monitoring.
Acknowledgments
This project is inspired by the application of Artificial Intelligence in Agriculture.
Special thanks to the creators of the Optuna and PyTorch libraries
