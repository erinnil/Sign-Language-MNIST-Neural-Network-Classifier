import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np

# Load the training dataset
train_data = pd.read_csv('sign_mnist_13bal_train.csv')

# Separate the data (features) and the classes
X_train = train_data.drop('class', axis=1)  # Features (all columns except the first one)
X_train = X_train / 255.0
y_train = train_data['class']   # Target (first column)

# Load the testing dataset
test_data = pd.read_csv('sign_mnist_13bal_test.csv')

# Separate the data (features) and the classes
X_test = test_data.drop('class', axis=1)  # Features (all columns except the first one)
X_test = X_test / 255.0
y_test = test_data['class']   # Target (first column)

# Create validation dataset by splitting training data (20% for validation)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create neural network with different hidden layer size (changed from 8 to 64 neurons)
neural_net_model = MLPClassifier(hidden_layer_sizes=(64), random_state=42, tol=0.005, max_iter=1000)

neural_net_model.fit(X_train, y_train)

# Determine model architecture 
layer_sizes = [neural_net_model.coefs_[0].shape[0]]  # Start with the input layer size
layer_sizes += [coef.shape[1] for coef in neural_net_model.coefs_]  # Add sizes of subsequent layers
layer_size_str = " x ".join(map(str, layer_sizes))
print(f"Training set size: {len(y_train)}")
print(f"Validation set size: {len(y_validate)}")
print(f"Test set size: {len(y_test)}")
print(f"Layer sizes: {layer_size_str}")

# Predict on all datasets
y_pred_train = neural_net_model.predict(X_train)
y_pred_validate = neural_net_model.predict(X_validate)
y_pred_test = neural_net_model.predict(X_test)

# Calculate accuracies
train_accuracy = (y_pred_train == y_train).mean() * 100
validate_accuracy = (y_pred_validate == y_validate).mean() * 100
test_accuracy = (y_pred_test == y_test).mean() * 100

print(f"----------")
print(f"Training Accuracy: {train_accuracy:.1f}%")
print(f"Validation Accuracy: {validate_accuracy:.1f}%")
print(f"Test Accuracy: {test_accuracy:.1f}%")

# Create dictionaries to hold total and correct counts for each class on test data
correct_counts = defaultdict(int)
total_counts = defaultdict(int)

# Count correct test predictions for each class
for true, pred in zip(y_test, y_pred_test):
    total_counts[true] += 1
    if true == pred:
        correct_counts[true] += 1

# Calculate and print accuracy for each class
print(f"----------")
print("Per-class Test Accuracies:")
for class_id in sorted(total_counts.keys()):
    accuracy = correct_counts[class_id] / total_counts[class_id] * 100
    print(f"Class {class_id}: {accuracy:3.0f}%")

# Generate confusion matrix to find most misidentified classes
conf_matrix = confusion_matrix(y_test, y_pred_test)
class_labels = sorted(list(set(y_test)))

# Calculate misidentification rates for each class
misidentification_rates = {}
for i, true_class in enumerate(class_labels):
    total_true = np.sum(conf_matrix[i, :])
    correct = conf_matrix[i, i]
    misidentified = total_true - correct
    misidentification_rate = (misidentified / total_true) * 100 if total_true > 0 else 0
    misidentification_rates[true_class] = misidentification_rate

# Find the 3 most misidentified classes
most_misidentified = sorted(misidentification_rates.items(), key=lambda x: x[1], reverse=True)[:3]

print(f"----------")
print("Most Misidentified Classes (letters):")
for class_id, rate in most_misidentified:
    print(f"Class {class_id}: {rate:.1f}% misidentification rate")

# Show what these misidentified classes are most often confused with
print(f"----------")
print("Common Misidentifications:")
for class_id, rate in most_misidentified:
    class_idx = class_labels.index(class_id)
    # Get the row for this class from confusion matrix
    row = conf_matrix[class_idx, :]
    # Set the correct prediction to 0 to find top misclassifications
    row_copy = row.copy()
    row_copy[class_idx] = 0
    # Find top 2 misclassifications
    top_mistakes = np.argsort(row_copy)[-2:][::-1]
    
    if row_copy[top_mistakes[0]] > 0:
        print(f"Class {class_id} most often confused with: ", end="")
        mistake_classes = []
        for mistake_idx in top_mistakes:
            if row_copy[mistake_idx] > 0:
                mistake_class = class_labels[mistake_idx]
                count = row_copy[mistake_idx]
                mistake_classes.append(f"Class {mistake_class} ({count} times)")
        print(", ".join(mistake_classes[:2]))
