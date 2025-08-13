# Sign-Language-MNIST-Neural-Network-Classifier

# 📌 What my project is / does
This project, Sign Language MNIST Neural Network Classifier, is a machine learning program that recognizes American Sign Language (ASL) hand signs from images.
It uses the Sign Language MNIST dataset, where each grayscale image represents a letter, and trains a neural network to classify them.
The model outputs overall accuracy, per-class accuracy, and identifies which signs are most often confused with others.

# 🎯 Why I made my project
I wanted to combine computer vision and machine learning to solve a real-world communication challenge: recognizing sign language.
This project was a chance to practice:
  Loading and preparing image datasets in CSV format
  Building and tuning a neural network from scratch with scikit-learn
  Analyzing model performance beyond just accuracy (confusion matrix, misidentifications)

# 🛠 How I made my project
1. Data Preparation
  Used pandas to load sign_mnist_13bal_train.csv and sign_mnist_13bal_test.csv.
  Normalized pixel values to a 0–1 range for better model stability.
  Split the training set into 80% training and 20% validation data.
2. Model Building
  Used MLPClassifier from scikit-learn with:
    784 input features (28×28 pixels)
    One hidden layer of 64 neurons
    max_iter=1000 for convergence
  Trained on the prepared training set.
3. Evaluation
  Measured accuracy for training, validation, and test sets.
  Calculated per-class accuracy.
  Generated a confusion matrix to see common mistakes.
  Identified the 3 most misidentified classes and what they’re usually confused with.

# ⚠️ What I struggled with & what I learned
Struggles:
At first, training accuracy was high but validation accuracy lagged, classic overfitting.
Balancing the hidden layer size and training iterations to get good accuracy without long training times.
Understanding confusion matrix indexing when figuring out misidentification patterns.
It took me a very long time to understand the system and finally debug all of my problems.

What I learned:
Normalizing image data is essential for faster convergence in neural networks.
Adding a validation split helps tune models more effectively and avoid overfitting.
The confusion matrix is a powerful tool for diagnosing where and why the model is wrong.
Even with good accuracy, some sign letters are naturally harder to distinguish (like visually similar hand shapes).

# The script:
Loads balanced training and testing CSV datasets.
Splits the training set into training and validation subsets.
Normalizes pixel values to [0, 1] for better training stability.
Trains an MLP (Multi-layer Perceptron) with 64 hidden neurons.

# Reports:
Training, validation, and test accuracies
Per-class test accuracies
The top 3 most misidentified classes and their most common confusions

# 📂 Dataset Requirements
You’ll need two CSV files in the working directory:
sign_mnist_13bal_train.csv — balanced training dataset
sign_mnist_13bal_test.csv — balanced testing dataset

# Each CSV should contain:
First column: class — integer label representing the letter’s class
Remaining columns: flattened pixel values (0–255) for each image

# ⚙️ How It Works
1. Data Loading & Normalization
  Loads CSV files with Pandas.
  Separates features (X) and labels (y).
  Normalizes pixel values by dividing by 255.0.
2. Validation Split
  Reserves 20% of the training set for validation (train_test_split).
3. Model Creation & Training
  MLPClassifier with:
    Hidden layer: 64 neurons
    Tolerance for stopping: 0.005
    max_iter: 1000
    random_state: 42 for reproducibility
4. Evaluation Metrics
  Calculates accuracies for:
    Training set
    Validation set
    Test set
  Calculates per-class accuracy.
  Generates a confusion matrix to find the 3 most misidentified classes and their most common wrong predictions.

📊 Example Output
  yaml
  Copy
  Edit
  Training set size: 8320
  Validation set size: 2080
  Test set size: 2600
  Layer sizes: 784 x 64 x 13
  ----------
  Training Accuracy: 99.5%
  Validation Accuracy: 95.7%
  Test Accuracy: 96.2%
  ----------
  Per-class Test Accuracies:
  Class 0: 98%
  Class 1: 95%
  ...
  ----------
  Most Misidentified Classes (letters):
  Class 2: 8.3% misidentification rate
  Class 5: 7.1% misidentification rate
  Class 9: 6.8% misidentification rate
  ----------

# Common Misidentifications:
Class 2 most often confused with: Class 7 (15 times), Class 8 (10 times)
...

# 📌 Notes
  The hidden layer size (64) can be tuned for speed vs. accuracy.
  The dataset is balanced — results may vary with unbalanced datasets.
  You can adapt the script for more classes by changing the dataset.


