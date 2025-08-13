# Sign-Language-MNIST-Neural-Network-Classifier

📌 Description
This project trains and evaluates a neural network classifier using scikit-learn’s MLPClassifier to recognize American Sign Language (ASL) letters from the Sign Language MNIST dataset.
The dataset consists of grayscale 28×28 pixel images of hand signs representing letters A–Y (excluding J and Z in many versions). The network learns to distinguish between 13 balanced classes in this version of the dataset.

The script:

Loads balanced training and testing CSV datasets.

Splits the training set into training and validation subsets.

Normalizes pixel values to [0, 1] for better training stability.

Trains an MLP (Multi-layer Perceptron) with 64 hidden neurons.

Reports:

Training, validation, and test accuracies

Per-class test accuracies

The top 3 most misidentified classes and their most common confusions

📂 Dataset Requirements
You’ll need two CSV files in the working directory:

sign_mnist_13bal_train.csv — balanced training dataset

sign_mnist_13bal_test.csv — balanced testing dataset

Each CSV should contain:

First column: class — integer label representing the letter’s class

Remaining columns: flattened pixel values (0–255) for each image

⚙️ How It Works
Data Loading & Normalization

Loads CSV files with Pandas.

Separates features (X) and labels (y).

Normalizes pixel values by dividing by 255.0.

Validation Split

Reserves 20% of the training set for validation (train_test_split).

Model Creation & Training

MLPClassifier with:

Hidden layer: 64 neurons

Tolerance for stopping: 0.005

max_iter: 1000

random_state: 42 for reproducibility

Evaluation Metrics

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
Common Misidentifications:
Class 2 most often confused with: Class 7 (15 times), Class 8 (10 times)
...
▶️ How to Run
Install dependencies:

bash
Copy
Edit
pip install pandas matplotlib scikit-learn numpy
Place sign_mnist_13bal_train.csv and sign_mnist_13bal_test.csv in the same folder as the script.

Run the script:

bash
Copy
Edit
python sign_mnist_classifier.py
📌 Notes
The hidden layer size (64) can be tuned for speed vs. accuracy.

The dataset is balanced — results may vary with unbalanced datasets.

You can adapt the script for more classes by changing the dataset.


