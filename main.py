import pandas as pd
from logistic_regression import LogisticReg
from sklearn.model_selection import train_test_split
import warnings
import random

random.seed(10)  # Set random seed for reproducibility
warnings.filterwarnings("ignore")  # Suppress warnings

# Load data
data = pd.read_csv("edited_data.csv")
data = data.drop(data.index[200:], axis=0).dropna()  # Remove NaN values
X = data.drop(["Valentine_Date"], axis=1)  # Features
y = data["Valentine_Date"]  # Target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)  # Train-test split

# Train and evaluate model
model = LogisticReg()  # Initialize model
model.fit_model(X_train, y_train)  # Fit the model
predictions = model.make_predictions(X_test)  # Make predictions
accuracy = model.calculate_accuracy(predictions, y_test)  # Calculate accuracy
print("Accuracy:", accuracy)  # Print accuracy
