from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()

# Feature names
feature_names = iris.feature_names

# Target names
target_names = iris.target_names

# Data (features)
data = iris.data

# Target (labels)
target = iris.target

# Create a DataFrame for the dataset
df = pd.DataFrame(data, columns=feature_names)

# Summary statistics
summary_stats = df.describe()

# Visualizations
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.hist(data[:, i], bins=20)
    plt.xlabel(feature_names[i])
    plt.ylabel("Frequency")

# Show a boxplot for each feature
df.boxplot()
plt.show()


# Check for missing values
missing_values = df.isnull().sum()
from sklearn.model_selection import train_test_split

# Split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

# Fit and transform the scaler on the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing data using the same scaler
X_test_scaled = scaler.transform(X_test)








from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Initialize the classifiers
logistic_classifier = LogisticRegression(solver='liblinear', multi_class='auto')
decision_tree_classifier = DecisionTreeClassifier()
svm_classifier = SVC()

# Train the models using the training data
logistic_classifier.fit(X_train_scaled, y_train)
decision_tree_classifier.fit(X_train_scaled, y_train)
svm_classifier.fit(X_train_scaled, y_train)

# Perform hyperparameter tuning using cross-validation (example for the Decision Tree)
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15]
}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_decision_tree = grid_search.best_estimator_

# You can perform similar hyperparameter tuning for other models as well.

# Once you've trained and tuned the models, you can use them for predictions on the test data.
y_pred_logistic = logistic_classifier.predict(X_test_scaled)
y_pred_decision_tree = best_decision_tree.predict(X_test_scaled)
y_pred_svm = svm_classifier.predict(X_test_scaled)

# Evaluate the models' performance on the test data using appropriate metrics.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Evaluate Logistic Regression model
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
precision_logistic = precision_score(y_test, y_pred_logistic, average='weighted')
recall_logistic = recall_score(y_test, y_pred_logistic, average='weighted')
f1_logistic = f1_score(y_test, y_pred_logistic, average='weighted')
confusion_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)

# Evaluate Decision Tree model
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
precision_decision_tree = precision_score(y_test, y_pred_decision_tree, average='weighted')
recall_decision_tree = recall_score(y_test, y_pred_decision_tree, average='weighted')
f1_decision_tree = f1_score(y_test, y_pred_decision_tree, average='weighted')
confusion_matrix_decision_tree = confusion_matrix(y_test, y_pred_decision_tree)

# Evaluate SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
confusion_matrix_svm = confusion_matrix(y_test, y_pred_svm)

# Print or use these metrics as needed to evaluate each model's performance.
print('svm accuracy: ', accuracy_svm)
print('logistic accuracy: ', accuracy_logistic)
print('decision tree accuracy: ', accuracy_decision_tree)
print('svm precision: ', precision_svm)
print('logistic precision: ', precision_logistic)
print('decision tree precision: ', precision_decision_tree)
print('svm recall: ', recall_svm)
print('logistic recall: ', recall_logistic)
print('decision tree recall: ', recall_decision_tree)
print('svm f1: ', f1_svm)
print('logistic f1: ', f1_logistic)
print('decision tree f1: ', f1_decision_tree)
print('svm confusion matrix: ', confusion_matrix_svm)
print('logistic confusion matrix: ', confusion_matrix_logistic)
print('decision tree confusion matrix: ', confusion_matrix_decision_tree)
