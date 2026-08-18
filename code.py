
#Link for google colab notebook - https://colab.research.google.com/drive/1iqQmhHIe13c4PB8cfp7L4CGa8-nyVg2d?usp=sharing


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""Column Types:
*   Numerical - age,educationno, capitalgain,capitalloss, hoursperweek
*   Categorical - workclass, maritalstatus, relationship, race, sex, possibility
*   Mixed - education, occupation




"""

#To read the csv file uploaded into google colab
df = pd.read_csv('/content/Dataset - missing_values-SalaryData_Train.csv')

df.head(15)

df.info()

#Gives the number of missing values in every column
missing_values = df.isnull().sum()

missing_values

#Gives the first 15 rows
df.sample(15)

df.info()

df.duplicated().sum()

#Gives the number of missing values in every column
df.isnull().sum()

missing_percentage = df.isnull().mean()*100
missing_percentage

#For categorical values, fill the missing data with mode

df['maritalstatus'] = df['maritalstatus'].fillna(df['maritalstatus'].mode()[0])

df['maritalstatus'].mode()[0]

df['race'] = df['race'].fillna(df['race'].mode()[0])

df['race'].mode()[0]

df['sex'] = df['sex'].fillna(df['sex'].mode()[0])

df['sex'].mode()[0]

#For non-categorical values, fill the missing data with mean
df['hoursperweek'] = df['hoursperweek'].fillna(df['hoursperweek'].mean())

df['hoursperweek'].mean()

df.isnull().sum()

#Drop the capitalgain and capitalloss columns since they are zero for most of the time

num_zeros_cg = (df['capitalgain'] == 0).sum()
print(f"Number of zeros in 'capitalgain' column: {num_zeros_cg}")
num_zeros_cl = (df['capitalloss'] == 0).sum()
print(f"Number of zeros in 'capitalloss' column: {num_zeros_cl}")

"""Univariate analysis on Categorical Columns"""

small_cat_columns = ['sex','Possibility']
big_cat_columns = ['maritalstatus','relationship','race','workclass']
#For two or three categories, we use pie-chart for univariate analysis
for col in small_cat_columns:
  print(f"Value counts for {col}:\n", df[col].value_counts())
  df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
  plt.title(f'Pie chart of {col}')
  plt.ylabel('')
  plt.show()

#For more than three categories we use bar plot for univariate analysis
for col in big_cat_columns:
    # Frequency distribution
    print(f"Value counts for {col}:\n", df[col].value_counts())

    # Bar plot
    sns.countplot(x=df[col])
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()

"""Univariate analysis for numerical columns
These are the numerical columns:
age,educationno, capitalgain,capitalloss, hoursperweek


"""

numerical_columns = ['age','educationno', 'capitalgain','capitalloss', 'hoursperweek']
for col in numerical_columns:
    # Summary statistics
    print(f"Summary statistics for {col}:\n", df[col].describe())

    # Histogram
    df[col].hist(bins=30, edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

    # KDE plot
    sns.kdeplot(df[col], shade=True)
    plt.title(f'Density plot of {col}')
    plt.show()

    # Box plot
    sns.boxplot(x=df[col])
    plt.title(f'Box plot of {col}')
    plt.show()

"""By looking at the barplots, it is true that the columns capitalgain and capitalloss are 0s for most of the time(capitalgain - 27623 and capitalloss - 28765 times out of the 30161 rows) and hence the mentioned columns need not be considered for further data anlysis and can be dropped from the dataframe"""

columns_to_drop = ['capitalgain','capitalloss']
df = df.drop(columns = columns_to_drop)
print(df.head())

"""Univariate analysis of Mixed Column Types"""

mixed_columns = ['education','occupation']

for col in mixed_columns:
    # Check the number of unique values to decide how to treat it
    unique_values = df[col].nunique()
    print(f"{col} has {unique_values} unique values")

    if unique_values < 10:  # Arbitrary threshold to treat as categorical
        # Bar plot for categorical treatment
        sns.countplot(x=df[col])
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.show()
    else:
        # Histogram for numerical treatment
        df[col].hist(bins=30, edgecolor='black')
        plt.title(f'Histogram of {col}')
        plt.xticks(rotation=45)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

"""Now, we concentrate on finding out the outliers. For this, we calculate the outliers for every column and flag them. Then, we go row-wise so as to count the number of columns in which each row is counted as an outlier. If the number of columns in which the row is counted as an outlier is more than or equal to 70% of the total number of columns, we term the row as an outlier and remove it.

We check whether it is an outlier or not using the IQR method(Inter-quartile range)
"""

# Function to detect numerical outliers using IQR
def detect_outliers_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Return a boolean mask where True indicates an outlier
    return (column < lower_bound) | (column > upper_bound)

# Function to detect outliers in categorical columns based on frequency
def detect_outliers_categorical(column, total_rows, percentage=0.5):
    threshold = total_rows * (percentage / 100)  # Calculate 0.5% of the total rows
    counts = column.value_counts()
    rare_categories = counts[counts < threshold].index
    # Return a boolean mask where True indicates an outlier
    return column.isin(rare_categories)

# Create a DataFrame to store outlier flags
outlier_flags = pd.DataFrame()

# Detect outliers in numerical columns using IQR
for col in df.select_dtypes(include=[np.number]).columns:
    outlier_flags[col] = detect_outliers_iqr(df[col]).astype(int)

# Detect outliers in categorical columns using 0.5% frequency threshold
total_rows = len(df)  # Total number of rows in the dataset
for col in df.select_dtypes(include=['object']).columns:
    outlier_flags[col] = detect_outliers_categorical(df[col], total_rows).astype(int)

# Count how many columns flag each row as an outlier
outlier_counts = outlier_flags.sum(axis=1)

# Define the threshold: rows flagged as outliers in more than 70% of the columns
num_columns = len(df.columns)
threshold_70_percent = int(0.7 * num_columns)

# Identify rows to remove (those with outliers in more than 70% of the columns)
rows_to_remove = outlier_counts > threshold_70_percent

# Remove the outlier rows
df_cleaned = df[~rows_to_remove]

# Check how many rows were removed and the resulting cleaned DataFrame
print(f"Rows removed: {rows_to_remove.sum()}")
print(f"Remaining rows: {len(df_cleaned)}")
print(df_cleaned.head())

df

from sklearn.preprocessing import LabelEncoder

df

df

df_new = df.copy()

"""Now, we use LabelEncoder to give labels for categorical columns so that they can also impact the model during the training."""

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Columns to encode
categorical_columns = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native']

# Apply LabelEncoder to each categorical column
for col in categorical_columns:
    df_new[col] = label_encoder.fit_transform(df_new[col])

df_new

df_new['Possibility'] = df_new['Possibility'].apply(lambda x: 0 if x == '<=0.5' else 1)

df_new

corr_matrix = df_new.corr()

"""A correlation matrix is created so as to check the correlation between every two set of columns(bi-variate analysis). It is concluded that no two columns have a strong positive or negative correlation."""

# Create a heatmap of the correlation matrix
plt.figure(figsize=(8, 6))  # Set the figure size
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()

df_new

"""The following code implements Naive Bayes theorem from scratch using only the numpy and pandas libraries. The classification is done as 0 or 1 where 0 denotes the possibility being less(<0.5) and 1 denotes the possibility of being more(>=0.5)."""

from collections import defaultdict

# Split features and target
X = df_new.drop('Possibility', axis=1).values
y = df_new['Possibility'].values

class NaiveBayes:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.feature_means = defaultdict(lambda: defaultdict(float))
        self.feature_vars = defaultdict(lambda: defaultdict(float))
        self.classes = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        # Calculate prior probabilities
        class_counts = np.bincount(y)
        self.class_probs = {cls: count / n_samples for cls, count in enumerate(class_counts)}

        # Calculate likelihoods for categorical features
        for cls in self.classes:
            X_cls = X[y == cls]
            n_cls_samples = X_cls.shape[0]
            for i in range(X.shape[1]):
                feature_values, feature_counts = np.unique(X_cls[:, i], return_counts=True)
                self.feature_probs[cls][i] = {value: count / n_cls_samples for value, count in zip(feature_values, feature_counts)}

        # Calculate mean and variance for numerical features
        for cls in self.classes:
            X_cls = X[y == cls]
            for i in range(X.shape[1]):
                if np.issubdtype(X_cls[:, i].dtype, np.number):  # Check if the feature is numerical
                    self.feature_means[cls][i] = np.mean(X_cls[:, i])
                    self.feature_vars[cls][i] = np.var(X_cls[:, i]) + 1e-6  # Adding a small constant to avoid division by zero

    def predict(self, X):
        predictions = []
        for sample in X:
            class_probs = {}
            for cls in self.classes:
                class_prob = np.log(self.class_probs[cls])
                feature_prob = 0
                for i, feature_value in enumerate(sample):
                    if feature_value in self.feature_probs[cls][i]:
                        feature_prob += np.log(self.feature_probs[cls][i][feature_value])
                    else:
                        feature_prob += np.log(1e-6)  # Smoothing for unseen feature values

                # Handling numerical features
                if np.issubdtype(type(sample[i]), np.number):
                    mean = self.feature_means[cls][i]
                    var = self.feature_vars[cls][i]
                    feature_prob += -0.5 * np.log(2 * np.pi * var) - ((sample[i] - mean) ** 2) / (2 * var)

                class_probs[cls] = class_prob + feature_prob
            predictions.append(max(class_probs, key=class_probs.get))
        return np.array(predictions)

# Initialize and train Naive Bayes model
nb = NaiveBayes()
nb.fit(X, y)

# Predict on training data
predictions = nb.predict(X)
print("Predictions:", predictions)
print("True labels:", y)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.2f}")

from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y, predictions)
recall = recall_score(y, predictions)
f1 = f1_score(y, predictions)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

value_counts = df_new['Possibility'].value_counts()
print("Value counts:")
print(value_counts)

from sklearn.metrics import classification_report

# Assuming y is the true labels and predictions are the model's predictions
print(classification_report(y, predictions))

"""The above mentioned are the accuracy, precision, recall and F1 scores for individual classes and the overall dataset.
1. Precision is the ratio of true positive predictions to the total number of positive predictions (both true positives and false positives).
2. Recall is the ratio of true positive predictions to the total number of actual positive instances (both true positives and false negatives).
3. The F1 score is the harmonic mean of precision and recall, providing a single metric to evaluate the balance between precision and recall.

The following code performs Naive Bayes(Library-implemented) on the dataset
Accuracy: 0.76
Precision: 0.53
Recall: 0.70
F1 Score: 0.60
"""

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Split features and target
X = df_new.drop('Possibility', axis=1)
y = df_new['Possibility']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predictions
y_pred = nb_model.predict(X_test)

# Evaluate
nb_accuracy = accuracy_score(y_test, y_pred)
nb_precision = precision_score(y_test, y_pred)
nb_recall = recall_score(y_test, y_pred)
nb_f1 = f1_score(y_test, y_pred)

# Print results
print(f"Naive Bayes - Accuracy: {nb_accuracy:.2f}, Precision: {nb_precision:.2f}, Recall: {nb_recall:.2f}, F1 Score: {nb_f1:.2f}")

"""The following code performs SVM on the dataset
Accuracy: 0.80
Precision: 0.72
Recall: 0.36
F1 Score: 0.48
"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Split features and target
X = df_new.drop('Possibility', axis=1)
y = df_new['Possibility']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train SVM
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# Predictions
y_pred = svm_model.predict(X_test)

# Evaluate
svm_accuracy = accuracy_score(y_test, y_pred)
svm_precision = precision_score(y_test, y_pred)
svm_recall = recall_score(y_test, y_pred)
svm_f1 = f1_score(y_test, y_pred)

# Print results
print(f"SVM - Accuracy: {svm_accuracy:.2f}, Precision: {svm_precision:.2f}, Recall: {svm_recall:.2f}, F1 Score: {svm_f1:.2f}")

"""The following code performs Decision Tree on the dataset
Accuracy: 0.78
Precision: 0.56
Recall: 0.55
F1 Score: 0.55
"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Split features and target
X = df_new.drop('Possibility', axis=1)
y = df_new['Possibility']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Predictions
y_pred = dt_model.predict(X_test)

# Evaluate
dt_accuracy = accuracy_score(y_test, y_pred)
dt_precision = precision_score(y_test, y_pred)
dt_recall = recall_score(y_test, y_pred)
dt_f1 = f1_score(y_test, y_pred)

# Print results
print(f"Decision Tree - Accuracy: {dt_accuracy:.2f}, Precision: {dt_precision:.2f}, Recall: {dt_recall:.2f}, F1 Score: {dt_f1:.2f}")

"""The following code performs KNN on the dataset
Accuracy: 0.79
Precision: 0.61
Recall: 0.56
F1 Score: 0.58
"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Split features and target
X = df_new.drop('Possibility', axis=1)
y = df_new['Possibility']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Predictions
y_pred = knn_model.predict(X_test)

# Evaluate
knn_accuracy = accuracy_score(y_test, y_pred)
knn_precision = precision_score(y_test, y_pred)
knn_recall = recall_score(y_test, y_pred)
knn_f1 = f1_score(y_test, y_pred)

# Print results
print(f"KNN - Accuracy: {knn_accuracy:.2f}, Precision: {knn_precision:.2f}, Recall: {knn_recall:.2f}, F1 Score: {knn_f1:.2f}")

"""The following code performs Ensemble modelling(without using any ensemble libraries) by combining the above classifiers so as to produce more accuracy.

After performing, the reuslts are
Ensemble Model - Accuracy: 0.81
Ensemble Model - Precision: 0.64
Ensemble Model - Recall: 0.60
Ensemble Model - F1 Score: 0.62
Naive Bayes (sklearn) - Accuracy: 0.76, Precision: 0.53, Recall: 0.70, F1 Score: 0.60
SVM - Accuracy: 0.80, Precision: 0.72, Recall: 0.36, F1 Score: 0.48
Decision Tree - Accuracy: 0.78, Precision: 0.57, Recall: 0.55, F1 Score: 0.56
KNN - Accuracy: 0.79, Precision: 0.61, Recall: 0.56, F1 Score: 0.58
Custom Naive Bayes - Accuracy: 0.80, Precision: 0.58, Recall: 0.75, F1 Score: 0.66
"""

import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Custom Naive Bayes
class NaiveBayes:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.feature_means = defaultdict(lambda: defaultdict(float))
        self.feature_vars = defaultdict(lambda: defaultdict(float))
        self.classes = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        # Calculate prior probabilities
        class_counts = np.bincount(y)
        self.class_probs = {cls: count / n_samples for cls, count in enumerate(class_counts)}

        # Calculate likelihoods for categorical features
        for cls in self.classes:
            X_cls = X[y == cls]
            n_cls_samples = X_cls.shape[0]
            for i in range(X.shape[1]):
                feature_values, feature_counts = np.unique(X_cls[:, i], return_counts=True)
                self.feature_probs[cls][i] = {value: count / n_cls_samples for value, count in zip(feature_values, feature_counts)}

        # Calculate mean and variance for numerical features
        for cls in self.classes:
            X_cls = X[y == cls]
            for i in range(X.shape[1]):
                if np.issubdtype(X_cls[:, i].dtype, np.number):  # Check if the feature is numerical
                    self.feature_means[cls][i] = np.mean(X_cls[:, i])
                    self.feature_vars[cls][i] = np.var(X_cls[:, i]) + 1e-6  # Adding a small constant to avoid division by zero

    def predict(self, X):
        predictions = []
        for sample in X:
            class_probs = {}
            for cls in self.classes:
                class_prob = np.log(self.class_probs[cls])
                feature_prob = 0
                for i, feature_value in enumerate(sample):
                    if feature_value in self.feature_probs[cls][i]:
                        feature_prob += np.log(self.feature_probs[cls][i][feature_value])
                    else:
                        feature_prob += np.log(1e-6)  # Smoothing for unseen feature values

                # Handling numerical features
                if np.issubdtype(type(sample[i]), np.number):
                    mean = self.feature_means[cls][i]
                    var = self.feature_vars[cls][i]
                    feature_prob += -0.5 * np.log(2 * np.pi * var) - ((sample[i] - mean) ** 2) / (2 * var)

                class_probs[cls] = class_prob + feature_prob
            predictions.append(max(class_probs, key=class_probs.get))
        return np.array(predictions)

# Initialize models
sklearn_models = {
    'Naive Bayes (sklearn)': GaussianNB(),
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier()
}

# Prepare data (assuming df_new has already been defined and processed)
X = df_new.drop('Possibility', axis=1).values
y = df_new['Possibility'].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and predict using sklearn models
model_predictions = {}

for name, model in sklearn_models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Predict hard labels
    y_pred = model.predict(X_test)

    # Store predictions
    model_predictions[name] = y_pred

# Train and predict using custom Naive Bayes
custom_nb = NaiveBayes()
custom_nb.fit(X_train, y_train)
custom_nb_predictions = custom_nb.predict(X_test)
model_predictions['Custom Naive Bayes'] = custom_nb_predictions

# Ensemble Voting (Hard Voting)
def ensemble_voting(predictions_dict):
    # Transpose the predictions so we can get a list of predictions for each sample across models
    model_predictions_list = np.array(list(predictions_dict.values())).T

    # Majority voting: For each sample, find the most common prediction
    final_predictions = [np.bincount(pred).argmax() for pred in model_predictions_list]
    return np.array(final_predictions)

# Get ensemble predictions
ensemble_preds = ensemble_voting(model_predictions)

# Evaluate the ensemble model
accuracy = accuracy_score(y_test, ensemble_preds)
precision = precision_score(y_test, ensemble_preds)
recall = recall_score(y_test, ensemble_preds)
f1 = f1_score(y_test, ensemble_preds)

print(f"Ensemble Model - Accuracy: {accuracy:.2f}")
print(f"Ensemble Model - Precision: {precision:.2f}")
print(f"Ensemble Model - Recall: {recall:.2f}")
print(f"Ensemble Model - F1 Score: {f1:.2f}")

# Compare individual models
for name, preds in model_predictions.items():
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"{name} - Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1 Score: {f1:.2f}")

"""The following code snippet is to produce a comparison graph between all the classifier models used in this collab notebook."""

# Define the model names and their corresponding accuracies
model_names = [
    'Custom Naive Bayes',
    'Naive Bayes (sklearn)',
    'SVM',
    'Decision Tree',
    'KNN',
    'Ensemble Model'
]
accuracies = [
    0.80,  # Accuracy of Custom Naive Bayes
    0.76,         # Accuracy of Naive Bayes (sklearn)
    0.80,        # Accuracy of SVM
    0.78,         # Accuracy of Decision Tree
    0.79,        # Accuracy of KNN
    0.81   # Accuracy of Ensemble Model
]

# Create a bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, accuracies, color='skyblue')

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.ylim(0, 1)  # Assuming accuracy is in the range [0, 1]

# Add value labels on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02,
        f'{height:.2f}',
        ha='center',
        va='bottom'
    )

# Rotate x labels for better readability if necessary
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()