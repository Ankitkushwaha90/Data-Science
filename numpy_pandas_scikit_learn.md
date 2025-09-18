# Comprehensive Tutorial on NumPy, Pandas, and Scikit-learn (sklearn)

This tutorial provides a complete guide to **NumPy**, **Pandas**, and **Scikit-learn (sklearn)**, focusing on best practices, key concepts, and commonly used examples. We’ll start with NumPy (the foundation for numerical computing), move to Pandas (for data manipulation), and end with sklearn (for machine learning). Each section includes explanations, code examples, and outputs where relevant. 

**Prerequisites**: Ensure Python is installed and install the required libraries via:
```bash
pip install numpy pandas scikit-learn
```

Code examples are self-contained and can be run in a Jupyter Notebook or Python script for interactive learning. Best practices include using vectorized operations (avoid loops where possible), handling missing data properly, and splitting data for ML models.

---

## 1. NumPy: Numerical Computing Basics

NumPy is a library for efficient array operations, linear algebra, and random number generation. It’s the backbone for Pandas and sklearn.

### 1.1 Installation and Import

```python
import numpy as np
```

### 1.2 Creating Arrays

- **From lists**: Basic array creation.
- **Zeros/Ones/Empty**: Initialize arrays.
- **Arange/Linspace**: Generate sequences.

**Example**:
```python
# Array from list
arr = np.array([1, 2, 3, 4])
print(arr)  # Output: [1 2 3 4]

# 2D array
arr_2d = np.array([[1, 2], [3, 4]])
print(arr_2d)
# Output:
# [[1 2]
#  [3 4]]

# Zeros and ones
zeros = np.zeros((2, 3))  # 2x3 array of zeros
ones = np.ones((2, 3))    # 2x3 array of ones

# Arange (like range) and linspace (evenly spaced)
seq = np.arange(0, 10, 2)  # [0 2 4 6 8]
lin = np.linspace(0, 1, 5) # [0.   0.25 0.5  0.75 1.  ]
```

**Best Practice**: Use `np.array()` with explicit type control, e.g., `np.array([1, 2], dtype=np.float32)`.

### 1.3 Array Attributes and Operations

- **Shape, Size, Dtype**: Inspect arrays.
- **Reshape**: Change dimensions.
- **Broadcasting**: Operate on different shapes.

**Example**:
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # (2, 3)
print(arr.size)   # 6
print(arr.dtype)  # int64 (or similar)

# Reshape
reshaped = arr.reshape(3, 2)
print(reshaped)
# [[1 2]
#  [3 4]
#  [5 6]]

# Element-wise operations (vectorized)
add = arr + 10  # [[11 12 13] [14 15 16]]
mul = arr * 2   # [[ 2  4  6] [ 8 10 12]]

# Broadcasting: Add scalar or smaller array
vec = np.array([1, 2, 3])
broadcast = arr + vec  # Adds row-wise: [[2 4 6] [5 7 9]]
```

**Best Practice**: Leverage broadcasting to avoid loops for efficiency.

### 1.4 Indexing and Slicing

- Similar to Python lists but multi-dimensional.
- **Boolean indexing**: Filter based on conditions.

**Example**:
```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Slicing
slice_row = arr[1, :]    # [4 5 6]
slice_col = arr[:, 2]    # [3 6 9]
slice_sub = arr[1:3, 1:3] # [[5 6] [8 9]]

# Boolean
mask = arr > 5  # [[False False False] [False False  True] [ True  True  True]]
filtered = arr[mask]  # [6 7 8 9]
```

### 1.5 Mathematical Functions

- **Sum, Mean, Std**: Aggregations.
- **Dot product**: Matrix multiplication.
- **Random**: Generate data.

**Example**:
```python
arr = np.array([[1, 2], [3, 4]])

# Aggregations
print(np.sum(arr))     # 10
print(np.mean(arr))    # 2.5
print(np.std(arr))     # 1.118...

# Linear algebra
vec1 = np.array([1, 2])
vec2 = np.array([3, 4])
dot = np.dot(vec1, vec2)  # 11

mat_mul = np.dot(arr, arr)  # [[ 7 10] [15 22]]

# Random
rand_arr = np.random.rand(2, 3)  # Uniform [0,1)
rand_int = np.random.randint(1, 10, size=(2, 3))  # Integers
```

**Best Practice**: Use `np.random.seed(42)` for reproducibility in experiments.

### 1.6 Common Use Case: Vectorized Computations

Avoid loops for speed:

**Example**:
```python
# Bad: Loop
result = []
for i in range(1000):
    result.append(i * 2)

# Good: Vectorized
result_np = np.arange(1000) * 2
```

---

## 2. Pandas: Data Manipulation and Analysis

Pandas builds on NumPy for tabular data handling with **DataFrames** and **Series**.

### 2.1 Import and Basics

```python
import pandas as pd
```

### 2.2 Creating DataFrames and Series

- **Series**: 1D labeled array.
- **DataFrame**: 2D table.

**Example**:
```python
# Series
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=['a', 'b', 'c', 'd', 'e', 'f'])
print(s)
# a    1.0
# b    3.0
# ...

# DataFrame from dict
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'Score': [85, 90, 95]}
df = pd.DataFrame(data)
print(df)
#      Name  Age  Score
# 0   Alice   25     85
# 1     Bob   30     90
# 2 Charlie   35     95
```

**Best Practice**: Use meaningful indices and column names.

### 2.3 Reading/Writing Data

Commonly used for CSV, Excel, etc.

**Example**:
```python
# Read CSV (assume 'data.csv' exists)
# df = pd.read_csv('data.csv')

# Write
df.to_csv('output.csv', index=False)
```

### 2.4 Indexing and Selection

- **loc/iloc**: Label vs. position-based.
- **at/iat**: Single value.

**Example**:
```python
# loc (label-based)
print(df.loc[0:1, ['Name', 'Age']])
#     Name  Age
# 0  Alice   25
# 1    Bob   30

# iloc (position-based)
print(df.iloc[0, 0])  # 'Alice'

# Boolean filtering
high_score = df[df['Score'] > 90]
print(high_score)
#      Name  Age  Score
# 2 Charlie   35     95
```

### 2.5 Data Cleaning

- **Dropna, Fillna**: Handle missing values.
- **Drop duplicates**.

**Example**:
```python
df_with_nan = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
cleaned = df_with_nan.dropna()  # Drops rows with NaN
filled = df_with_nan.fillna(0)  # Fills NaN with 0

duplicates = pd.DataFrame({'A': [1, 1], 'B': [2, 2]})
unique = duplicates.drop_duplicates()
```

**Best Practice**: Impute missing values based on context (e.g., mean for numerical: `df.fillna(df.mean())`).

### 2.6 GroupBy and Aggregation

Group and apply functions like sum, mean.

**Example**:
```python
df = pd.DataFrame({'Group': ['A', 'A', 'B', 'B'],
                   'Value': [10, 20, 30, 40]})
grouped = df.groupby('Group').agg({'Value': ['sum', 'mean']})
print(grouped)
#       Value      
#         sum  mean
# Group             
# A        30  15.0
# B        70  35.0
```

### 2.7 Merging and Joining

Like SQL joins.

**Example**:
```python
df1 = pd.DataFrame({'Key': ['K1', 'K2'], 'A': [1, 2]})
df2 = pd.DataFrame({'Key': ['K1', 'K3'], 'B': [3, 4]})
merged = pd.merge(df1, df2, on='Key', how='left')
print(merged)
#   Key  A    B
# 0  K1  1  3.0
# 1  K2  2  NaN
```

### 2.8 Time Series (Common Use)

**Example**:
```python
dates = pd.date_range('20230101', periods=6)
df_ts = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df_ts.resample('M').mean())  # Monthly mean
```

**Best Practice**: Use `pd.to_datetime()` for date conversions.

### 2.9 Visualization Integration

Pandas integrates with Matplotlib.

**Example**:
```python
df.plot(kind='bar')  # Plots bar chart
```

---

## 3. Scikit-learn (sklearn): Machine Learning

Sklearn provides tools for classification, regression, clustering, etc. It follows a consistent API: `fit`, `predict`, `transform`.

### 3.1 Import

```python
from sklearn import datasets, model_selection, linear_model, metrics, preprocessing, cluster
```

### 3.2 Loading Data

Use built-in datasets.

**Example**:
```python
iris = datasets.load_iris()
X = iris.data
y = iris.target
```

### 3.3 Data Preprocessing

- **Scaling**: Normalize features.
- **Encoding**: Categorical to numerical.

**Example**:
```python
# Scaling
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encoding (for categorical)
encoder = preprocessing.LabelEncoder()
y_encoded = encoder.fit_transform(['red', 'blue', 'red'])
```

**Best Practice**: Always scale features for models like SVM or KNN.

### 3.4 Train-Test Split

**Example**:
```python
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
```

**Best Practice**: Use `random_state` for reproducibility; stratify for imbalanced classes: `stratify=y`.

### 3.5 Regression Example (Linear Regression)

Common for predicting continuous values.

**Example**:
```python
# Load data
diabetes = datasets.load_diabetes()
X_d = diabetes.data
y_d = diabetes.target
X_train_d, X_test_d, y_train_d, y_test_d = model_selection.train_test_split(X_d, y_d, test_size=0.2)

# Model
reg = linear_model.LinearRegression()
reg.fit(X_train_d, y_train_d)
preds = reg.predict(X_test_d)

# Evaluate
mse = metrics.mean_squared_error(y_test_d, preds)
print(mse)  # ~3000 (varies)
```

### 3.6 Classification Example (Logistic Regression)

For binary/multi-class.

**Example**:
```python
logreg = linear_model.LogisticRegression()
logreg.fit(X_train, y_train)
preds = logreg.predict(X_test)
accuracy = metrics.accuracy_score(y_test, preds)
print(accuracy)  # ~1.0 for Iris
```

Other classifiers: `from sklearn.svm import SVC`, `from sklearn.tree import DecisionTreeClassifier`.

### 3.7 Clustering Example (KMeans)

Unsupervised.

**Example**:
```python
kmeans = cluster.KMeans(n_clusters=3, n_init=10)  # n_init suppresses warning
kmeans.fit(X)
labels = kmeans.labels_
```

**Best Practice**: Use `silhouette_score` to evaluate: `metrics.silhouette_score(X, labels)`.

### 3.8 Cross-Validation

For better evaluation.

**Example**:
```python
scores = model_selection.cross_val_score(logreg, X, y, cv=5)
print(scores.mean())  # ~0.96
```

### 3.9 Pipelines

Combine steps.

**Example**:
```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', preprocessing.StandardScaler()),
                 ('classifier', linear_model.LogisticRegression())])
pipe.fit(X_train, y_train)
```

**Best Practice**: Use pipelines to avoid data leakage (e.g., scaling on train only).

### 3.10 Hyperparameter Tuning

Use GridSearch.

**Example**:
```python
from sklearn.model_selection import GridSearchCV
params = {'C': [0.1, 1, 10]}
grid = GridSearchCV(linear_model.LogisticRegression(), params, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)  # {'C': 1} (example)
```

---

## Best Practices Summary

- **NumPy**: Vectorize operations; check shapes before ops.
- **Pandas**: Chain methods (e.g., `df.query('Age > 30').groupby('Group').mean()`); avoid `inplace=True` for clarity.
- **Sklearn**: Split data early; use cross-validation; handle class imbalance with `class_weight='balanced'`.
- **General**: Profile code with `%timeit` in Jupyter; document with comments.

This covers 80% of common uses. Practice on datasets like **Iris** or **Boston Housing** for deeper understanding!
