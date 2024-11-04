import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

# I will implement kernelized multilayered perceptron (KMLP)
# I will test the performance of KMLP against:
# - Logistic Regression (LR)
# - Naive Bayes (NB)
# - Support Vector Machine (SVM)
# - Desicion Tree (DT)
# - Random Forest (RF)

# scikit-learn.org/stable/index.html
# www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

"""
1. age
2. sex
3. chest pain type (4 values)
4. resting blood pressure
5. serum cholestoral in mg/dl
6. fasting blood sugar > 120 mg/dl
7. resting electrocardiographic results (values 0,1,2)
8. maximum heart rate achieved
9. exercise induced angina
10. oldpeak = ST depression induced by exercise relative to rest
11. the slope of the peak exercise ST segment
12. number of major vessels (0-3) colored by flourosopy
13. thal: 0 = normal; 1 = fixed defect; 2 = reversable defect

The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.

The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.

"""

# --- Use pandas for statistical analysis --- 
# Load CSV file into DataFrame
HeartDisease_fn = 'heart.csv'  # Replace with your file path
df = pd.read_csv(HeartDisease_fn)

N_features = len(df.columns) - 1

print(df.columns[0:])

# Plot histograms for the first 6 columns
fig1, axes1 = plt.subplots(2, 3, figsize=(15, 8))
for i, col in enumerate(df.columns[:6]):
    row, col_pos = divmod(i, 3)
    df[col].hist(ax=axes1[row, col_pos], bins=30)
    axes1[row, col_pos].set_title(f'Histogram of {col}')

fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
for i, col in enumerate(df.columns[:6]):
    row, col_pos = divmod(i, 3)
    df[col].plot.box(ax=axes2[row, col_pos])
    axes2[row, col_pos].set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()


# Plot histograms for the last 7 columns
fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))
for i, col in enumerate(df.columns[7:13]):
    row, col_pos = divmod(i, 3)
    df[col].hist(ax=axes3[row, col_pos], bins=30)
    axes3[row, col_pos].set_title(col)

fig4, axes4 = plt.subplots(2, 3, figsize=(15, 8))
for i, col in enumerate(df.columns[7:13]):
    row, col_pos = divmod(i, 3)
    df[col].plot.box(ax=axes4[row, col_pos])
    axes4[row, col_pos].set_title(f'Boxplot of {col}')


plt.tight_layout()
plt.show()


# --- Use dictionary for ML algorithm ---
def load_csv(filename):
    lines = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

# train and test data are subsets of the entire heart disease dataset

def load_HD_data(fn):
    return load_csv(fn)

def load_HD_train_data(fn):
    return load_HD_data(fn)

def load_HD_valid_data(fn):
    return load_HD_data(fn)

#data = extract_features(load_adult_train_data(fn))

data = load_HD_train_data(HeartDisease_fn)

print(len(data))
print(data[0])
print(len(data[0]))
print(data[0].keys())
print(data[14].values())
print(data[0]['age'])