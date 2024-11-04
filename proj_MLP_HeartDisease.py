import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier # test/comparison/validation w/ scikit-learn's MLP implementation

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

class MLP():
    def __init__(self, HeartDisease_fn):
        self.HeartDisease_fn = HeartDisease_fn

        #data = extract_features(load_adult_train_data(fn))

        self.train_data, self.train_labs = self.load_HD_train_data()
        self.test_data, self.test_labs = self.load_HD_test_data()

        self.N_features = self.train_data.shape[1]
        self.N_examples = self.train_data.shape[0]
        print(f'Data has {self.N_features} features and {self.N_examples} examples.')

        self.train_labs[self.train_labs == 0] = -1
        # -1 = No heart disease
        # 1 = heart disease

        df = pd.read_csv(self.HeartDisease_fn)
        self.feature_names = list(df.columns[0:self.N_features])
        print(f'Feature names are: {self.feature_names}')

        self.dataStats2(self.feature_names, self.train_data)

        self.preprocessData()

        self.dataStats2(self.feature_names, self.train_data)


    def preprocessData(self):
        # standardize data to have mean = 0 and variance = 1
        #scaler = StandardScaler(with_mean=False, with_std=False)  
        scaler = StandardScaler()  
        # Don't cheat - fit only on training data
        scaler.fit(self.train_data)  
        self.train_data = scaler.transform(self.train_data)  
        # apply same transformation to test data
        #self.test_data = scaler.transform(self.test_data)  

        print("original data stats:")
        print(scaler.mean_)
        print(scaler.var_)
        print()

        print("transformed data stats:")
        print(np.mean(self.train_data, axis=0))
        print(np.var(self.train_data, axis=0))
        print()

        #print(self.train_data)
        #print(self.train_data[0])
        #print(len(self.train_data[0]))
        #print(self.train_labs)


    # --- Use dictionary for ML algorithm ---
    def load_data(self):
        with open(self.HeartDisease_fn) as f:
            header = f.readline()
            labels = []
            features = []
            for line in f.readlines():
                data = line.split(',')
                labels.append(int(data[-1]))
                features.append([float(x) for x in data[0:(len(data)-1)]])
            features = np.array(features)
            labels = np.array(labels)
            return features, labels

    # train and test data are subsets of the entire heart disease dataset

    def load_HD_data(self):
        return self.load_data()

    def load_HD_train_data(self):
        return self.load_HD_data()

    def load_HD_test_data(self):
        return self.load_HD_data()
    

    # --- Use pandas for statistical analysis --- 
    def dataStats(self):
        # Load CSV file into DataFrame

        df = pd.read_csv(self.HeartDisease_fn)

        N_features = len(df.columns) - 1

        print(df.columns[0:])

        # Plot histograms for the first 6 columns
        fig1, axes1 = plt.subplots(2, 3, figsize=(15, 8))
        for i, col in enumerate(df.columns[:6]):
            row, col_pos = divmod(i, 3)
            df[col].hist(ax=axes1[row, col_pos], bins=30)
            axes1[row, col_pos].set_title(f'Histogram of {col}')

        plt.tight_layout()

        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
        for i, col in enumerate(df.columns[:6]):
            row, col_pos = divmod(i, 3)
            df[col].plot.box(ax=axes2[row, col_pos])
            axes2[row, col_pos].set_title(f'Boxplot of {col}')

        plt.tight_layout()

        # Plot histograms for the last 7 columns
        fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))
        for i, col in enumerate(df.columns[7:13]):
            row, col_pos = divmod(i, 3)
            df[col].hist(ax=axes3[row, col_pos], bins=30)
            axes3[row, col_pos].set_title(col)

        plt.tight_layout()

        fig4, axes4 = plt.subplots(2, 3, figsize=(15, 8))
        for i, col in enumerate(df.columns[7:13]):
            row, col_pos = divmod(i, 3)
            df[col].plot.box(ax=axes4[row, col_pos])
            axes4[row, col_pos].set_title(f'Boxplot of {col}')

        plt.tight_layout()

        plt.show()


    def dataStats2(self, data_headers, data):
        N_bins = int(np.sqrt(self.N_examples))

        data_1st_half = np.arange(0, self.N_features//2)
        data_2nd_half = np.arange(self.N_features//2, self.N_features)

        # Set up the subplot grid
        fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(15, 8))
        fig.suptitle('Histograms of Each Feature')

        # Flatten the axes array for easier indexing
        axes = axes.flatten()

        # Plot a histogram for each feature
        for i in range(self.N_features):
            axes[i].hist(data[:, i], bins=20, color='skyblue', edgecolor='black')
            axes[i].set_title(f'Histogram of {self.feature_names[i]}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')

        # Hide the last subplot (14th subplot) as we only have 13 features
        axes[-1].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the title
        plt.show()


if __name__ == '__main__':
    HeartDisease_fn = 'heart.csv'  # Replace with your file path

    mlp = MLP(HeartDisease_fn)

    #mlp.dataStats()

    