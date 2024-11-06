import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier # test/comparison/validation w/ scikit-learn's MLP implementation

#outlier removal --
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

# -----------------


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
    def __init__(self, HeartDisease_fn, remove_outliers=True):
        self.HeartDisease_fn = HeartDisease_fn
        self.remove_outliers = remove_outliers

    def getTrainTestData(self):
        # load dataset

        HD_data, HD_labs = self.load_HD_data()

        self.N_features = HD_data.shape[1]
        self.N_examples = HD_data.shape[0]

        # Get feature names
        df = pd.read_csv(self.HeartDisease_fn)
        self.feature_names = list(df.columns[0:self.N_features])
        print(f'Feature names are: {self.feature_names}')

        if self.remove_outliers:
            # remove outliers from dataset
            #self.dataStats(self.feature_names, HD_data)
            HD_data = self.rmOutliers(HD_data, method='IForest')
            #self.dataStats(self.feature_names, HD_data)
            
            self.N_examples = HD_data.shape[0]

        print(f'Data has {self.N_features} features and {self.N_examples} examples.')

        # extract training and test data from dataset
        self.train_data, self.train_labs = self.load_HD_train_data((HD_data,HD_labs))
        self.test_data, self.test_labs = self.load_HD_test_data((HD_data,HD_labs))

        self.train_labs[self.train_labs == 0] = -1
        # -1 = No heart disease
        # 1 = heart disease

        #self.dataStats(self.feature_names, self.train_data)
        self.preprocessData()
        #self.dataStats(self.feature_names, self.train_data)


    def rmOutliers(self, data, method='IForest'):        
        if method == 'IForest':
            # Initialize the Isolation Forest model
            iso_forest = IsolationForest(random_state=42)  # 5% contamination threshold
            outliers = iso_forest.fit_predict(data)

            # Keep only the inliers
            data_no_outliers = data[outliers == 1]

        elif method == 'z-score':
            # Calculate the Z-scores for each feature
            z_scores = np.abs(zscore(data))

            # Define a threshold for Z-scores; common choice is 3 (for ~99.7% confidence level)
            threshold = 2
            data_no_outliers = data[(z_scores < threshold).all(axis=1)]

        elif method == 'IQR':
            threshold = 0.5
            data_no_outliers = data.copy()
            for i in range(data.shape[1]):  # Iterate over each feature
                Q1 = np.percentile(data[:, i], 25)
                Q3 = np.percentile(data[:, i], 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                data_no_outliers = data[(data[:, i] >= lower_bound) & (data[:, i] <= upper_bound)]
        
        else:
            print('Error using MLP.rmOutliers: method must be either "IForest", "z-score", or "IQR"')
            data_no_outliers = []

        return data_no_outliers


    def preprocessData(self):
        # standardize data to have mean = 0 and variance = 1
        #scaler = StandardScaler(with_mean=False, with_std=False)  
        scaler = StandardScaler()  
        # Don't cheat - fit only on training data
        scaler.fit(self.train_data)  
        self.train_data = scaler.transform(self.train_data)  
        # apply same transformation to test data
        #self.test_data = scaler.transform(self.test_data)  

        """
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
        """


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

    def load_HD_data(self):
        return self.load_data()

    # train and test data are subsets of the entire heart disease dataset

    def load_HD_train_data(self, data):
        train_data = data
        return train_data

    def load_HD_test_data(self, data):
        test_data = data
        return test_data


    def dataStats(self, data_headers, data):
        N_bins = int(np.sqrt(self.N_examples))

        data_1st_half = np.arange(0, self.N_features//2)
        data_2nd_half = np.arange(self.N_features//2, self.N_features)

        # --- histograms ---

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

        # --- box plots ---

        # Set up the subplot grid
        fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(15, 8))
        fig.suptitle('Histograms of Each Feature')

        # Flatten the axes array for easier indexing
        axes = axes.flatten()

        # Plot a histogram for each feature
        for i in range(self.N_features):
            axes[i].boxplot(data[:, i])
            axes[i].set_title(f'Box plot of {self.feature_names[i]}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')

        # Hide the last subplot (14th subplot) as we only have 13 features
        axes[-1].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the title
        plt.show()


    def train(self):
        theta = 0

        #self.train_data

        return theta


    def predict(self, theta):
        #self.test_data

        preds = []

        self.test_preds = preds

    
    def evaluate(self, labs, preds):
        pass



if __name__ == '__main__':
    HeartDisease_fn = 'heart.csv'  # Replace with your file path

    mlp = MLP(HeartDisease_fn, remove_outliers=True)

    mlp.getTrainTestData()

    mlp.dataStats(mlp.feature_names, mlp.train_data)
    #mlp.dataStats(mlp.feature_names, mlp.test_data)

    #theta = mlp.train()

    #mlp.predict(theta)

    #mlp.evaluate(mlp.test_labs, mlp.test_pred)

    