import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier # test/comparison/validation w/ scikit-learn's MLP implementation
import random

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

        HD_labs[HD_labs == 0] = -1
        # -1 = No heart disease
        # 1 = heart disease

        self.N_features = HD_data.shape[1]
        self.N_examples = HD_data.shape[0]

        # Get feature names
        df = pd.read_csv(self.HeartDisease_fn)
        self.feature_names = list(df.columns[0:self.N_features])
        print(f'Feature names are: {self.feature_names}')

        if self.remove_outliers:
            # remove outliers from dataset
            #self.dataStats(self.feature_names, HD_data)
            HD_data, HD_labs = self.rmOutliers(HD_data, HD_labs, method='IForest')
            #self.dataStats(self.feature_names, HD_data)
            
            self.N_examples = HD_data.shape[0]

        print(f'Data has {self.N_features} features and {self.N_examples} examples.')

        dec = np.random.randint(low=-1, high=1) # decision for which half of data to look at as training or test data

        # extract training and test data from dataset
        self.train_data, self.train_labs = self.load_HD_train_data((HD_data,HD_labs), dec)
        self.test_data, self.test_labs = self.load_HD_test_data((HD_data,HD_labs), dec)

        #self.dataStats(self.feature_names, self.train_data)
        self.preprocessData()
        #self.dataStats(self.feature_names, self.train_data)


    def rmOutliers(self, data, labs, method='IForest'):        
        if method == 'IForest':
            # Initialize the Isolation Forest model
            iso_forest = IsolationForest(random_state=42)  # 5% contamination threshold
            outliers = iso_forest.fit_predict(data)

            # Keep only the inliers
            data_no_outliers = data[outliers == 1]
            labs_no_outliers = labs[outliers == 1]

        elif method == 'z-score':
            # Calculate the Z-scores for each feature
            z_scores = np.abs(zscore(data))

            # Define a threshold for Z-scores; common choice is 3 (for ~99.7% confidence level)
            threshold = 2
            data_no_outliers = data[(z_scores < threshold).all(axis=1)]
            labs_no_outliers = labs[(z_scores < threshold).all(axis=1)]

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
                labs_no_outliers = labs[(data[:, i] >= lower_bound) & (data[:, i] <= upper_bound)]

        
        else:
            print('Error using MLP.rmOutliers: method must be either "IForest", "z-score", or "IQR"')
            data_no_outliers = []
            labs_no_outliers = []

        return (data_no_outliers, labs_no_outliers)


    def preprocessData(self):
        # standardize data to have mean = 0 and variance = 1
        #scaler = StandardScaler(with_mean=False, with_std=False)  
        scaler = StandardScaler()  
        # Don't cheat - fit only on training data
        scaler.fit(self.train_data)  
        self.train_data = scaler.transform(self.train_data)  
        # apply same transformation to test data
        self.test_data = scaler.transform(self.test_data)  

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

        data = self.load_data()

        #features = data[0][0:5,:]
        #labs = data[1][0:5]
        #data = (features, labs)

        """

        features = data[0]
        labs = np.transpose(np.matrix(data[1]))

        D = np.concatenate((features, labs), axis=1)
        D = np.array(D)

        #print(D)
        np.random.shuffle(D)
        #print(D)

        N = D.shape[1] - 1

        features = D[:,0:(N-1)]

        labs = D[:,N]
        labs = np.array(labs, dtype=int)

        return features, labs
        """
        return data

    # train and test data are subsets of the entire heart disease dataset

    def load_HD_train_data(self, data, dec):
        N = data[0].shape[0] // 2

        if dec == -1:
            train_data = data[0][0:N, :]
            train_labs = data[1][0:N]
        else:
            train_data = data[0][(N + 1):, :]
            train_labs = data[1][(N + 1):]

        return (train_data, train_labs)

    def load_HD_test_data(self, data, dec):
        N = data[0].shape[0] // 2

        # choose opposite group from training data
        if dec == 1:
            test_data = data[0][0:N, :]
            test_labs = data[1][0:N]
        else:
            test_data = data[0][(N + 1):, :]
            test_labs = data[1][(N + 1):]

        return (test_data, test_labs)


    def dataStats(self, data):
        N_bins = int(np.sqrt(data.shape[0]))

        # --- histograms ---

        # Set up the subplot grid
        fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(15, 8))
        fig.suptitle('Histograms of Each Feature')

        # Flatten the axes array for easier indexing
        axes = axes.flatten()

        # Plot a histogram for each feature
        for i in range(self.N_features):
            axes[i].hist(data[:, i], bins=N_bins, color='skyblue', edgecolor='black')
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
        fig.suptitle('Box plots of Each Feature')

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


    # --- activation functions ---
    def tanh(self, a):
        return ( (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a)) )

    def sigmoid(self, a):
        return ( 1 / (1 + np.exp(-a)) )

    def gauss(self, a, mu, sigma):
        return ( np.exp(-0.5 * ((a - mu) / sigma)**2) )
        


    def initialize_weights(self, N_features, sz):
        M = len(sz) # total number of layers
        W = [0] * (M + 1)
        
        np.random.seed(seed=47)

        n = 0
        for m in range(-1, M):
            # if we are at the first layer.
            if m == -1:
                W[n] = np.random.randn(N_features+1, sz[m+1]) # including bias weight

            # if we are at the last layer.
            elif m == (M - 1):
                W[n] = np.random.randn(sz[m]+1, 1) # include bias weight

            else:
                W[n] = np.random.randn(sz[m]+1, sz[m+1]) # include bias weight

            n += 1

        return W

    def train(self, hidden_layers_sizes=(5,5,), iters=100, rate=1e-3):

        x = self.train_data
        y = np.transpose(np.matrix(self.train_labs))

        N_train_examples = x.shape[0]

        #n, d = x.shape # n = number of examples, d = number of features
        #print(n, d)

        b = np.ones((N_train_examples,1))
        x = np.concatenate((b,x), axis=1) # include bias value

        W = self.initialize_weights(self.N_features, hidden_layers_sizes) # random gauss. mu = 0, sigma = 1

        M = len(hidden_layers_sizes) # number of hidden layers
        print(M)

        Z = [0] * (M + 1)

        for I in range(iters):

            #--- forward propagation step ---
            n = 0
            for m in range(-1, M):
                w = W[n]

                # if we are at the first layer.
                if m == -1:
                    z_in = x
                else:
                    z_in = z_out

                d = np.matmul(np.transpose(w), np.transpose(z_in)) # dot product b/w weights and input neuron
                z_out = np.transpose(np.tanh(d)) # neuron's output
                
                if m == (M - 1):
                    Z[n] = z_out
                    #z[z < 0] = -1 # if z is negative, classify as no heart disease
                    #z[z > 0] = 1 # if z is positive, classify as heart disease
                else:
                    b = np.ones((N_train_examples,1))
                    z_out = np.concatenate((b, z_out), axis=1) # include bias value
                    Z[n] = z_out
                
                n += 1 # move to next layer

            #--- backward propagation step ---
            
            #output layer error
            delta = [0] * (M + 1)

            # calculate delta for last neuron
            delta[-1] = np.multiply((Z[-1] - y), (1 - np.power(Z[-1], 2))) # derivative of tanh activation

            # backpropagate the error for each hidden layer
            for m in range(M - 1, -2, -1):
                if m == -1:
                    z_in = x
                else:
                    z_in = Z[m]
                
                # compute the weight gradient and update weights for the current layer
                grad_w = np.matmul(np.transpose(z_in), delta[m + 1])
                W[m+1] = W[m+1] - rate * grad_w / self.N_examples  # goal is to minimize squared error, so subtract the gradient from the weights

                # compute delta for the next layer back
                if m > -1:
                    w_no_bias = W[m+1][1:, :] # Exclude bias for backpropagation
                    d = np.matmul(delta[m + 1], np.transpose(w_no_bias)) 
                    delta[m] = np.multiply(d, (1 - np.power(Z[m][:, 1:], 2)))

            #print error every few steps
            if I % 10 == 0:
                sq = np.array(np.square(y - Z[M]))
                loss = np.mean(sq)  # Sample loss (MSE)
                med = np.median(sq)
                max = np.max(sq)
                min = np.min(sq)
                print(f"Iteration {I}, Loss: {loss}, Median: {med}, Max: {max}, Min: {min}")

        return W


    def predict(self, theta):
        W = theta
        M = len(W) - 1 # number of hidden layers

        x = self.test_data
        N_test_examples = x.shape[0]

        b = np.ones((N_test_examples,1))
        x = np.concatenate((b,x), axis=1) # include bias value

        n = 0
        for m in range(-1, M):
            w = W[n]

            # if we are at the first layer.
            if m == -1:
                z_in = x
            else:
                z_in = z_out

            d = np.matmul(np.transpose(w), np.transpose(z_in))
            z_out = np.transpose(np.tanh(d))
            
            if m == (M - 1):
                z = z_out
            else:
                b = np.ones((N_test_examples,1))
                z_out = np.concatenate((b, z_out), axis=1) # include bias value

            n += 1 # move to next layer

        probs = np.array(np.transpose(z)) # probability values

        z[z < 0] = -1 # if z is negative, classify as no heart disease
        z[z > 0] = 1 # if z is positive, classify as heart disease

        z = np.array(np.transpose(z), dtype=int) # ensure output is an array of integer type

        return (z, probs)

    
    def evaluate(self, labels, preds, probs):

        preds[preds == -1] = 0
        TP = np.sum(preds == labels)   # correct
        FP = np.sum(preds == -labels)  # error
        preds[preds == 0] = -1

        preds[preds == 1] = 0
        TN = np.sum(preds == labels) #correct
        FN = np.sum(preds == -labels)  # error
        preds[preds == 0] = 1

        total_pred_pos = np.sum(preds[preds == 1])
        total_pred_neg = -np.sum(preds[preds == -1])
        #print((TP+FP) == total_pred_pos)
        #print((FN+TN) == total_pred_neg)

        total_real_pos = np.sum(labels[labels == 1])
        total_real_neg = -np.sum(labels[labels == -1])
        #print((TP+FN) == total_real_pos)
        #print((FP+TN) == total_real_neg)

        confusion_matrix = (TP, FP, TN, FN)

        acc = (TP + TN) / (TP + FN + FP + TN)
        #acc2 = np.mean(labels == preds)
        #print(np.abs(acc - acc2) < 1e-16)

        recall = TP / (TP + FN) # aka TPR

        FPR = FP / (FP + TN)

        precision = TP / (TP + FP)

        # for roc, need to step thresholds using probability. i.e.,
        # threshold = -1:0.1:1
        # for i = 0:len(threshold)
        #   probs[probs < threshold] = -1 # if z is negative, classify as no heart disease
        #   probs[probs > threshold] = 1 # if z is positive, classify as heart disease
        #   calculate_confusion_matrix(probs)
        #   TPR[i] =... , FPR[i] =...
        #
        # roc = plot(FPR, TPR)

        return (acc, recall, FPR, precision, confusion_matrix)



if __name__ == '__main__':
    HeartDisease_fn = 'heart.csv'  # Replace with your file path

    mlp = MLP(HeartDisease_fn, remove_outliers=True)

    mlp.getTrainTestData()

    #mlp.dataStats(mlp.train_data)
    #mlp.dataStats(mlp.test_data)

    theta = mlp.train(iters=1000, hidden_layers_sizes=(20,20,20,20,20,), rate=1.0)
    preds, probs = mlp.predict(theta)

    #n_examples = mlp.train_data.shape[0]
    #sk_mlp = MLPClassifier(hidden_layer_sizes=(20,20,20,20,20,), activation='tanh', solver='sgd', batch_size=n_examples, learning_rate_init=1/n_examples, max_iter=1000).fit(mlp.train_data, mlp.train_labs)
    #probs = sk_mlp.predict_proba(mlp.test_data)
    #preds = sk_mlp.predict(mlp.test_data)


    acc, recall, FPR, precision, confusion_matrix = mlp.evaluate(mlp.test_labs, preds, probs)
    print(f'Accuracy: {acc}, Recall: {recall}, FPR: {FPR}, Precision: {precision}')

    