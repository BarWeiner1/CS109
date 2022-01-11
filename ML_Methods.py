import csv
from collections import Counter

import numpy as np
import pandas as pd


class NaiveBayes():
    def __init__ (self, testingSet, trainingSet):
        self.testingSet,self.trainingSet  = testingSet, trainingSet
        self.Py0, self.Py1 = 0, 0
        self.Pxiy1, self.Pxiy0 = np.zeros(len(self.trainingSet.iloc[0])-1), np.zeros(len(self.trainingSet.iloc[0])-1)


    def trainModel(self):
        index = len(self.trainingSet.iloc[0])-1
        y0, y1 = 0, 0
        for i in range(len(self.trainingSet)):
            if self.trainingSet.iloc[i][index] == 0:
                y0 += 1
            else:
                y1 += 1
            for j in range(len(self.trainingSet.iloc[0])-1):
                if self.trainingSet.iloc[i][index] == 0 and self.trainingSet.iloc[i][j] == 1:
                    self.Pxiy0[j] += 1
                elif self.trainingSet.iloc[i][index] == 1 and self.trainingSet.iloc[i][j] == 1:
                    self.Pxiy1[j] += 1

        self.Py0, self.Py1 = y0/(y0+y1), y1/(y0+y1)

        #MLE
        self.Pxiy1 = self.Pxiy1 / y1
        self.Pxiy0 = self.Pxiy0/y0

        #Laplace
        self.Pxiy1 = (self.Pxiy1 + 1)/ (y1 +2)
        self.Pxiy0 = (self.Pxiy0 + 1)/ (y0 +2)

    def classifyTest(self):
        rL, pL = list(), list()
        for i in range(1, len(self.testingSet)):
            p0, p1 = np.log(self.Py0),np.log(self.Py1)
            for j in range(len(self.testingSet.iloc[0])-1):
                if self.testingSet.iloc[i][j] == 0:
                    p1 += np.log(1-self.Pxiy1[j])
                    p0 += np.log(1-self.Pxiy0[j])
                else:
                    p1 += np.log(self.Pxiy1[j])
                    p0 += np.log(self.Pxiy0[j])

            rL.append(self.testingSet.iloc[i][len(self.testingSet.iloc[j])-1])
            if p0 > p1:
                pL.append(0)
            else:
                pL.append(1)
        return rL, pL

    def evaluate(self):
        self.trainModel()
        A, probs = self.classifyTest()
        count_occurences = Counter(A)
        count_0, count_1 = 0, 0
        for i in range(len(probs)):
            if probs[i] == A[i]:
                if probs[i] == 0:
                    count_0 += 1
                elif probs[i] == 1:
                    count_1 += 1

        print(f"Class 0: tested {count_occurences[0]}, correctly classified {count_0} ")
        print(f"Class 1: tested {count_occurences[1]}, correctly classified {count_1} ")
        print(f"Overall: tested {count_occurences[0] + count_occurences[1]}, correctly classified {count_0 + count_1}")
        print("Accuracy = ", (count_0 + count_1)/(count_occurences[1]+count_occurences[0]))

testSet = pd.read_csv("heart-train.csv")
trainSet = pd.read_csv("heart-train.csv")
simple = NaiveBayes(testSet,trainSet)
simple.evaluate()

class logistic_regression():
    def __init__(self, learning_rate, num_steps):
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.weights,self.predictions,self.test_data  = None, None, None
        self.correct_0, self.correct_1 = 0, 0
        self.count_0, self.count_1 = 0, 0
        self.accuracy = None

    def train(self, train_features, train_labels):
        theta = np.zeros(train_features.shape[1])
        for i in range(self.num_steps):
            s = np.dot(train_features, theta)
            preds = sigmoid(s)
            gradient = np.dot(train_features.T, train_labels - preds)
            theta += self.learning_rate * gradient
        self.weights = theta

    def predict(self, dataset):
        self.test_data = pd.read_csv(dataset)
        predicted_correct = 0
        predictions = np.round(sigmoid(np.dot(np.array(self.test_data.iloc[:, :-1]), self.weights)))

        for c, r in self.test_data.iterrows():
            p = predictions[c]
            if p == r['Label']:
                if p == 0:
                    self.correct_0 += 1
                else:
                    self.correct_1 += 1
                predicted_correct += 1
                if r['Label'] == 0:
                    self.count_0 += 1
                else:
                    self.count_1 += 1
        self.accuracy = predicted_correct / (self.test_data.shape[0])


    def print_results(self):
        print(f'Class 0: tested {self.count_0}, correctly classified {self.correct_0}')
        print(f'Class 1: tested {self.count_1}, correctly classified {self.correct_1}')
        print(f'Overall: tested {self.count_0 + self.count_1}, correctly classified {self.correct_0 + self.correct_1}')
        print(f'Accuracy: {self.accuracy}')

def sigmoid(vec):
    return 1 / (1 + np.exp(-vec))

lr = logistic_regression(learning_rate = 0.0001,num_steps = 10000)
train = pd.read_csv('ancestry-train.csv')
target = train.iloc[:, -1]
train_data = train.iloc[:, :-1]
lr.train(train_data, target)
lr.predict('ancestry-test.csv')
lr.print_results()






