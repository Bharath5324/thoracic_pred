
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

class ThoracicPredictor:
    le = dict()

    def __init__(self, data):
        self.data = pd.DataFrame(data)
        self.data.index = self.data['id']
        self.data.drop('id', axis = 1, inplace=True)
        for i in self.data.columns:
            if i != 'AGE' and i != 'PRE4' and i != 'PRE5':
                self.le[i] = LabelEncoder()
                self.le[i].fit(self.data[i])
                self.data[i] = self.le[i].transform(data[i])
        self.y = self.data['Risk1Yr']
        self.x = self.data.drop(['Risk1Yr'], axis = 1)
        self.columns = self.x.columns
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = 0.3)
        self.trainTest()

    def trainTest(self):
        self.classifier = SVC(kernel = 'poly', random_state = 1 )
        self.classifier.fit(self.x_train, self.y_train)
        self.y_pred = self.classifier.predict(self.x_test)
        self.getAccuracy()

    def getAccuracy(self):
        self.conf_mat = metrics.confusion_matrix(self.y_test, self.y_pred)
        TP = self.conf_mat[0,0]
        FP = self.conf_mat[1,0]
        TN = self.conf_mat[1,1]
        FN = self.conf_mat[0,1]

        self.sensitivity = TP/(TP+FN)
        self.specificity = TN/(TN+FP)
        self.precision = TP/(TP+FP)
        if TN == FN == 0:
            self.pred_val = "is invalid due to biased data"
        else:
            self.pred_val = TN/(TN + FN)
        self.accuracy = (TP + FN)/ (TP + TN + FP + FN)
        print(self.__repr__())


    def __repr__(self):
        return f" * The model is trained to an accuracy of  {self.accuracy}, sensitivity of {self.sensitivity}, specificity of {self.specificity}, precision of {self.precision} and to a predictive value of {self.pred_val}"

    def predict(self, data):
        data = pd.DataFrame(data, index=[0])
        for i in list(self.columns):
            if type(data[i][0]) == np.bool_:
                if data[i][0]:
                    data[i][0] = 'T'
                elif not data[i][0]:
                    data[i][0] = 'F'
            if i != 'AGE' and i != 'PRE4' and i != 'PRE5':
                data[i] = self.le[i].transform(data[i]) 
        print(data)
        return self.classifier.predict(data)[0]
    