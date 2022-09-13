import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

import seaborn as sns
import scipy.stats as st
import argparse

from Model.ModelAbstract import ModelAbstract

import warnings
warnings.filterwarnings('ignore')


class Arrhythmia_classification(ModelAbstract):
    def __init__(self, datapath = '', test_size = 0.2, target = 'diagnosis', remove_filter = [] ):
        self.target = target
        self.test_size = test_size
        self.datapath = datapath
        self.remove_filter = remove_filter

    def label_encoding(self, old_column):
        le = LabelEncoder()
        le.fit(old_column)
        new_column = le.transform(old_column)
        return new_column

    def preparedata(self):
        df_ref = pd.read_csv("C:/Users/kobe/Downloads/data_arrhythmia.csv", sep=';')

        names = []
        for col in df_ref.columns:
            names.append(col)

        name_replace_map = dict(zip(list(range(280)), names))

        self.df = pd.read_csv(self.datapath, names=list(range(280)), sep=',', index_col=False)
        self.df = self.df.rename(columns=name_replace_map)

        self.df.dropna(axis=0, inplace=True)
        self.df.drop(self.df.columns[20:-2], axis=1, inplace=True)

        j = []
        for i in self.df.diagnosis:
            if i > 1:
                j.append(1)
            else:
                j.append(0)
        self.df.diagnosis = j

        # encoding string parameters
        for i in self.df.columns:
            if type(self.df[i][0]) == str:
                self.df[i] = self.label_encoding(self.df[i])

    def train(self):
        y = self.df[self.target].values
        x = self.df.drop([self.target], axis=1).values
        # spliting  data
        X_train, X_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=self.test_size)

        model_2 = XGBClassifier(eval_metric='mlogloss')

        params = {
            "n_estimators": st.randint(3, 40),
            "max_depth": st.randint(3, 40),
            "learning_rate": st.uniform(0.05, 0.4),
            "colsample_bytree": st.beta(10, 1),
            "subsample": st.beta(10, 1),
            "gamma": st.uniform(0, 10),
            'objective': ['binary:logistic'],
            'scale_pos_weight': st.randint(0, 2),
            "min_child_weight": st.expon(0, 50),

        }

        # Random Search Training with 5 folds Cross Validation
        model_result = RandomizedSearchCV(model_2, params, cv=5,
                                  n_jobs=1, n_iter=100)

        model_result.fit(X_train, y_train)

        pred_final = model_result.predict(X_test)

        print("accuracy is: ", accuracy_score(y_test, pred_final))

        filename = f"{__name__}.pkl"
        pickle.dump(model_result, open(filename, 'wb'))
        #should push to cloud

        return model_result

    def inference(self, input_data ):
        input_data = self.inference_data_prepare(input_data)
        filename = f"{__name__}.pkl"
        if os.path.exists(filename):
            loaded_model = pickle.load(open(filename, 'rb'))
        else:
            self.preparedata()
            loaded_model = self.train()

        result = loaded_model.predict(input_data)
        return result

# #changable parameters
# target = "diagnosis"
# test_size = 0.2
#
# df_ref = pd.read_csv("C:/Users/kobe/Downloads/data_arrhythmia.csv",sep=';')
#
# names = []
# for col in df_ref.columns:
#     names.append(col)
#
# name_replace_map = dict(zip(list(range(280)), names))
#
#
# df = pd.read_csv("./data/arrhythmia.data",names = list(range(280)),sep=',', index_col=False)
#
# df = df.rename(columns=name_replace_map)
#
# df.dropna(axis=0, inplace=True)
# df.drop(df.columns[20:-2],axis=1, inplace=True)
#
# #df.drop(['T','P','J','LG'],axis=1, inplace=True)
# #df = df.rename({278: 'diagnosis'}, axis=1)
#
# j = []
# for i in df.diagnosis:
#     if i > 1 :
#         j.append(1)
#     else:
#         j.append(0)
# df.diagnosis = j
# df.head()



# extracting x and y
# y = df[target].values
# x = df.drop([target], axis=1).values
# #spliting  data
# X_train, X_test, y_train, y_test = train_test_split(x,
#                                                     y,
#                                                     test_size=test_size)
#
#
#
#
# model_2 = XGBClassifier(eval_metric='mlogloss')
#
# params = {
#     "n_estimators": st.randint(3, 40),
#     "max_depth": st.randint(3, 40),
#     "learning_rate": st.uniform(0.05, 0.4),
#     "colsample_bytree": st.beta(10, 1),
#     "subsample": st.beta(10, 1),
#     "gamma": st.uniform(0, 10),
#     'objective': ['binary:logistic'],
#     'scale_pos_weight': st.randint(0, 2),
#     "min_child_weight": st.expon(0, 50),
#
# }
#
# # Random Search Training with 5 folds Cross Validation
# clf1 = RandomizedSearchCV(model_2, params, cv=5,
#                           n_jobs=1, n_iter=100)
#
# clf1.fit(X_train, y_train)
#
# pred_final = clf1.predict(X_test)
#
# print("accuracy is: ", accuracy_score(y_test, pred_final))

#######################################

# save models

# filename = 'clf2.pkl'
# pickle.dump(clf1, open(filename, 'wb'))
#
# print(clf1.best_params_)



# cm = confusion_matrix(y_test, pred_final)
# sns.set(rc={"figure.figsize":(4, 2)})
# sns.heatmap(cm, annot=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # sagemaker
    parser.add_argument('--datapath', type=str, required=True, help='data path')
    parser.add_argument('--test_size', type=float, required=True, help='test size rate')
    parser.add_argument('--pred_target', type=str, required=True, help='pred target')
    parser.add_argument('--save_path', type=str, required=True, help='model save path')
    opt = parser.parse_args()

    model = Arrhythmia_classification(opt.datapath, opt.test_size, opt.pred_target)

    model.prepare_data()

    model.train(opt.save_path)
