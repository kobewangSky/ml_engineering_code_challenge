import pickle
import os

from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

import seaborn as sns
import scipy.stats as st

from Model.ModelAbstract import ModelAbstract
import pyspark as spark
from pyspark.sql import SparkSession
import wandb
import xgboost as xgb
from wandb.xgboost import wandb_callback

wandb.init(project="ml_engineering_code_challenge", entity="bluce54088")

import warnings
warnings.filterwarnings('ignore')

class Arrhythmia_classification(ModelAbstract):
    def __init__(self, datapath = '', test_size = 0.2, inference_data = [], remove_col = '20,-2'):
        self.test_size = test_size
        self.datapath = datapath
        self.inference_data = inference_data
        self.remove_col = remove_col
    def label_encoding(self, old_column):
        le = LabelEncoder()
        le.fit(old_column)
        new_column = le.transform(old_column)
        return new_column

    def preparedata(self):

        spark = SparkSession \
            .builder \
            .appName("Python Spark SQL basic example") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()

        self.df = spark.read.csv(self.datapath)
        self.df = self.df.toPandas()

        self.target = self.df[self.df.columns[len(self.df.columns) - 1]].name

        self.df.dropna(axis=0, inplace=True)
        if self.remove_col != None:
            print(f"remove area {self.remove_col}")
            rangeA, rangeB = self.remove_col.split(',')
            self.df.drop(self.df.columns[int(rangeA):int(rangeB)], axis=1, inplace=True)

        def function(x):
            if int(x) > 1:
                return 1
            else:
                return 0

        self.df[ self.target] = self.df[ self.target].apply(function)


        # encoding string parameters
        for i in self.df.columns:
            if type(self.df[i][0]) == str:
                self.df[i] = self.label_encoding(self.df[i])

        #self.df = self.df.apply(pd.to_numeric, errors='coerce')

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
        wandb.init(config=params)

        # Random Search Training with 5 folds Cross Validation
        model_result = RandomizedSearchCV(model_2, params, cv=5,
                                  n_jobs=1, n_iter=100)

        model_result.fit(X_train, y_train)

        pred_final = model_result.predict(X_test)

        print("accuracy is: ", accuracy_score(y_test, pred_final))
        wandb.log({"acc": accuracy_score(y_test, pred_final)})

        # Optional

        filename = f"{__name__}.pkl"
        pickle.dump(model_result, open(filename, 'wb'))
        #should push to cloud

        return model_result

    def inference(self ):
        input_data = self.inference_data_prepare(self.inference_data)
        filename = f"{__name__}.pkl"
        if os.path.exists(filename):
            loaded_model = pickle.load(open(filename, 'rb'))
        else:
            self.preparedata()
            loaded_model = self.train()

        result = loaded_model.predict(input_data)
        return result

    def DataAnalytics(self):

        self.preparedata()
        filename = self.plotScatterMatrix(self.df,20, 10)

        wandb.log({"visual": wandb.Image(filename)})



