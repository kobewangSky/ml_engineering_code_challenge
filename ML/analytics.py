import argparse
from Model.Arrhythmia_classification import Arrhythmia_classification
from pyspark.sql import SQLContext,SparkSession
from pyspark import SparkContext,SparkConf


spark = SparkSession.builder.appName("AnalyticsSpark").getOrCreate()
sparkcont = SparkContext.getOrCreate(SparkConf().setAppName("AnalyticsSpark"))
logs = sparkcont.setLogLevel("ERROR")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # sagemaker
    parser.add_argument('--datapath', type=str, required=True, help='data path')
    opt = parser.parse_args()

    model = Arrhythmia_classification(datapath=opt.datapath)

    model.DataAnalytics()