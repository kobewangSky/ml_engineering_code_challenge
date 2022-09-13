import argparse
from Model.Arrhythmia_classification import Arrhythmia_classification
from pyspark.sql import SQLContext,SparkSession
from pyspark import SparkContext,SparkConf
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark = SparkSession.builder.appName("ReadWriteSpark").getOrCreate()
sparkcont = SparkContext.getOrCreate(SparkConf().setAppName("ReadWriteSpark"))
logs = sparkcont.setLogLevel("ERROR")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # sagemaker
    parser.add_argument('--datapath', type=str, required=True, help='data path')
    parser.add_argument('--test_size', type=float, required=True, help='test size rate')
    opt = parser.parse_args()

    model = Arrhythmia_classification(opt.datapath, opt.test_size)

    model.preparedata()

    model.train()