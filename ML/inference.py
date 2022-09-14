
import argparse
from Model.Arrhythmia_classification import Arrhythmia_classification
from pyspark.sql import SQLContext,SparkSession
from pyspark import SparkContext,SparkConf
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark = SparkSession.builder.appName("InferenceSpark").getOrCreate()
sparkcont = SparkContext.getOrCreate(SparkConf().setAppName("InferenceSpark"))
logs = sparkcont.setLogLevel("ERROR")

if __name__ == '__main__':

    #[[46,0,163,86,99,163,393,150,113,-5,121,66,56,69.,42.,24.,68.,0.,0.,0.,27.7]]
    parser = argparse.ArgumentParser()
    # sagemaker
    parser.add_argument('--test_data', type=str, required=False, help='test inference data')
    opt = parser.parse_args()

    model = Arrhythmia_classification(inference_data=opt.test_data)

    result = model.inference()

    print(f"result == {result}")

