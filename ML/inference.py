
import argparse
from Model.Arrhythmia_classification import Arrhythmia_classification

if __name__ == '__main__':

    #[[46,0,163,86,99,163,393,150,113,-5,121,66,56,69.,42.,24.,68.,0.,0.,0.,27.7]]
    parser = argparse.ArgumentParser()
    # sagemaker
    parser.add_argument('--test_data', type=str, required=False, help='test inference data')
    opt = parser.parse_args()

    model = Arrhythmia_classification()

    result = model.inference(opt.test_data)

    print(f"result == {result}")

