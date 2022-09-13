import argparse
from Model.Arrhythmia_classification import Arrhythmia_classification

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # sagemaker
    parser.add_argument('--datapath', type=str, required=True, help='data path')
    parser.add_argument('--test_size', type=float, required=True, help='test size rate')
    parser.add_argument('--pred_target', type=str, required=True, help='pred target')
    parser.add_argument('--save_path', type=str, required=True, help='model save path')
    opt = parser.parse_args()

    model = Arrhythmia_classification(opt.datapath, opt.test_size, opt.pred_target)

    model.preparedata()

    model.train()