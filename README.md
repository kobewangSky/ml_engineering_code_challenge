# ml_engineering_code_challenge


1.docker build
```bash
$ git clone https://github.com/kobewangSky/ml_engineering_code_challenge.git
$ cd ml_engineering_code_challenge/ML/
$ docker build -t ml_engineering_code_challenge .
```

2.docker run train
```bash
$ docker run -it --rm ml_engineering_code_challenge 
$ spark-submit --master local --num-executors 2 --executor-memory 1G --executor-cores 2 --driver-memory 1G train.py --datapath ./data/arrhythmia.data --test_size 0.2
```
check here to see the training result 
https://wandb.ai/bluce54088/ml_engineering_code_challenge?workspace=user-


3.docker run inference
```bash
$ docker run -it --rm ml_engineering_code_challenge 
$ spark-submit --master local --num-executors 2 --executor-memory 1G --executor-cores 2 --driver-memory 1G inference.py --test_data 46,0,163,86,99,163,393,150,113,-5,121,66,56,69.,42.,24.,68.,0.,0.,0.,27.7
```

4.docker Data Analytics
```bash
$ docker run -it --rm ml_engineering_code_challenge 
$ spark-submit --master local --num-executors 2 --executor-memory 1G --executor-cores 2 --driver-memory 1G analytics.py --datapath ./data/arrhythmia.data
```
check here to see the Analytics result
https://wandb.ai/bluce54088/ml_engineering_code_challenge?workspace=user-
