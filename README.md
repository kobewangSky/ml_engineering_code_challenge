# ml_engineering_code_challenge


1.docker build
```bash
$ git clone https://github.com/kobewangSky/ml_engineering_code_challenge.git
$ docker build -t ml_engineering_code_challenge .
```

2.docker run train
```bash
$ docker run -it --rm ml_engineering_code_challenge 
$ spark-submit --master local --num-executors 2 --executor-memory 1G --executor-cores 2 --driver-memory 1G train.py --datapath ./data/arrhythmia.data --test_size 0.2
```

3.docker run inference
```bash
$ docker run -it --rm ml_engineering_code_challenge 
$ spark-submit --master local --num-executors 2 --executor-memory 1G --executor-cores 2 --driver-memory 1G inference.py --test_data 46,0,163,86,99,163,393,150,113,-5,121,66,56,69.,42.,24.,68.,0.,0.,0.,27.7
```