# ml_engineering_code_challenge


1.docker build
```bash
$ git clone https://github.com/kobewangSky/ml_engineering_code_challenge.git
$ docker build -t ml_engineering_code_challenge .
```

2.docker run train
```bash
$ docker run -it --rm ml_engineering_code_challenge 
$ python train.py --datapath ./data/arrhythmia.data --test_size 0.2 --pred_target diagnosis
```
