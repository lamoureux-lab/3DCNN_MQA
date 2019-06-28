# 3DCNN_MQA: pytorch 1.0

This branch reproduces main results of the paper in pytorch 1.0

## Preparation
1. Download the [CASP_SCWRL](http://proteinfoldingproject.com/static/datasets/CASP_SCWRL.tar.gz) and [CASP11Stage1_SCWRL](http://proteinfoldingproject.com/static/datasets/CASP11Stage1_SCWRL.tar.gz) and [CASP11Stage2_SCWRL](http://proteinfoldingproject.com/static/datasets/CASP11Stage2_SCWRL.tar.gz) datasets, upack them to some location.

2. Change paths in <Dataset>/Description/*.dat to your locations

3. Install TorchProteinLibrary "dev" branch(d9ebec8cb17141f6c784e1c38b40db19d9d18dc1)


## Training and Test

1. Training script is *src/train_qa.py*

2. Testing script is *src/test_qa.py* or *run_server.sh*

3. Plotting training curves and evaluation on the CASP11 dataset 
can be done using *scripts/plotTrainingQA.py* and *scripts/plotTestQA.py*

Example training results:
<img src="https://github.com/lamoureux-lab/3DCNN_MQA/raw/pytorch1.0/results/Debug_correlations.png" width="400">

Example test results output:

```
CASP11Stage1:
Tau =  -0.3464883638231874
Pearson =  -0.5708989753301368
Loss =  0.08033214285714287
CASP11Stage2:
Tau =  -0.27394172704188546
Pearson =  -0.390677068797744
Loss =  0.07000843373493974
```