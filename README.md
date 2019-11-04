# Pytorch 1.0, python 3.0
Branch pytorch1.0 contains the updated code for this paper.

# 3DCNN_MQA
Deep convolutional networks for fold recognition.
This repository has two branches: Release and helios. The Release branch
contains the code to replicate the main results of the publication. The 
helios branch contains complete code to replicate all the plots and tables in the publication, however it is sparsely documented.

## Preparation
1. Download the [CASP_SCWRL](https://drive.google.com/open?id=18cgfyStwa47qZzVGmgfM8lAUhyU_z9Hn) and [CASP11Stage1_SCWRL](https://drive.google.com/open?id=1bCZYmdf9del6XpB1tDwAfk9v7-M7BMVd) and [CASP11Stage2_SCWRL](https://drive.google.com/open?id=1WFddGJhpUG84ScXMAHtM8WxosyJvs3R2) datasets, upack them to some location.

2. Run the script *scripts/Datasets/heliosChangePrefix.py*. This script 
changes location of the datasets in the description files.

3. Compile the library. It compiles the CUDA procedures to load protein structures and other utilities. It requires Torch installation and CUDA version > 7.5. To build the library do the following steps:
  * cd src/Library
  * mkdir build
  * cd build 
  * cmake ..
  * make

## Training and Test

1. Launch training. The training takes approximatelly 2-3 days on TitanX Maxwell.
  * create directory *models* in the root of the repository (or somwhere else) and set *-models_dir* parameter everywhere accordingly
  * cd src/Training
  * change the *datasets_dir* parameter in the script *runTrainingAgregate.sh*
  and launch it

2. Test the model. The test phase takes around 2 hours.
  * cd src/Testing
  * change the *datasets_dir* parameter in the script *testModelSampled.sh*
  and launch it

3. In the end you will have the directory *models/name_of_experiment* where all
the output data is stored. To visualize the results:
  * cd scripts/Figures
  * change the *datasets_path* parameters in the scripts *plotTrainingProcess.py* and *plotTestResults.py*
  and launch them.

The script output should look like this:
```
Last validation result QA4:  -0.364180809477 -0.639459024499 0.0843657142857
Epoch 66 -0.403769280154 -0.676949232063 0.0836771428571
Epoch 42 -0.394173258982 -0.680343481073 0.112994285714
Epoch 22 -0.383505056489 -0.6535814553 0.0762371428571
...
```

The outputs will be stored in *models/name_of_experiment*. Examples of figures you will obtain are:
<table style="width:100%">
  <tr>
    <th>CASP11Stage2_SCWRL_sFinal_funnels</th>
    <th>kendall_validation</th>
  </tr>
  <tr>
    <th>
      <img src="https://github.com/lamoureux-lab/3DCNN_MQA/raw/Release/doc/CASP11Stage2_SCWRL_sFinal_funnels.png" width="400">
    </th>
    <th>
      <img src="https://github.com/lamoureux-lab/3DCNN_MQA/raw/Release/doc/kendall_validation.png" width="400">
    </th>
  </tr>
</table>

The script *plotTestResults.py* also outputs the performance measures on the 
test sets. Example output:

```
Test dataset:  CASP11Stage1_SCWRL
Num targets =  84
Excluded CAPRI target T0798
Excluded CAPRI target T0825
Excluded CAPRI target T0797
Num included targets =  81
Correlations:
Pearson =  -0.535246216884
Spearman =  -0.424928568326
Kendall =  -0.325462616687
Z-score: -1.2194271292
Loss: 0.0639580246914
```

The scipts *plotTestResults.py* and *plotTrainingProcess.py* also have the parameter *models_dir*, pointing to the directory with the saved experiments.

## Pretrained model
You can download [pretrained model](https://drive.google.com/open?id=1WpEujbJ2lmiTEgeaRybl0C4hABxd2IVJ) as well as the output of RWPlus, ProQ# and VoroMQA algorithms.

## Grad-CAM analysis
To replicate the Grad-CAM data from the paper you need to install PyMOL and its
python bindings. The main script that generates the tables is *src/Exploration/average_B_factors.py*. Before running this script change the parameters in the beginning, specifically *DATASETS_PATH*. Also in the main 
part change variables *generate*, *process*, and *make_table* to *True*. This script takes approximately 4 hours to finish.
The generated table can be then included in a LaTeX document and compiled using *pdflatex*. The raw data can be found in the directory *GradCAM* and 
the figures generated using PyMOL are in the directory *GradCAMOutput*. Here are a few examples:
<table style="width:100%">
  <tr>
    <th>T0762_BhageerathH_TS4</th>
    <th>T0762_MULTICOM-CONSTRUCT_TS1</th>
    <th>T0762</th>
  </tr>
  <tr>
    <th>
      <img src="https://github.com/lamoureux-lab/3DCNN_MQA/raw/Release/doc/T0762_BhageerathH_TS4.png" width="400">
    </th>
    <th>
      <img src="https://github.com/lamoureux-lab/3DCNN_MQA/raw/Release/doc/T0762_MULTICOM-CONSTRUCT_TS1.png" width="400">
    </th>
    <th>
      <img src="https://github.com/lamoureux-lab/3DCNN_MQA/raw/Release/doc/T0762_T0762.png" width="400">
    </th>
  </tr>
</table>
