# Note
This is the **actual MRNet baseline model** following [Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699) by **Stanford University**, to run this you need to create a new virtual environment and install several python libraries including,

```
conda list

#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                 conda_forge    conda-forge
_openmp_mutex             4.5                       2_gnu    conda-forge
absl-py                   2.1.0              pyhd8ed1ab_0    conda-forge
blas                      1.1                    openblas    conda-forge
bzip2                     1.0.8                h5eee18b_6  
c-ares                    1.34.3               heb4867d_0    conda-forge
ca-certificates           2024.9.24            h06a4308_0  
expat                     2.6.3                h6a678d5_0  
filelock                  3.16.1                   pypi_0    pypi
fsspec                    2024.10.0                pypi_0    pypi
grpcio                    1.62.2          py312h6a678d5_0  
imageio                   2.36.0                   pypi_0    pypi
importlib-metadata        8.5.0              pyha770c72_0    conda-forge
jinja2                    3.1.4                    pypi_0    pypi
joblib                    1.4.2                    pypi_0    pypi
lazy-loader               0.4                      pypi_0    pypi
ld_impl_linux-64          2.40                 h12ee557_0  
libabseil                 20240116.2      cxx17_h6a678d5_0  
libffi                    3.4.4                h6a678d5_1  
libgcc                    14.2.0               h77fa898_1    conda-forge
libgcc-ng                 14.2.0               h69a702a_1    conda-forge
libgfortran               14.2.0               h69a702a_1    conda-forge
libgfortran5              14.2.0               hd5240d6_1    conda-forge
libgomp                   14.2.0               h77fa898_1    conda-forge
libgrpc                   1.62.2               h2d74bed_0  
libopenblas               0.3.28          pthreads_h94d23a6_1    conda-forge
libprotobuf               4.25.3               he621ea3_0  
libstdcxx-ng              11.2.0               h1234567_1  
libuuid                   1.41.5               h5eee18b_0  
markdown                  3.6                pyhd8ed1ab_0    conda-forge
markupsafe                3.0.2              pyhe1237c8_0    conda-forge
mpmath                    1.3.0                    pypi_0    pypi
ncurses                   6.4                  h6a678d5_0  
networkx                  3.4.2                    pypi_0    pypi
numpy                     2.1.3                    pypi_0    pypi
numpy-base                2.0.1           py312he1a6c75_1  
nvidia-cublas-cu12        12.4.5.8                 pypi_0    pypi
nvidia-cuda-cupti-cu12    12.4.127                 pypi_0    pypi
nvidia-cuda-nvrtc-cu12    12.4.127                 pypi_0    pypi
nvidia-cuda-runtime-cu12  12.4.127                 pypi_0    pypi
nvidia-cudnn-cu12         9.1.0.70                 pypi_0    pypi
nvidia-cufft-cu12         11.2.1.3                 pypi_0    pypi
nvidia-curand-cu12        10.3.5.147               pypi_0    pypi
nvidia-cusolver-cu12      11.6.1.9                 pypi_0    pypi
nvidia-cusparse-cu12      12.3.1.170               pypi_0    pypi
nvidia-nccl-cu12          2.21.5                   pypi_0    pypi
nvidia-nvjitlink-cu12     12.4.127                 pypi_0    pypi
nvidia-nvtx-cu12          12.4.127                 pypi_0    pypi
openblas                  0.3.28          pthreads_h6ec200e_1    conda-forge
openssl                   3.4.0                hb9d3cd8_0    conda-forge
packaging                 24.2               pyhd8ed1ab_0    conda-forge
pandas                    2.2.3                    pypi_0    pypi
pillow                    11.0.0                   pypi_0    pypi
pip                       24.2            py312h06a4308_0  
protobuf                  4.25.3          py312h12ddb61_0  
python                    3.12.4               h5148396_1  
python-dateutil           2.9.0.post0              pypi_0    pypi
pytz                      2024.2                   pypi_0    pypi
re2                       2022.04.01           h27087fc_0    conda-forge
readline                  8.2                  h5eee18b_0  
scikit-image              0.24.0                   pypi_0    pypi
scikit-learn              1.5.2                    pypi_0    pypi
scipy                     1.14.1                   pypi_0    pypi
setuptools                75.1.0          py312h06a4308_0  
six                       1.16.0             pyh6c4a22f_0    conda-forge
sqlite                    3.45.3               h5eee18b_0  
sympy                     1.13.1                   pypi_0    pypi
tensorboard               2.18.0             pyhd8ed1ab_0    conda-forge
tensorboard-data-server   0.7.0           py312h52d8a92_1  
threadpoolctl             3.5.0                    pypi_0    pypi
tifffile                  2024.9.20                pypi_0    pypi
tk                        8.6.14               h39e8969_0  
torch                     2.5.1                    pypi_0    pypi
torchvision               0.20.1                   pypi_0    pypi
tqdm                      4.67.0                   pypi_0    pypi
triton                    3.1.0                    pypi_0    pypi
typing-extensions         4.12.2                   pypi_0    pypi
tzdata                    2024.2                   pypi_0    pypi
werkzeug                  3.1.3              pyhff2d567_0    conda-forge
wheel                     0.44.0          py312h06a4308_0  
xz                        5.4.6                h5eee18b_1  
zipp                      3.21.0             pyhd8ed1ab_0    conda-forge
zlib                      1.2.13               h5eee18b_1
```

### Model Performance Metrics (Experimental)

```
| Task      | AUC      | Accuracy | Sensitivity | Specificity |
|-----------|----------|----------|-------------|-------------|
| ACL       | 0.963    | 0.883    | 0.815       | 0.939       |
| Meniscus  | 0.849    | 0.750    | 0.769       | 0.735       |
| Abnormal  | 0.952    | 0.858    | 0.979       | 0.400       |

```

# MRNet replication

This repository reproduces (to some extent) the results proposed in the paper ["Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet"](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699) by Bien, Rajpurkar et al.

## Data

Data must be downloaded from MRNet's [official website](https://stanfordmlgroup.github.io/competitions/mrnet/) and put anywhere into your machine. Then, edit the  ``` train_mrnet.sh``` script file by expliciting the full path to ```MRNet-v1.0``` directory into the ```DATA_PATH``` variable.

```
MRNet-v1.0/
│── train/                          # Training data  
│   ├── axial/                      # Axial plane MRI scans (.npy)  
│   ├── coronal/                    # Coronal plane MRI scans (.npy)  
│   ├── sagittal/                   # Sagittal plane MRI scans (.npy)  
│
│── valid/                          # Validation data  
│   ├── axial/                      # Axial plane MRI scans (.npy)  
│   ├── coronal/                    # Coronal plane MRI scans (.npy)  
│   ├── sagittal/                   # Sagittal plane MRI scans (.npy)
│
│── train-acl.csv                   # ACL tear labels (train set)  
│── train-abnormal.csv              # Abnormality labels (train set)  
│── train-meniscus.csv              # Meniscus tear labels (train set)  
│── train.csv                       # Overall train set metadata  
│── valid_mlabel.csv                # Multi-label classification file  
│── valid_abnormal.csv              # Abnormality labels (validation set)  
│── valid_acl.csv                   # ACL tear labels (validation set)  
│── valid_meniscus.csv              # Meniscus tear labels (validation set)  
│── valid.csv                       # Overall validation set metadata  
```

## Execution
To perform an experiment just run
```
bash train_mrnet.sh
```

This will train three models for each view (sagittal, axial, coronal) of each task (acl tear recognition, meniscal tear recognition, abnormalities recognition), for a total of 9 models. After that, a logistic regression model is trained, for each task, to combine the predictions of the different view models.
All checkpoints, training and validation logs, and results will be saved inside the ```experiment``` folder (it will be created if it doesn't exists).
 
Training and evaluation code is based on PyTorch and scikit-learn frameworks. Some parts are borrowed from https://github.com/ahmedbesbes/mrnet .
