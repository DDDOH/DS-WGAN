# DS-WGAN

The official implementation of **Doubly Stochastic Generative Arrivals Modeling**



TODO:

- [ ] set appropriate server scheduling for multi server queue experiment (maybe a individual ipynb notebook).
- [ ] Verify results.
- [ ] erase the warning when running "python setup.py install"
- [ ] add args message for des_cpp
- [x] CUDA support.
- [x] Make the Poisson simulator layer into a PyTorch self defined layer
- [x] Arrival epochs simulator
- [x] Sample CIR process
- [x] Run-through-queue
- [x] Speed up multi server queue, C++ reimplement or multi-thread in python.

## Setup environment

We recommend using conda environment.

```
conda create --name dswgan
conda install -c anaconda scipy -y
conda install -c conda-forge matplotlib -y
pip install progressbar2
conda install -c conda-forge colored -y
conda install pytorch -c pytorch -y
pip install geomloss
conda install -c anaconda seaborn -y
```

To build and install the C++ implementad descrete event simulation library (mainly for simulating multi-server queue), we assume you have installed the adequate C++ compiler. After that, activate the conda environment, then
```
conda install -c conda-forge pybind11 -y
cd core/des/des_cpp
python setup.py install
```
which will compile and install the discrete event simulation library to your conda environment.

## Run experiments

Usage

```
python main.py --dataset cir
python main.py --dataset uniform
python main.py --dataset bimodal
python main.py --dataset bikeshare
python main.py --dataset callcenter
python main.py --dataset pgnorta
```



| Dataset    | Verified code | Verified results |
| ---------- | :-----------: | :--------------: |
| CIR        |       ✅       |                  |
| Uniform    |       ✅       |                  |
| Bimodal    |       ✅       |                  |
| Bikeshare  |       ✅       |                  |
| Callcenter |       ✅       |                  |
| PGnorta    |       ✅       |                  |

