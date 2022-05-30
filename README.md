# DS-WGAN

The official implementation of **Doubly Stochastic Generative Arrivals Modeling**



TODO:

- [ ] Make the Poisson simulator layer into a PyTorch self defined layer
- [x] Arrival epochs simulator
- [x] Sample CIR process
- [x] Run-through-queue
- [ ] Test each dataset argument.
- [ ] CUDA support.
- [ ] Speed up multi server queue, C++ reimplement or multi-thread in python.



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



## Run experiments

Usage

```
python main.py --dataset cir
python main.py --dataset uniform
python main.py --dataset bimodal
python main.py --dataset bikeshare
python main.py --dataset callcenter
```



| Dataset    | Verified code | Verified results |
| ---------- | :-----------: | :--------------: |
| CIR        |               |                  |
| Uniform    |               |                  |
| Bimodal    |               |                  |
| Bikeshare  |       ✅       |                  |
| Callcenter |       ✅       |                  |
| PGnorta    |               |                  |

