# LDP: A Local Diffusion Planner for Efficient Robot Navigation and Collision Avoidance

This repository is a sub-repository of the [LDP](https://github.com/WenhaoYu1998/LDP) repository, which contains the expert policy training and testing code for the paper "LDP: A Local Diffusion Planner for Efficient Robot Navigation and Collision Avoidance". The paper is accepted to IROS 2024.

### 📰 Requirements

- Ubunru 20.04
- ROS Noetic
- expert_enviroment.yaml

### 💿 Installation

1. Install simulation environment, please refer to [drlnav_env](./drlnav_env/README.md) for details.

2. Install python dependencies:

We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 
```console
$ mamba env create -f expert_environment.yaml
```

but you can use conda as well: 
```console
$ conda env create -f expert_environment.yaml
```

### ✍ Usage

#### Start the simulation environment
```console
# Open a new terminal
[LDP_SAC_expert]$ cd drlnav_env
[drlnav_env]$ source devel/setup.bash
[drlnav_env]$ roslaunch img_env test.launch
```

#### Training the expert policy
```console
# Open another new terminal
[LDP_SAC_expert]$ cd drlnav_env
[drlnav_env]$ source devel/setup.bash # NOTE: This source instruction is very important.
[drlnav_env]$ cd ../LDP_expert
[LDP_expert]$ chmod +x run_drl_train.sh
[LDP_expert]$ ./run_drl_train.sh
```
#### Testing the expert policy
```console
[LDP_expert]$ chmod +x run_drl_test.sh
[LDP_expert]$ ./run_drl_test.sh
```
### 🙏 Acknowledgement
+ Our simulation environment is based on the [drlnav_env](https://github.com/DRL-Navigation/img_env) of IROS2021.
+ Our SAC algorithm implementation is based on the [AMBS](https://github.com/jianda-chen/AMBS) of ICLR2022.