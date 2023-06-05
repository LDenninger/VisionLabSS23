# Experiment Template
This is a template directory structure for experiments with deep learning models.
It is intended to serve as the base for a new project.
Especially it is designed to be used with the [TorchGadgets](https://github.com/LDenninger/TorchGadgets), which provides a logger to log the experiments and further functionalities.
The API of TorchGadgets aims at working completely with config files which are separately handled within this template.

# Installation
Install conda environment:
```
conda create -n vision_lab python=3.9 optuna jupyter yaml ipdb scipy matplotlib torchmetrics tensorboard tqdm pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -c conda-forge;
```

Install TorchGadgets: 
```
git clone git@github.com:LDenninger/TorchGadgets.git
python setup.py install
```
There might be a few more pacakges that have to be manually installed

# Quick Start
Unpack data to tensor files: `python prepare_data.py`<br />
Source environment : `source env.sh` <br />
Set experiment: `setexp [exp. name]`<br />
Set run: `setrun [run name]`<br />
Initialize new experiment: `iexp -exp [exp. name]`<br />
Initialize new run: `irun -exp [exp. name] -run [run name]`<br />
Train the model: `train -exp [exp. name] -run [run name]`<br />
If a experiment and/or run is set, it is not required to pass the name as an argument.
