## RUN AFTER FIRST CLONING ##

# Initialize the submodule containing the logger and experiment manager
git submodule init
git submodule update
cd experiments
git checkout master
git pull

# Aliases to initialize conda environments for different machines
# CUDA 11
alias init_c11='conda create -n vision_lab python=3.9 optuna jupyter yaml ipdb scipy matplotlib tensorboard tqdm pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -c conda-forge; conda activate vision_lab;'