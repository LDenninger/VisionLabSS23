# Start conda virtual environment
if conda info --envs | grep -q vision_lab; then
    conda activate vision_lab
else
    echo Conda environment, vision_lab, does not exists yet. See ./init.bash for fixed generation commands.
fi

alias init_c11='conda create -n vision_lab python=3.9 optuna jupyter yaml ipdb scipy matplotlib tensorboard tqdm pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -c conda-forge; conda activate vision_lab;'
