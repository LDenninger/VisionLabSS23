# Vision Lab SS23
Small framework to train, evaluate and log different models and to share solutions.

## Dependencies
The framework requires the standard PyTorch libraries. See `./init.bash` for a working conda environment running on Cuda11. <br />
First Initialization: `source init.bash` <br />
Run: ` init_c11` to create a working conda environment for most cuda machines. <br />

## Structure
    .
    ├── data                            # Directory for saving the datasets
    ├── experiments                     # Directory holding the experiments/config files/ and util functions for logging
    │   ├── config                      # Default configuration files
    │   ├── exp_data                    # Data from experiments
    │   ├── [exp. name]
    │   │    ├── [run name]
    │   │    │    ├── checkpoints      # Model checkpoints
    │   │    │    ├── logs             # Tensoboard and further log files
    │   │    │    ├── plots            # Plots
    │   │    │    ├── visualizations   # Visualizations
    │   │    │    └── config.yaml      # Config file for the run
    │   │    └── ...                   
    │   └── ...
    ├── models                          # Different models and training scripts.
    ├── utils                           # Utility functions
    └── ...


## Usage
Before working, source experiment environment: `source env.bash` <br />
Set current experiment: `setexp [experiment name]` <br />
Set current run: `setrun [run name]` <br />
Check current setup: `setup` <br />
Initialize new run: `irun -config [config name]`
 - Config file needs to be in `experiments/config` 
Run a specified task: `python run.py -task [task ID]` <br />