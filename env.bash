conda activate vision_lab
# Define environment variable holding the experiment directory for logging
EXP_DIRECTORY_NAME='experiments'

EXP_DIRECTORY_PATH="$(cd "$EXP_DIRECTORY_NAME"; pwd)"

export EXP_DIRECTORY=$EXP_DIRECTORY_PATH
echo Experiment directory set to "${EXP_DIRECTORY}"
# Source shortcuts for experiment management
source $EXP_DIRECTORY/setup.bash
