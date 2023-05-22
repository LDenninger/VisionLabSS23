import argparse
import os

import torch
import torchvision as tv
import copy
import optuna

import json

from pathlib import Path as P

import utils

import numpy as np

import torchgadgets as tg

from prettytable import PrettyTable

import matplotlib.pyplot as plt

import seaborn as sn

from dataset import KTHActioNDataset


###--- Run Information ---###
# These list of runs can be used to run multiple trainings sequentially.

#EXPERIMENT_NAMES = ['convnext_large']*3
#RUN_NAMES = ['norm_class', 'large_class', 'large_bn_class']

EXPERIMENT_NAMES = []
RUN_NAMES = []
EVALUATION_METRICS = ['accuracy', 'accuracy_top3', 'accuracy_top5', 'confusion_matrix', 'f1', 'recall', 'precision']
EPOCHS = [20]



###--- Training ---###
# This is the function used for training all our experiments.
# The experiments are structured into the "/experiments" directory, where all TensorBoard and PyTorch files can be found

def training(exp_names, run_names):
    assert len(exp_names) == len(run_names)
    for exp_name, run_name in zip(exp_names, run_names):
        ##-- Load Config --##
        # Load the config from the run directory
        # All interactions with the experiments directory should be performed via the utils package
        # Load config for the model
        config = utils.load_config_from_run(exp_name, run_name)
        config['num_iterations'] = config['dataset']['train_size'] // config['batch_size']

        tg.tools.set_random_seed(config['random_seed'])

        ##-- Load Dataset --##
        # Simply load the dataset using TorchGadgets and define our dataset to apply the initial augmentations
        train_dataset = KTHActioNDataset(dataset_name='kth_actions', split='train', transforms=config['pre_processing'])
        test_dataset = KTHActioNDataset(dataset_name='kth_actions', split='test', transforms=config['pre_processing'])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)

        # Since we work with sequential data we apply the data augmentations to the sequences
        # We only apply a flattening of the labels to only ahve a leading dimension of batch_size*sequence_length
        config['pre_processing'] = [{
                "type": "label_flatten",
                "start_dim": 0,
                "end_dim": 1,
                "train": True,
                "eval": True
            }]
        ##-- Logging --##
        # Directory of the run that we write our logs to
        log_dir = os.path.join(os.getcwd(),'experiments', exp_name, run_name, 'logs')
        checkpoint_dir = os.path.join(os.getcwd(), 'experiments', exp_name, run_name, 'checkpoints')

        # Explicitely define logger to enable TensorBoard logging and setting the log directory
        logger = tg.logging.Logger(log_dir=log_dir, checkpoint_dir=checkpoint_dir, model_config=config, save_internal=True)

        tg.training.trainNN(config=config, logger=logger, train_loader=train_loader, test_loader=test_loader, return_all=False)



###--- Evaluation ---###
# This is the function used for evaluation.
# The different implementations of the evaluation metrics can be found in the TorchGadgets package
# The structure of this function is close to the training function.

def evaluation(exp_names, run_names, verbose=True):
    assert len(exp_names)==len(run_names)
    
    # Verbose output
    if verbose:
        print(f'\n======---- Evaluation ----======')
        print(f' Experiment Names: {exp_names}')
        print(f' Run Names: {run_names}')
        print(f' Evluation Metrics: {EVALUATION_METRICS}\n')
        table = PrettyTable()
        field_names = ['Exp. Name', 'Run Name']
        for m in EVALUATION_METRICS:
            if m!='confusion_matrix':
                field_names.append(m)
        table.field_names = field_names

    for i, (exp_name, run_name) in enumerate(zip(exp_names, run_names)):
        ##-- Load Config --##
        load_augm_config_train = utils.load_config('augm_train_preLoad')
        load_augm_config_test = utils.load_config('augm_test_preLoad')
        config = utils.load_config_from_run(exp_name, run_name)
        config['num_iterations'] = config['dataset']['train_size'] // config['batch_size']

        tg.tools.set_random_seed(config['random_seed'])
        ##-- Load Dataset --##
        data = tg.data.load_dataset('oxfordpet')
        train_dataset = data['train_dataset']
        test_dataset = data['test_dataset']
        train_dataset = tg.data.ImageDataset(dataset=train_dataset, transforms=load_augm_config_train)
        test_dataset = tg.data.ImageDataset(dataset=test_dataset, transforms=load_augm_config_test, train_set=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True, num_workers=8)
        ##-- Logging --##
        log_dir = os.path.join(os.getcwd(),'experiments', exp_name, run_name, 'logs')
        checkpoint_dir = os.path.join(os.getcwd(), 'experiments', exp_name, run_name, 'checkpoints')
        vis_dir = os.path.join(os.getcwd(), 'experiments', exp_name, run_name, 'visualizations')
        logger = tg.logging.Logger(log_dir=log_dir, checkpoint_dir=checkpoint_dir, model_config=config, save_internal=True)
        data_augmentor = tg.data.ImageDataAugmentor(config['pre_processing'])

        ##-- Model Loading --##
        # Load the weights from the checkpoint and initialize the model
        model = tg.models.NeuralNetwork(config['layers'])
        utils.load_model_from_checkpoint(exp_name, run_name, model, EPOCHS[i])

        ##-- Run Evaluation --##
        evaluation_result = tg.evaluation.run_evaluation(model, data_augmentor, test_loader, config, evaluation_metrics=EVALUATION_METRICS)

        # Extract data and set a prefix to not confuse the logged data with data logged during the training
        data = {}
        for k, v in evaluation_result.items():
            if k!='confusion_matrix':
                data[('evaluation/'+k)] = v
        # Log data
        logger.log_data(0, data)

        conf_dir = os.path.join(vis_dir, 'confusion_matrix.png')
        fig, ax = plt.subplots(figsize=(18,16))
        sn.heatmap(evaluation_result['confusion_matrix'][0], annot=True, linewidths=.5, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        fig.savefig(conf_dir)

        if verbose:
            data = [m[0] for k, m in data.items()]
            data = [exp_name, run_name] + data
            table.add_row(data)
    if verbose:
        print(table)


###--- Data Preperation ---###
# This is the function used for data preparation.
# We load the data from the shared filesystem and define the splits manually to make data loadign easier.

##-- Hyperparameters --##
# We take the  training/validation/test splits according to: https://github.com/tejaskhot/KTH-Dataset/
train = [11, 12, 13, 14, 15, 16, 17, 18]
validation =[19, 20, 21, 23, 24, 25, 1, 4]
test = [22, 2, 3, 5, 6, 7, 8, 9, 10]

shared_path = '/home/nfs/inf6/data/datasets/kth_actions'

ind_to_action = {
    0: 'boxing',
    1: 'handclapping',
    2: 'handwaving',
    3: 'jogging',
    4: 'running',
    5: 'walking'
}

ind_to_ds_split = {
    0: 'train',
    1: 'validation',
    2: 'test'
}

def prepare_data(dataset_name, val_as_train=False, seq_length=20, overlap=False, dropLast=True, overlap_stride=1):


    print(f'\n======---- Dataset Preparation ----======\n')
    print(f' Dataset Name: {dataset_name}')
    print(f' Sequence Length: {seq_length}')
    print(f' Overlap: {overlap}')
    print(f' Overlap Stride: {overlap_stride}')
    print(f' Drop Last: {dropLast}')
    print(f' Validation as Train: {val_as_train}\n')

    ds_meta = {
        'name': dataset_name,
        'val_as_train': val_as_train,
        'seq_length': seq_length,
        'overlap': overlap,
        'overlap_stride': overlap_stride,
    }

    # Paths to save the dataset to
    dataset_path = P(os.getcwd()) / 'data' / dataset_name
    train_path = dataset_path / 'train'
    val_path = dataset_path / ('validation' if not val_as_train else 'train') 
    test_path = dataset_path / 'test'


    #if os.path.exists(str(dataset_path)):
    #    print(f'Dataset already exists at {dataset_path}')
    #    return False
    
    try:
        os.makedirs(str(dataset_path))
        os.makedirs(str(train_path))
        os.makedirs(str(val_path))
        os.makedirs(str(test_path))
    except Exception as e:
        print(f'Failed to create the dataset directory at {dataset_path}:')
        print(str(e))


    
    for ds_split_ind, dataset_split_ids in enumerate([train, validation, test]):
        # Sequence id within the dataset split
        seq_id = 0
        # Capture meta information about the dataset
        meta_info = {}
        labels = []
        for act_ind, action in ind_to_action.items():
            action_path = P(shared_path) / 'processed' / action
            for person_id in dataset_split_ids:
                for sc_id in range(1,5):
                    # Meta information

                    vid_path = action_path / f'person{str(person_id).zfill(2)}_{action}_d{sc_id}'
                    if not os.path.exists(str(vid_path)):
                        print(f'Video missing: person{str(person_id).zfill(2)}_{action}_d{sc_id}\n')
                        continue


                    n_frames = len([entry for entry in os.listdir(str(vid_path)) if os.path.isfile(str(vid_path / entry))])

                    # Create the splits that define the sequences
                    # Create overlapping sequences with a stride defined by overlap_stride
                    if overlap:
                        split_starts = np.arange(1, n_frames+1, overlap_stride)
                        split_ends = split_starts + seq_length-1
                        if dropLast:
                            split_ends = split_ends[:-(seq_length//overlap_stride)]
                        else:
                            split_ends[-(seq_length//overlap_stride):] = n_frames
                    # Create independent sequences
                    else:
                        split_starts = np.arange(1, n_frames+1, seq_length)
                        split_ends = split_starts[1:] -1
                        split_ends = np.append(split_ends, n_frames)
                        # Drop last sequence if the sequence length is shorter than the pre-defined sequence length
                        if dropLast and (split_ends[-1]-split_starts[-1]!= seq_length-1):
                            split_starts = split_starts[:-1]
                            split_ends = split_ends[:-1]
                    
                    # Load the frames corresponding to the given video
                    images = []
                    for ind in range(1, n_frames+1):
                        frame_path = vid_path / f'image-{str(ind).zfill(3)}_64x64.png'
                        try:
                            images.append(tv.io.read_image(str(frame_path)))
                        except Exception as e:
                            print(f'Failed to read frame {str(frame_path)}:')
                            print(str(e))
                            continue
                    
                    # Iterate over the splits and save sequence to tensor file
                    for split_start, split_end in zip(split_starts, split_ends):
                        save_path = dataset_path / ind_to_ds_split[ds_split_ind] / f'sequence_{str(seq_id).zfill(5)}.pt'
                        seq_ten = torch.stack(images[split_start-1:split_end], dim=0)
                        # Save sequence as stacked tensor of frames
                        try:
                            torch.save(seq_ten, str(save_path))
                        except Exception as e:
                            print(f'Failed to save sequence {str(save_path)}: \n')
                            print(str(e))
                            continue
                        # Log meta information
                        labels.append(act_ind)
                        meta_info[seq_id] ={
                            'start_id': str(split_start),
                            'end_id': str(split_end),
                            'label': str(action),
                            'label_id': str(act_ind),
                            'person_id': str(person_id),
                            'scene_id': str(sc_id)

                        }
                        print(f'Dataset: {ind_to_ds_split[ds_split_ind]}, \tSeq. Id: {str(seq_id)}, \tAction: {action}, \tPerson: {str(person_id)}, \tVideo: person{str(person_id).zfill(2)}_{action}_d{sc_id}', end='\r')
                        seq_id += 1

        # Save meta information for the dataset split
        meta_info['length'] = seq_id
        with open(str(dataset_path / ind_to_ds_split[ds_split_ind] /'meta.json'), 'w') as f:
            json.dump(meta_info, f, indent=4)
        # Save the labels for the dataset split
        labels = torch.Tensor(labels)
        torch.save(labels, str(dataset_path / ind_to_ds_split[ds_split_ind] /'labels.pt'))

        print(f'Successfully created {ind_to_ds_split[ds_split_ind]} dataset to: {str(dataset_path / ind_to_ds_split[ds_split_ind])}')
        print(f'Dataset length: {str(seq_id)}')

        ds_meta[f'{ind_to_ds_split[ds_split_ind]}_length'] = str(seq_id)
        ds_meta[f'{ind_to_ds_split[ds_split_ind]}_pid'] = [str(i) for i in dataset_split_ids]
        
    with open(str(dataset_path / 'dataset_meta.json'), 'w') as f:
        json.dump(ds_meta, f, indent=4)

    print(f'Dataset preperation finished...')






if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    # Flags to signal which function to run
    argparser.add_argument('--train', action='store_true', default=False, help='Train the model')
    argparser.add_argument('--evaluate', action='store_true', default=False, help='Evaluate the model')
    argparser.add_argument('--tuning', action='store_true', default=False, help='Tune the hyperparameters')

    argparser.add_argument('--prep_data', action='store_true', default=False, help='Prepare the dataset')

    argparser.add_argument('--init_exp', action='store_true', default=False, help='Initialize a new experiment')
    argparser.add_argument('--init_run', action='store_true', default=False, help='Initialize a new run')

    argparser.add_argument('--copy_conf', action='store_true', default=False, help='Load a configuration file to run')

    argparser.add_argument('--clear_logs', action='store_true', default=False, help='Clear the logs of a given run')

    # Hyperparameters 
    argparser.add_argument('-exp', type=str, default=None, help='Experiment name')
    argparser.add_argument('-run', type=str, default=None, help='Run name')
    argparser.add_argument('-conf', type=str, default=None, help='Config name')
    argparser.add_argument('-data', type=str, default=None, help='Dataset name')

    # Additional parameter
    argparser.add_argument('-n', type=int, default=None)
    
    ##-- Function Calls --##
    # Here we simply determien which function to call and how to set the experiment and run name

    args = argparser.parse_args()

    if args.prep_data:
        assert args.data is not None, 'Please provide a dataset name'
        prepare_data(dataset_name=args.data)


    # If no experiment or run name is provided, the environment variables defining these have to be set
    if args.init_exp:
        assert (args.exp is not None or 'CURRENT_EXP' in os.environ), 'Please provide an experiment name'
        exp_name = args.exp if args.exp is not None else os.environ.get('CURRENT_EXP')
        utils.create_experiment(exp_name)
    
    if args.init_run:
        assert (args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ), 'Please provide an experiment and run name'
        exp_name = args.exp if args.exp is not None else os.environ.get('CURRENT_EXP')
        run_name = args.run if args.run is not None else os.environ.get('CURRENT_RUN')
        utils.create_run(exp_name, run_name)
    
    if args.copy_conf:
        assert (args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ) and args.conf is not None, 'Please provide an experiment and run name and the name of the config file'
        exp_name = args.exp if args.exp is not None else os.environ.get('CURRENT_EXP')
        run_name = args.run if args.run is not None else os.environ.get('CURRENT_RUN')
        config_name = args.conf if args.conf is not None else os.environ.get('CURRENT_CONFIG')
        utils.load_config(exp_name, run_name, config_name)
    if args.clear_logs:
        assert (args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ) and args.conf is not None, 'Please provide an experiment and run name and the name of the config file'
        exp_name = args.exp if args.exp is not None else os.environ.get('CURRENT_EXP')
        run_name = args.run if args.run is not None else os.environ.get('CURRENT_RUN')
        utils.clear_logs(exp_name, run_name)


    if args.train:
        assert ((args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ)) or (len(EXPERIMENT_NAMES)!=0 and len(RUN_NAMES)!=0), 'Please provide an experiment and run name'
        if len(EXPERIMENT_NAMES) != 0:
            assert len(EXPERIMENT_NAMES) == len(RUN_NAMES), 'Length of experiment and run list must be the same'
            exp_name = EXPERIMENT_NAMES
            run_name = RUN_NAMES
        else:       
            exp_name = [args.exp] if args.exp is not None else [os.environ.get('CURRENT_EXP')]
            run_name = [args.run] if args.run is not None else [os.environ.get('CURRENT_RUN')]
        training(exp_name, run_name)

    if args.evaluate:
        assert ((args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ)) or (len(EXPERIMENT_NAMES)!=0 and len(RUN_NAMES)!=0), 'Please provide an experiment and run name'
        if len(EXPERIMENT_NAMES) != 0:
            assert len(EXPERIMENT_NAMES) == len(RUN_NAMES), 'Length of experiment and run list must be the same'
            exp_name = EXPERIMENT_NAMES
            run_name = RUN_NAMES
        else:       
            exp_name = [args.exp] if args.exp is not None else [os.environ.get('CURRENT_EXP')]
            run_name = [args.run] if args.run is not None else [os.environ.get('CURRENT_RUN')]
        evaluation(exp_name, run_name)
    